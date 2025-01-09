import torch
from torch import nn
import pytorch_lightning as L
import traceback
import math
import random

from model.model_utils import *
from model.energy_based_transformer import Transformer, ModelArgs

#TODO remove CV stuff, clean code and make just for ARC

class EBWM_ARC(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        if(self.hparams.energy_loss_fn=='l1_loss'):
            self.energy_loss_fn = nn.L1Loss()
        elif(self.hparams.energy_loss_fn=='smooth_l1_loss'):
            self.energy_loss_fn = nn.SmoothL1Loss()
        else:#default
            self.energy_loss_fn = nn.MSELoss()

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func
        self.finished_warming_up = False

        self.image_processor = load_pretrained_backbone(self.hparams.backbone_type, self.hparams.vit_backbone_size, self.hparams.embedding_dim) # dtype 32 by default
        if self.hparams.backbone_type == "dinov2":
            del self.image_processor._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues

        for param in self.image_processor.parameters():
            param.requires_grad = False

        transformer_args = ModelArgs(dim = self.hparams.embedding_dim, n_layers = self.hparams.num_transformer_blocks, n_heads = self.hparams.multiheaded_attention_heads, max_batch_size = self.hparams.batch_size_per_device, max_seq_len=self.hparams.context_length+1, weight_initialization = self.hparams.weight_initialization_method)
        self.transformer = Transformer(params=transformer_args)

        if self.hparams.mlp_hidden_layers >= 1:
            self.energy_predictor = MLP(self.hparams.embedding_dim, int(self.hparams.embedding_dim*self.hparams.mlp_dim_multiplier), 1, self.hparams.mlp_dropout, self.hparams.mlp_layer_norm, self.hparams.mlp_layers)
            if self.hparams.energy_loss_coeff == 0.0 and self.hparams.out_of_bounds_loss_coeff == 0.0:
                self.energy_predictor.layers[-1].bias.requires_grad = False # if we are just using rec loss then bias term of energy_predictor does not matter. 
                # this was causing a "RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step. If this is intentional, you must enable the detection of unused parameters in DDP, either by setting the string value `strategy='ddp_find_unused_parameters_true'` or by setting the flag in the strategy with `strategy=DDPStrategy(find_unused_parameters=True)`." error 
        else:
            bias = False if (self.hparams.energy_loss_coeff == 0.0 and self.hparams.out_of_bounds_loss_coeff == 0.0) else True # see issue above for why this is done
            self.energy_predictor = nn.Linear(self.hparams.embedding_dim, 1, bias=bias) # just a single linear layer to pred energy instead
        init_whole_model_weights(self.energy_predictor, self.hparams.weight_initialization_method)
        
        if self.hparams.dataset_name != "arc-agi":
            self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0)
        else:
            self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
            # self.grid_dim_width = nn.Embedding(num_embeddings=30, embedding_dim=self.hparams.embedding_dim // 2)
            # init_whole_model_weights(self.grid_dim_width, self.hparams.weight_initialization_method)
            # self.grid_dim_height = nn.Embedding(num_embeddings=30, embedding_dim=self.hparams.embedding_dim // 2)
            # init_whole_model_weights(self.grid_dim_height, self.hparams.weight_initialization_method)
            
        
        # DEBUGGING CODE ################################################################################################################################################
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update
            

    
    def forward(self, x):        
        #real embeddings here are embeddings extracted from the real video, predicted_embeddings are the predictions (initial pred is often random)    
        if self.hparams.dataset_name != "arc-agi":
            context_length = x.shape[1]
            print("context_length", context_length)
            x = x.reshape(-1, *x.shape[2:]) # bs*s, c, w, h
            real_embeddings = self.image_processor(x)
        else:
            sequence_mask = x['sequence_mask'] # B, S-1 (minus one since only pred next one)
            real_embeddings = x['data'] # B, S, 900
            context_length = real_embeddings.shape[1]
            
        real_embeddings = real_embeddings.reshape(-1, context_length, self.hparams.embedding_dim) # bs, s, d
        
        predicted_embeddings = self.corrupt_embeddings(real_embeddings) # B, S, D ; is called predicted bc initial prediction may be poor
        predicted_embeddings = predicted_embeddings[:, :-1, :] # cannot use one of embeddings in seq since need an embed to condition on. this could be first but just in case condition_on_most_recent_embedding is set it needs to remove the last
        real_embeddings_comparison =  real_embeddings[:, 1:, :] # for geting energy loss and rec loss when comparing to predicted_embeddings so compare frame by frame, extract in same way
        real_embeddings_input = real_embeddings[:, :-1, :] #similar to predictions but is all but last frame since need first n frames to be real and last n frames to be predicted
        
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        
        
        reconstruction_loss = 0.0
        final_reconstruction_loss = 0.0
        energy_loss = 0.0
        out_of_bounds_loss = 0.0
        
        with torch.set_grad_enabled(True): # set to true for validation since grad would be off
            for mcmc_step in range(self.hparams.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                
                if self.finished_warming_up and self.hparams.langevin_dynamics_noise != 0: # only use langevyn dynamics once model is warmed up
                    ld_noise = torch.randn_like(predicted_embeddings.detach()) * langevin_dynamics_noise_std # langevin dynamics
                    predicted_embeddings = predicted_embeddings + ld_noise
                
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, 2(S-1), D
                
                refined_embeddings = self.transformer(all_embeddings, start_pos = 0, learning = True) # is B, 2(S-1), D; checked and there are no in place ops
                refined_embeddings = refined_embeddings[:, predicted_embeddings.shape[1]:] # B, (S-1), D; loses real_embeddings portion
                
                refined_embeddings = refined_embeddings.reshape(-1, self.hparams.embedding_dim) # flatten before MLP
                energy_preds = self.energy_predictor(refined_embeddings) + 0.5 # B*(S-1), 1; +0.5 is since will be centered around 0.5 since energy ranges from 0 to 1
                
                if self.hparams.reconstruction_coeff == 0.0: # in this case dont create graph since reconstruction wont supervise weights
                    predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [predicted_embeddings])[0]
                else:
                    predicted_embeds_grad = torch.autograd.grad([energy_preds.sum()], [predicted_embeddings], create_graph=True, retain_graph=True)[0] #not retrain_graph defaults to create_graph value
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.hparams.mcmc_num_steps * alpha)
                    predicted_embeds_grad = scale_clamp(predicted_embeds_grad, -min_and_max, min_and_max)
                    
                if torch.isnan(predicted_embeds_grad).any() or torch.isinf(predicted_embeds_grad).any():
                    self.hparams.txt_logger.warning("NAN OR INF GRADIENTS DETECTED during MCMC - please investigate")
                    continue            
                    
                predicted_embeddings = predicted_embeddings - alpha * predicted_embeds_grad
                
                #loss calculations
                if self.hparams.reconstruction_coeff != 0.0:
                    
                    if self.hparams.dataset_name != "arc-agi":
                        reconstruction_loss += self.reconstruction_criterion(predicted_embeddings, real_embeddings_comparison)
                    else:
                        unmasked_loss = self.reconstruction_criterion(predicted_embeddings, real_embeddings_comparison)

                        seq_len = predicted_embeddings.size(1)
                        arc_mask = torch.ones(seq_len, dtype=torch.bool, device = self.device)
                        arc_mask[1::2] = False  # Set all odd indices to False, bc these correspond to the condition which cant predict
                        arc_mask[0] = False # Set index 0 to False since dont know pattern yet

                        batch_size = predicted_embeddings.size(0)
                        arc_mask = arc_mask.unsqueeze(0).expand(batch_size, -1) # BS, S-1

                        complete_mask = arc_mask & sequence_mask
                        complete_mask = complete_mask.unsqueeze(-1) #BS, S-1, 1; unmasked_loss.shape is BS, S-1, D

                        masked_loss = unmasked_loss * complete_mask # mask it
                        reconstruction_loss += masked_loss.sum() / (complete_mask.sum() * masked_loss.shape[2]) # compute loss by averaging
                    
                    if mcmc_step == (self.hparams.mcmc_num_steps - 1):
                        if self.hparams.dataset_name != "arc-agi":
                            final_reconstruction_loss = self.reconstruction_criterion(predicted_embeddings, real_embeddings_comparison).detach()
                        else:
                            final_reconstruction_loss = masked_loss.sum().detach() / (complete_mask.sum() * masked_loss.shape[2])
                            
                        reconstruction_loss = reconstruction_loss / self.hparams.mcmc_num_steps # normalize so is indifferent to number of mcmc steps
                                    
                if self.hparams.energy_loss_coeff != 0.0:
                    energy_labels = self.calc_distance(predicted_embeddings, real_embeddings_comparison)
                    if self.hparams.energy_loss_hinge == 0.0: # for doing margin based energy regressive pred, helps with inherent randomness in metric used for REBM
                        energy_loss += self.energy_loss_fn(energy_preds.squeeze(), energy_labels.reshape(-1)) # labels do not have grad, preds do; both after ops here will have (B*(S-1))
                    else:
                        energy_loss += hinged_mse_loss(energy_preds.squeeze(), energy_labels.reshape(-1), margin = self.hparams.energy_loss_hinge)
                        
                if self.hparams.out_of_bounds_loss_coeff != 0.0:
                    out_of_bounds_loss += calc_out_of_bounds_loss(energy_preds)
                
                #pure logging things (no function for training)
                if mcmc_step == 0:
                    initial_pred_energies = energy_preds.squeeze().mean().detach()
                if mcmc_step == (self.hparams.mcmc_num_steps - 1):
                    final_pred_energies = energy_preds.squeeze().mean().detach()
                    
                    
                                        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        
        total_loss = self.hparams.energy_loss_coeff * energy_loss +  self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.out_of_bounds_loss_coeff * out_of_bounds_loss
        #NOTE when returning losses make sure to detach things from comp graph
        if isinstance(energy_loss, torch.Tensor): #these are just in case are just using one or the other
            energy_loss = energy_loss.detach()
        if isinstance(reconstruction_loss, torch.Tensor):
            reconstruction_loss = reconstruction_loss.detach()
        if isinstance(out_of_bounds_loss, torch.Tensor):
            out_of_bounds_loss = out_of_bounds_loss.detach()
        return total_loss, self.hparams.energy_loss_coeff * energy_loss, self.hparams.reconstruction_coeff * reconstruction_loss, final_reconstruction_loss, self.hparams.out_of_bounds_loss_coeff * out_of_bounds_loss, initial_final_pred_energies_gap, predicted_embeddings.detach() # already detached final_rec_loss
            
    
    def corrupt_embeddings(self, embeddings):
        #corrpution ideas: all 0's, shifted temporally random num in 1, context_frames (so gets wrong time frame), shifted along seq axis (so gets wrong video frame), random gaussian noise, 
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            predicted_embeddings = embeddings.clone()
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_embeddings = torch.randn_like(embeddings)
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_embeddings = torch.zeros_like(embeddings) # default just condition on nothing. found it did not produce as good of reps. as random_noise
        else:
            raise ValueError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        return predicted_embeddings
    
    def calc_distance(self, pred_embeddings, gt_embeddings): # gt is ground truth
        with torch.set_grad_enabled(False): # dont want grad since dont want model to see how energy was calculated
            # both embed tensors have shape (B, (S-1), D)

            if self.hparams.embeddings_distance_fn == 'euclidean':
                raw_distance = torch.norm(gt_embeddings - pred_embeddings, dim=2, p=2)
            elif self.hparams.embeddings_distance_fn == 'manhattan':
                raw_distance = torch.norm(gt_embeddings - pred_embeddings, dim=2, p=1)
            elif self.hparams.embeddings_distance_fn == 'cosine':
                cos_sim = torch.nn.functional.cosine_similarity(gt_embeddings, pred_embeddings, dim=2)
                raw_distance = (-1 * cos_sim) + 1  # Convert similarity to distance
            else:
                raise ValueError(f"Invalid distance function specified: {self.hparams.embeddings_distance_fn}")

            # Average raw distance across the sequence length and make sure is non negative due to rounding
            min_val = torch.min(raw_distance).detach()
            adj_distance = torch.where(raw_distance < 0, raw_distance - min_val, raw_distance)
            if torch.any(adj_distance < 0):
                raise ValueError("no values should be less than zero, adjust above code")

            # Apply normalization transformations if needed
            if self.hparams.embeddings_distance_fn == 'normalized_euclidean':
                mean_distance = torch.mean(adj_distance)
                std_distance = torch.std(adj_distance)
                adj_distance = (adj_distance - mean_distance) / (std_distance + 1e-6)
            elif self.hparams.embeddings_distance_fn == 'cosine':
                if self.hparams.scale_cosine_sim_decay != 0: # just a way of scaling cosine sim that I found correlates to a good energy empirically
                    adj_distance = 1 - torch.exp(-1 * self.hparams.scale_cosine_sim_decay * adj_distance)
            return adj_distance
        
    def warm_up_finished(self):
        self.finished_warming_up = True
        self.langevin_dynamics_noise_std.requires_grad = self.hparams.langevin_dynamics_noise_learnable
        
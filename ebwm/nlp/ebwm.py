import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from ebwm.model_utils import *
from ebwm.energy_based_transformer import Transformer, ModelArgs


class EBWM_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
        self.vocab_size = len(tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        
        if not self.hparams.vocab_to_embed_uses_prob_dist: # if are not using the prob dist * embed as vocab to embed
            self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias = False, device = self.device) #NOTE this is ebwm special, since we want to input a prob dist and pred this prob dist but the transformer needs an embedding as input
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method)

        transformer_args = ModelArgs(dim = self.hparams.embedding_dim, n_layers = self.hparams.num_transformer_blocks, n_heads = self.hparams.multiheaded_attention_heads, max_batch_size = self.hparams.batch_size_per_device, max_seq_len=self.hparams.context_length+1, weight_initialization = self.hparams.weight_initialization_method)
        self.transformer = Transformer(params=transformer_args)
        
        self.energy_predictor = nn.Linear(self.hparams.embedding_dim, 1, bias = False, device = self.device)
        init_whole_model_weights(self.energy_predictor, self.hparams.weight_initialization_method)
        
        self.finished_warming_up = False
        
    def forward(self, x): # accepts input_ids as input
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        alpha = torch.clamp(self.alpha, min=0.0001)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V
                
        with torch.set_grad_enabled(True):
            for mcmc_step in range(self.hparams.mcmc_num_steps):
                if self.hparams.mcmc_no_chain:
                    predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V
                
                predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

                if self.hparams.normalize_initial_condition:
                    predicted_tokens = self.softmax(predicted_tokens)
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, 2*S, D
                
                refined_embeddings = self.transformer(all_embeddings, start_pos = 0, learning = True) # is B, 2*S, D; checked and there are no in place ops
                refined_embeddings = refined_embeddings[:, predicted_embeddings.shape[1]:] # B, S, D; loses real_embeddings portion
                refined_embeddings = refined_embeddings.reshape(-1, self.hparams.embedding_dim) # flatten before MLP
                energy_preds = self.energy_predictor(refined_embeddings) + 0.5 # B*S, 1; +0.5 is since will be centered around 0.5 since energy ranges from 0 to 1
                predicted_energies.append(energy_preds)
                
                predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=True, retain_graph=True)[0] #not retrain_graph defaults to create_graph value
                
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.hparams.mcmc_num_steps * alpha)
                    # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max) #TODO remove line and add back line above? not for now since saves memory
                    
                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")
                
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after  
                
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
                
                predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size)
                predicted_distributions.append(predicted_tokens_for_loss)        

        return predicted_distributions, predicted_energies

    def forward_loss_wrapper(self, x):
        input_ids = x['input_ids'].squeeze()[:, :-1]
        predicted_distributions, predicted_energies = self(input_ids)

        next_token_indices = x['input_ids'].squeeze()[:, 1:] # squeeze was to remove 1 on 2nd dim
        if self.hparams.training_type == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D

        reconstruction_loss = 0
        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, predicted_energies)):
            cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)
            reconstruction_loss += cce_loss
            if mcmc_step == (self.hparams.mcmc_num_steps - 1):
                ppl_loss = torch.exp(cce_loss).detach()
                final_reconstruction_loss = cce_loss.detach()
                reconstruction_loss = reconstruction_loss / self.hparams.mcmc_num_steps # normalize so is indifferent to number of mcmc steps
                
            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (self.hparams.mcmc_num_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()
        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        total_loss = self.hparams.reconstruction_coeff * reconstruction_loss

        log_dict = {
            'loss': total_loss,
            'final_step_loss': final_reconstruction_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings):
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), device = self.device) * self.hparams.gaussian_random_noise_scaling
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), device = self.device)
        else:
            raise NotImplementedError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        
        return predicted_tokens
    
    def warm_up_finished(self):
        if self.hparams.clamp_max_after_warm_up != 0.0:
            print(f"changing clamp value after warming up from {self.hparams.clamp_futures_grad_max_change} (see next line)")
            self.hparams.clamp_futures_grad_max_change = self.hparams.clamp_max_after_warm_up
            print(f"to the value {self.hparams.clamp_futures_grad_max_change}")
        self.finished_warming_up = True
        #can implement langevin here later
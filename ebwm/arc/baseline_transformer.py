import torch
from torch import nn
import pytorch_lightning as L
import traceback
import math
import random

from model.model_utils import *
from model.energy_based_transformer import Transformer, ModelArgs

##TODO remove CV stuff, clean code and make just for ARC

class Baseline_Transformer_ARC(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_processor = load_pretrained_backbone(self.hparams.backbone_type, self.hparams.vit_backbone_size, self.hparams.embedding_dim) # dtype 32 by default
        if self.hparams.backbone_type == "dinov2":
            del self.image_processor._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues

        for param in self.image_processor.parameters():
            param.requires_grad = False

        transformer_args = ModelArgs(dim = self.hparams.embedding_dim, n_layers = self.hparams.num_transformer_blocks, n_heads = self.hparams.multiheaded_attention_heads, max_batch_size = self.hparams.batch_size_per_device, max_seq_len=self.hparams.context_length, weight_initialization = self.hparams.weight_initialization_method)
        self.transformer = Transformer(params=transformer_args)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method)
        
        if self.hparams.dataset_name != "arc-agi":
            self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0)
        else:
            self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
        
        self.finished_warming_up = False
        


    def forward(self, x):  
        if self.hparams.dataset_name != "arc-agi":
            context_length = x.shape[1]
            print("context_length", context_length)
            x = x.reshape(-1, *x.shape[2:]) # bs*s, c, w, h
            embeddings = self.image_processor(x)
        else:
            sequence_mask = x['sequence_mask'] # B, S-1 (minus one since only pred next one)
            embeddings = x['data'] # B, S, 900
            context_length = embeddings.shape[1]
            print("context_length", context_length)
        
        embeddings = embeddings.reshape(-1, (context_length+1), self.hparams.embedding_dim) # BS, S+1, D; +1 is since need to load 1 extra frame for final pred
        input_embeddings = embeddings[:, :-1] # only input first S 
        next_embeddings = embeddings[:, 1:] # compare to last S (as make pred of next element)
        
        predicted_embeddings = self.transformer(input_embeddings, start_pos = 0, learning = True)
        predicted_embeddings = self.output(predicted_embeddings)
        
        if self.hparams.dataset_name != "arc-agi":
            loss = self.reconstruction_criterion(predicted_embeddings, next_embeddings)
        else:
            unmasked_loss = self.reconstruction_criterion(predicted_embeddings, next_embeddings)

            seq_len = predicted_embeddings.size(1)
            arc_mask = torch.ones(seq_len, dtype=torch.bool, device = self.device)
            arc_mask[1::2] = False  # Set all odd indices to False, bc these correspond to the condition which cant predict
            arc_mask[0] = False # Set index 0 to False since dont know pattern yet

            batch_size = predicted_embeddings.size(0)
            arc_mask = arc_mask.unsqueeze(0).expand(batch_size, -1) # BS, S-1

            complete_mask = arc_mask & sequence_mask
            complete_mask = complete_mask.unsqueeze(-1) #BS, S-1, 1; unmasked_loss.shape is BS, S-1, D

            masked_loss = unmasked_loss * complete_mask # mask it
            loss = masked_loss.sum() / (complete_mask.sum() * masked_loss.shape[2]) # compute loss by averaging
        return loss, predicted_embeddings.detach()
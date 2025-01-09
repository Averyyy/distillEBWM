import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models

import math
import random
from ebwm.model_utils import *
from ebwm.transformer import Transformer, ModelArgs



class Baseline_Transformer_CV(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0)

        self.image_processor = load_pretrained_backbone(self.hparams.backbone_type, self.hparams.vit_backbone_size, self.hparams.embedding_dim) # dtype 32 by default
        if self.hparams.backbone_type == "dinov2":
            del self.image_processor._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues

        for param in self.image_processor.parameters():
            param.requires_grad = False

        transformer_args = ModelArgs(dim = self.hparams.embedding_dim, n_layers = self.hparams.num_transformer_blocks, n_heads = self.hparams.multiheaded_attention_heads, max_batch_size = self.hparams.batch_size_per_device, max_seq_len=self.hparams.context_length, weight_initialization = self.hparams.weight_initialization_method)
        self.transformer = Transformer(params=transformer_args)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method)
        
        self.finished_warming_up = False

    def forward(self, embeddings): # accepts embeddings as input, predicts next embeddings. assumes embeddings is of shape BS, S, D
        predicted_embeddings = self.transformer(embeddings, start_pos = 0, learning = True)
        predicted_embeddings = self.output(predicted_embeddings)
        
        return predicted_embeddings
    
    def forward_loss_wrapper(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.reshape(-1, *x.shape[2:]) # BS*(S+1), C, W, H
        embeddings = self.image_processor(x)
        embeddings = embeddings.reshape(batch_size, seq_length, self.hparams.embedding_dim) # BS, S+1, D
        input_embeddings = embeddings[:, :-1] # only input first S; so BS, S, D

        predicted_embeddings = self(input_embeddings)

        next_embeddings = embeddings[:, 1:] # compare to last S (as make pred of next element); so BS, S, D
        loss = self.reconstruction_criterion(predicted_embeddings, next_embeddings) # predicted_embeddings has grad and is BS, S, D; next_embeddings is same shape and does not have grad

        log_dict = {
            'loss': loss,
        }
        return log_dict
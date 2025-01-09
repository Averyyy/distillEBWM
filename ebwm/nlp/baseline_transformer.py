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
from ebwm.transformer import Transformer, ModelArgs


class Baseline_Transformer_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
        self.vocab_size = len(tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)

        transformer_args = ModelArgs(dim = self.hparams.embedding_dim, n_layers = self.hparams.num_transformer_blocks, n_heads = self.hparams.multiheaded_attention_heads, max_batch_size = self.hparams.batch_size_per_device, max_seq_len=self.hparams.context_length, weight_initialization = self.hparams.weight_initialization_method)
        self.transformer = Transformer(params=transformer_args)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.vocab_size, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method)
        
        self.finished_warming_up = False
        

    def forward(self, x): # accepts input_ids as input
        embeddings = self.embeddings(x) # x here is input_ids
        
        predicted_embeddings = self.transformer(embeddings, start_pos = 0, learning = True) # BS, S, D
        predicted_logits = self.output(predicted_embeddings) #BS, S, vocab_size
        predicted_distribution = self.log_softmax(predicted_logits).reshape(-1, self.vocab_size) # BS*S, V; reshape since preds for nll should be 2d
        return predicted_distribution
        

    def forward_loss_wrapper(self, x):
        input_ids = x['input_ids'].squeeze()[:, :-1] # x['input_ids'] shape is BS, S+1, only input first S--remove next tokens

        predicted_distribution = self(input_ids)
        
        next_token_indices = x['input_ids'].squeeze()[:, 1:] # squeeze was to remove 1 on 2nd dim
        if self.hparams.training_type == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
        
        cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=self.tokenizer_pad_token_id)
        ppl_loss = torch.exp(cce_loss).detach()

        log_dict = {
            'loss': cce_loss,
            'perplexity': ppl_loss
        }
        return log_dict
    
    
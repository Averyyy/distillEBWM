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
import numpy as np
from functools import partial
from PIL import Image
from torchvision.transforms import ToPILImage
from datetime import datetime
import torch.distributed as dist


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out  # Add the residual connection

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.layers.append(ResidualBlock(hidden_size, dropout_rate))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.layers.append(ResidualBlock(hidden_size, dropout_rate))
        else:
            self.layers.append(nn.Linear(hidden_size, final_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def calc_out_of_bounds_loss(energy): # gives loss for < 0 or > 1
    lower_bound_loss = torch.abs(energy)
    upper_bound_loss = torch.abs(energy - 1)
    loss = torch.where(energy < 0, lower_bound_loss, 
                    torch.where(energy > 1, upper_bound_loss, torch.zeros_like(energy)))
    loss = torch.mean(loss)
    
    return loss

def log_pred_futures(futures, device, dataset_name, i, denormalize):
    denormalized_futures = denormalize(futures.clone(), dataset_name, device = device)

    to_pil = ToPILImage()
    for b in range(denormalized_futures.shape[0]):  # Loop over the batch size
        if b % 16 == 0:
            for s in range(denormalized_futures.shape[1]):  # Loop over the sequence length
                frame_to_save = to_pil(denormalized_futures[b, s].cpu())  # Extract a frame (C x W x H)
                
                # Save the image
                current_time = datetime.now().strftime("%H_%M_%S")
                frame_to_save.save(f"./logs/debug/mcmc_futures/{current_time}_batch_{b}_seq_{s}_dev_{device}_iter_{i}.png")

def denormalize(tensor, dataset_name, device, custom_normalization):
    tensor = tensor.clone().detach()

    # Define default normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_mean = torch.tensor(default_mean, device=device).view(1, 1, 3, 1, 1)
    default_std = torch.tensor(default_std, device=device).view(1, 1, 3, 1, 1)
    # Dataset-specific normalization lookup
    if custom_normalization:
        normal_lookup = {
            "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
            "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
            "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet": ([1, 1, 1], [0, 0, 0]),
            "something": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet1k": ([1, 1, 1], [0, 0, 0])
        }
        dataset_std, dataset_mean = normal_lookup.get(dataset_name, ([1, 1, 1], [0, 0, 0]))

        # Convert means and stds to tensors and reshape for broadcast compatibility
        dataset_mean = torch.tensor(dataset_mean, device=device).view(1, 1, 3, 1, 1)
        dataset_std = torch.tensor(dataset_std, device=device).view(1, 1, 3, 1, 1)
        

        # Perform denormalization
        # First reverse the dataset-specific normalization
        tensor = tensor * dataset_std + dataset_mean
    # Then reverse the default normalization
    return tensor * default_std + default_mean

# def scale_clamp(tensor, min_value, max_value): #this is made to be a differentiable version of torch's clamp
#     scale_down_factor = torch.where(tensor > max_value, tensor / max_value, torch.ones_like(tensor))
#     scale_up_factor = torch.where(tensor < min_value, tensor / min_value, torch.ones_like(tensor))
    
#     combined_scale_factor = torch.where(tensor > max_value, scale_down_factor, 
#                                         torch.where(tensor < min_value, scale_up_factor, torch.ones_like(tensor)))
    
#     scaled_tensor = tensor / combined_scale_factor
    
#     return scaled_tensor

def scale_clamp(tensor, min_value, max_value):
    scale_factor = torch.ones_like(tensor)
    scale_factor = torch.where(tensor > max_value, tensor / max_value, scale_factor)
    scale_factor = torch.where(tensor < min_value, tensor / min_value, scale_factor)
    
    scaled_tensor = tensor / scale_factor
    return scaled_tensor

def load_trained_pl_model(ckpt_path, new_hparams, txt_logger, for_inference = False):
    from base_model_trainer import ModelTrainer
    checkpoint = torch.load(ckpt_path)
    model = ModelTrainer(new_hparams, txt_logger = txt_logger)
    model.load_state_dict(checkpoint['state_dict'])
    if for_inference:
        model.cuda().eval()
        model.model.eval()
    return model.model

def print_model_layers_and_status(model):
    for name, module in model.named_modules():
        print(f'Layer: {name}, Type: {type(module).__name__}, Training Mode: {module.training}')

def init_whole_model_weights(model, weight_initialization_method):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(f"Unknown weight init method")
    
    model.apply(init_weights)


def load_pretrained_backbone(backbone_type, backbone_size, embedding_dim):
    vit_backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    
    backbone_type = backbone_type
    
    if backbone_type == 'dinov2':
        backbone_name = vit_backbone_archs[backbone_size]
        return torch.hub.load('facebookresearch/dinov2', model=f"dinov2_{backbone_name}")
    else:
        raise ValueError(f"Invalid backbone type: {backbone_type}")
    
def hinged_mse_loss(predictions, targets, margin=0.1):
    """
    Compute the Hinged MSE loss between predictions and targets.
    :param predictions: Predicted values.
    :param targets: Ground truth values.
    :param margin: The threshold below which errors are ignored.
    :return: Hinged MSE loss.
    """
    errors = torch.abs(predictions - targets)
    hinged_errors = torch.where(errors > margin, errors, torch.zeros_like(errors))
    loss = torch.mean(hinged_errors ** 2)
    return loss

def find_subsequences(input_tensor, sub_seq):
    sub_seq_len = len(sub_seq)
    batch_size, seq_len = input_tensor.shape
    sub_seq_tensor = torch.tensor(sub_seq, device=input_tensor.device)
    sub_seq_tensor = sub_seq_tensor.view(1, -1)
    windows = input_tensor.unfold(1, sub_seq_len, 1)
    matches = (windows == sub_seq_tensor).all(dim=2).long()
    
    if not matches.any(dim=1).all():
        raise ValueError("Sub-sequence not found in one or more sequences.")
    
    start_positions = matches.argmax(dim=1)
    return start_positions

def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor

def analyse_tokens(input_tensor, tokenizer):
    '''for debugging only'''
    decode = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    for i in range(input_tensor.shape[0]):
        print(input_tensor[i].tolist())
        print(decode[i])
        print('-'*60)
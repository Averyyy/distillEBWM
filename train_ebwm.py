import time
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import wandb
import random
import torch.nn.functional as F


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)
from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import initialize, print_rank, get_rank, save_rank, get_tokenizer

# Import EBWM modules without modification
from ebwm.nlp.ebwm import EBWM_NLP
from ebwm.model_utils import init_whole_model_weights

def get_student_model(args, device):
    """Initialize EBWM student model matching teacher dimensions"""
    teacher_config = AutoConfig.from_pretrained(args.teacher_model_path)
    
    hparams = {
        'embedding_dim': teacher_config.n_embd,
        'num_transformer_blocks': teacher_config.n_layer,
        'multiheaded_attention_heads': teacher_config.n_head,
        'context_length': args.max_length,
        'weight_initialization_method': 'xavier',
        'tokenizer': args.model_path,
        'mcmc_num_steps': 3,
        'mcmc_step_size_learnable': True,
        'mcmc_step_size': 30000,
        'vocab_to_embed_uses_prob_dist': False,
        'training_type': 'distill',
        'clamp_futures_grad_max_change': 0.1,
        'denoising_initial_condition': 'random_noise',
        'normalize_initial_condition': True,
        'batch_size_per_device': args.batch_size,
        'gaussian_random_noise_scaling': 1,
        'mcmc_no_chain': False,
        'clamp_futures_grad': True,
        'absolute_clamp': True,
    }
    
    model = EBWM_NLP(hparams)
    model = model.to(device)
    
    # Initialize weights
    for module in model.modules():
        if hasattr(module, 'weight'):
            init_whole_model_weights(module, hparams['weight_initialization_method'])
            
    print_rank(f'Initialized EBWM student with {sum(p.numel() for p in model.parameters()):,} parameters')
    return model

def get_teacher_model(args, device):
    """Load pre-trained teacher model"""
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    teacher.eval()
    print_rank(f'Loaded teacher model with {sum(p.numel() for p in teacher.parameters()):,} parameters')
    return teacher

def compute_distill_loss(student_logits, teacher_logits, mask):
    # Ensure sequence lengths match
    seq_len = min(student_logits.size(1), teacher_logits.size(1))
    student_logits = student_logits[:, :seq_len, :]
    teacher_logits = teacher_logits[:, :seq_len, :]
    mask = mask[:, :seq_len]

    # Calculate masked KL divergence
    kl_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='none'
    ).sum(dim=-1)
    
    # Apply mask and normalize
    masked_loss = (kl_loss * mask).sum() / mask.sum()
    
    return masked_loss

def train_step(args, model, teacher, batch, device):
    """EBWM training step with MCMC sampling"""
    model_batch, no_model_batch, gen_data = batch
    
    # Access elements properly
    input_ids = model_batch['input_ids'].to(device)
    mask = model_batch['attention_mask'].to(device)
    labels = no_model_batch['label'].to(device)
    
    # Get sequence dimensions
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1) 
    
    # Teacher forward
    with torch.no_grad():
        teacher_outputs = teacher(
            input_ids=input_ids[:, :-1],
            attention_mask=(input_ids[:, :-1] != args.pad_token_id).long()
        )
        teacher_logits = teacher_outputs.logits
    
    # Student forward (returns (pred_dists, energies))
    student_outputs, _ = model(input_ids)
    
    # Use final MCMC step logits
    final_logits = student_outputs[-1]
    final_logits = final_logits.view(-1, seq_len, final_logits.size(-1))
    
    min_seq_len = min(final_logits.size(1), teacher_logits.size(1))
    final_logits = final_logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    
    # Compute losses
    distill_loss = compute_distill_loss(final_logits, teacher_logits, mask)
    reconstruction_loss = F.cross_entropy(
        final_logits.view(-1, final_logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100
    )
    
    total_loss = 0.5 * distill_loss + 0.5 * reconstruction_loss
    return total_loss, distill_loss, reconstruction_loss

def train(args, model, teacher, dataloader, device):
    model.train()
    total_loss = 0.0
    total_distill = 0.0
    total_recon = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        loss, distill_loss, recon_loss = train_step(args, model, teacher, batch, device)
        
        model.backward(loss)
        model.step()
        
        total_loss += loss.item()
        total_distill += distill_loss.item()
        total_recon += recon_loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            avg_distill = total_distill / args.log_interval
            avg_recon = total_recon / args.log_interval
            
            if args.wandb and get_rank() == 0:
                wandb.log({
                    "loss": avg_loss,
                    "distill_loss": avg_distill,
                    "recon_loss": avg_recon,
                    "lr": model.optimizer.param_groups[0]['lr']
                })
            
            print_rank(
                f"Step {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"Distill: {avg_distill:.4f} | "
                f"Recon: {avg_recon:.4f}"
            )
            
            total_loss = 0.0
            total_distill = 0.0
            total_recon = 0.0

def main():
    args = get_args()
    initialize(args)
    device = torch.cuda.current_device()
    args.wandb = True
    # Initialize WandB
    if args.wandb and get_rank() == 0:
        wandb.init(
            project="ebwm-distill",
            config=vars(args),
            name="ebwm-distill",
        )
    
    # Load tokenizer
    tokenizer = get_tokenizer(args)
    args.pad_token_id = tokenizer.eos_token_id
    
    # Initialize models
    teacher = get_teacher_model(args, device)
    model = get_student_model(args, device)
    
    # Prepare dataset
    dataset = LMTrainDataset(
        args, tokenizer, args.data_dir, 
        "train", args.train_num, ratio=0.9,
        rng_sample=random.Random(args.seed)
    )
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,  
        model_parameters=model.parameters(),
        config=args.deepspeed_config
    )
    
    # Training loop
    for epoch in range(args.epochs):
        sampler = DistributedSampler(
            dataset, 
            shuffle=True, 
            rank=get_rank(), 
            num_replicas=dist.get_world_size()
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=dataset.collate
        )
        
        train(args, model_engine, teacher, dataloader, device)
        
        # Save checkpoint
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            model_engine.save_checkpoint(args.save_dir, f"epoch_{epoch+1}")
            
    if args.wandb and get_rank() == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
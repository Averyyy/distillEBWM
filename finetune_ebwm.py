# distillEBWM/finetune_ebwm.py

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer
from utils import get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics
from peft import PeftModel

from ebwm.nlp.ebwm import EBWM_NLP

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    config.is_model_parallel = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float16
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32
        )
        model = model.half()

    # If using LoRA or another PEFT method for teacher:
    if args.peft is not None and args.teacher_peft_path is not None:
        if args.peft == "lora":
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
        else:
            raise NotImplementedError
    else:
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    return model


def get_optimizer(args, model):
    while isinstance(model, DDP):
        model = model.module
    param_groups = get_optimizer_params(args, model)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler {args.lr_decay_style} not supported.")
    return lr_scheduler


def setup_ebwm_student_model(args, device):
    """
    This function loads or initializes the EBWM as the student model.
    It respects your hyperparams in args and sets up for training.
    """
    # Suppose you have a set of EBWM hyperparams
    # We fill them from the user’s arguments.
    hparams = {
        "tokenizer": args.model_path,
        "embedding_dim": 768, 
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "batch_size_per_device": args.batch_size,
        "context_length": args.max_length,
        "weight_initialization_method": "xavier",
        # EBWM-specific hyperparams
        "mcmc_step_size": 0.01,
        "mcmc_step_size_learnable": True,
        "mcmc_num_steps": 5,
        "mcmc_no_chain": False,
        "normalize_initial_condition": True,
        "vocab_to_embed_uses_prob_dist": False,
        "absolute_clamp": 0.0,
        "clamp_futures_grad": False,
        "clamp_futures_grad_max_change": 1.0,
        "denoising_initial_condition": "random_noise",
        "gaussian_random_noise_scaling": 1.0,
        "training_type": "finetune",
        "reconstruction_coeff": 1.0,
        "clamp_max_after_warm_up": 0.0
    }

    student_model = EBWM_NLP(hparams)
    student_model = student_model.float().to(device)

    if args.model_ckpt is not None and os.path.exists(args.model_ckpt):
        print_rank(f"Loading pretrained EBWM checkpoint from: {args.model_ckpt}")
        map_location = f"cuda:{device}" if isinstance(device, int) else device
        checkpoint = torch.load(args.model_ckpt, map_location=map_location)
        student_model.load_state_dict(checkpoint['state_dict'], strict=False)

    return student_model


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("At least one of do_train/do_eval must be True.")

    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def get_distil_loss(args, tokenizer, logits, teacher_model, model_batch, no_model_batch):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits

    if "sfkl" in args.type:
        distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "srkl" in args.type:
        distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "jsd" in args.type:
        distil_loss = js_distance(logits, teacher_logits, no_model_batch)
    elif "tvd" in args.type:
        distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
    elif "fkl" in args.type or args.type == "kd":
        distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
    elif "rkl" in args.type:
        distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
    else:
        raise NotImplementedError
    return distil_loss


def finetune_ebwm(args, tokenizer, ebwm_student_model, teacher_model, device, dataset):
    print_rank("Starting EBWM Fine-tuning")

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None

    # For convenience, wrap the PyTorch Lightning model in a basic forward method:
    # We'll define a small wrapper to mimic huggingface input signature
    # because the data loader expects “input_ids, attention_mask, labels” etc.

    class EBWMWrapper(nn.Module):
        def __init__(self, pl_model):
            super().__init__()
            self.pl_model = pl_model
        def forward(self, input_ids=None, **kwargs):
            # Create a dummy dict simulating the wrapper
            x = {
                "input_ids": input_ids
            }
            # forward_loss_wrapper returns a dict with "loss" key
            result_dict = self.pl_model.forward_loss_wrapper(x)
            out = nn.functional.one_hot(input_ids, num_classes=self.pl_model.vocab_size).float()
            batch_size, seq_len = input_ids.shape
            with torch.no_grad():
                logits_2d = self.pl_model(input_ids) # shape [B*S, vocab_size]
            # reshape back
            logits_3d = logits_2d.view(batch_size, seq_len, self.pl_model.vocab_size)
            # Return a structure that has .logits and .loss
            class OutputWrap:
                pass
            ret = OutputWrap()
            ret.logits = logits_3d
            ret.loss = result_dict["loss"]
            return ret

    # Construct the wrapper
    model = EBWMWrapper(ebwm_student_model).to(device)
    # Then we proceed with standard deepspeed approaches
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    # Initialize deepspeed
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"]
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        # config_params=ds_config
    )

    # Prepare data
    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )

    loss_func = nn.CrossEntropyLoss()

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else 0.0
    if "dev" in dataset:
        prev_avg_loss = evaluate_ebwm(args, tokenizer, model_engine, dataset["dev"], "dev", 0, device, adaptive_threshold)
    else:
        prev_avg_loss = 9999999.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model_engine.train()

        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()

            # Standard forward + backward
            outputs = model_engine(**model_batch)  # calls EBWMWrapper.forward
            logits = outputs.logits  # shape [B, S, vocab_size]
            ebwm_loss = outputs.loss  # from forward_loss_wrapper

            if teacher_model is not None:
                # get distill loss
                distil_loss = get_distil_loss(args, tokenizer, logits, teacher_model, model_batch, no_model_batch)
                loss = (1 - args.kd_ratio) * ebwm_loss + args.kd_ratio * distil_loss
            else:
                loss = ebwm_loss

            model_engine.backward(loss)
            model_engine.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time

            def get_log(log_loss, log_distil_loss, log_time):
                return (
                    f"train | epoch {epoch:3d} | Iter: {step:6d}/{args.total_iters * args.gradient_accumulation_steps:6d} "
                    f"| global iter: {global_step:6d}/{args.total_iters:6d} | loss: {log_loss:.4f} | ds_loss: {log_distil_loss:.4f} "
                    f"| lr: {lr_scheduler.get_last_lr()[0]:.4e} | micro time: {elapsed_time:.3f} | step time: {log_time:.3f}"
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / args.log_interval
                )
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

            # save
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    # For EBWM you can save the Lightning state dict:
                    print_rank(f"EBWM Model save to {save_dir_path}")
                    torch.save({"state_dict": ebwm_student_model.state_dict()}, os.path.join(save_dir_path, "ebwm_student_model.ckpt"))
                    # Also save tokenizer or other config if needed
                    tokenizer.save_pretrained(save_dir_path)
                dist.barrier()

            # eval
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                if "dev" in dataset:
                    curr_avg_loss = evaluate_ebwm(args, tokenizer, model_engine, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                    if "adaptive" in args.type:
                        if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                            adaptive_threshold += 0.1
                            adaptive_threshold = min(adaptive_threshold, 1.0)
                            prev_avg_loss = curr_avg_loss
                model_engine.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            if global_step > args.total_iters:
                break
    return model_engine


def evaluate_ebwm(args, tokenizer, model_engine, dataset, split, epoch, device, adaptive_threshold=None):
    collate_fn = dataset.collate
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, 
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model_engine.eval()
    all_loss = 0.0
    step = 0
    all_response_ids = []

    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dp_rank != 0))):
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            outputs = model_engine(**model_batch)
            logits = outputs.logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1

    avg_loss = all_loss / step
    if dp_rank == 0:
        log_str = f"{split} | avg_loss: {avg_loss:.4f}"
        if "adaptive" in args.type:
            log_str += f" | threshold: {adaptive_threshold}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    return avg_loss


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EBWM EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))

    # get tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)

    dp_world_size = dist.get_world_size()
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    # teacher model
    teacher_model = None
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)

    # student model setup
    ebwm_student_model = setup_ebwm_student_model(args, device)

    if args.do_train:
        finetune_ebwm(args, tokenizer, ebwm_student_model, teacher_model, device, dataset)

    if args.do_eval and "test" in dataset:
        evaluate_ebwm(args, tokenizer, ebwm_student_model, dataset["test"], "test", 0, device)


if __name__ == "__main__":
    main()
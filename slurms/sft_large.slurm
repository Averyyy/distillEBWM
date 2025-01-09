#!/bin/bash
#SBATCH --partition=gpuA100x4         # GPU partition to use
#SBATCH --account=bdta-delta-gpu      # Account to charge
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --gres=gpu:4                  # Number of GPUs
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=04:00:00               # Maximum runtime
#SBATCH --job-name=sft_large       # Job name
#SBATCH --output=/work/hdd/bdta/aqian1/distillEBWM/logs/sft_large%x_%j.out     # Standard output log file
#SBATCH --error=/work/hdd/bdta/aqian1/distillEBWM/logs/sft_large%x_%j.err      # Standard error log file

# Load environment and activate test_setup
source activate test_setup

# Navigate to the project directory
cd /work/hdd/bdta/aqian1/distillEBWM

# Run the command

# preprocessing
# bash scripts/gpt2/tools/process_data_pretrain.sh /work/hdd/bdta/aqian1/distillEBWM 6000 4

# sft
bash scripts/gpt2/sft/sft_large.sh /work/hdd/bdta/aqian1/distillEBWM 5965 4

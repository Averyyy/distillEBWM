#!/bin/bash

#SBATCH --account=bdta-delta-gpu
#SBATCH --array=0
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=/u/aqian1/research/distillEBWM/logs/%x_%j.out
#SBATCH --error=/u/aqian1/research/distillEBWM/logs/%x_%j.err

# Paths
REPO_DIR=/work/hdd/bdta/distillEBWM
SCRATCH_DIR=$REPO_DIR/scratch
OUTPUT_DIR=$REPO_DIR/output
LOG_DIR=/u/aqian1/research/distillEBWM/logs
DATA_DIR=$REPO_DIR/data
# ENV_DIR=/work/hdd/bdta/conda_envs/distillEBWM_env
NEW_ENV_DIR=/work/hdd/bdta/conda_envs/clean_distillEBWM_env

# Create directories
mkdir -p $SCRATCH_DIR $OUTPUT_DIR $LOG_DIR $DATA_DIR $ENV_DIR

# Load Anaconda module
module load anaconda3_gpu/23.7.4
module load cuda/12.3.0
# Initialize Conda
eval "$(conda shell.bash hook)"

# Create or activate Conda environment
# if [ ! -d "$ENV_DIR" ]; then
#     conda create -y -p $ENV_DIR python=3.10
# fi
source activate $NEW_ENV_DIR

# Debugging: Verify environment and Python installation
echo "Loaded modules:"
module list
echo "Conda environments:"
conda info --envs
which python
python --version

# Install dependencies (first time load)
# bash install.sh

# Copy files to scratch directory
cd $SCRATCH_DIR
cp -r $SLURM_SUBMIT_DIR/* $SCRATCH_DIR
cd $SCRATCH_DIR

# Run script
python generate.py --data-dir $DATA_DIR --save $OUTPUT_DIR --deepspeed_config ./test_config.json --model-path ./checkpoints/gpt2-small

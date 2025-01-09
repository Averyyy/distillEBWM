#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdta-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --job-name=distillm_experiment
#SBATCH --output=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.out
#SBATCH --error=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.err

source activate test_setup
cd /work/hdd/bdta/aqian1/distillEBWM

# modify script name to run different tasks
bash scripts/gpt2/distillm/train_0.1B_0.7B.sh /work/hdd/bdta/aqian1/distillEBWM 4137 4
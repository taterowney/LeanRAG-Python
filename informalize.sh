#!/bin/bash

#SBATCH --job-name=informalize
#SBATCH --output=logs/informalize.out
#SBATCH --error=logs/informalize.err
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150G



# --exclude=babel-0-31

source ~/miniconda3/bin/activate ImProver_env
export HF_HOME=/data/user_data/$USER/HF

python test.py

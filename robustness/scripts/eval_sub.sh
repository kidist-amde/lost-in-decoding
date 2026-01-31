#!/bin/sh
#SBATCH --job-name=eval_sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=250G
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
 
conda activate ir

nvidia-smi

 

dataset=$1
model=$2

python eval.py --dataset $dataset --model_name $model 

 
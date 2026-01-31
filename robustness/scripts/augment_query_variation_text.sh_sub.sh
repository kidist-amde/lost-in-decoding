#!/bin/sh
#SBATCH --job-name=augment_query_variation_text_sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=70G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100 
#SBATCH --time=00-20:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
  
conda activate ir

nvidia-smi



dataset=$1
attack_method=$2
seed=$3
 

python attack_queries_penha.py --dataset $dataset --seed $seed --attack_method $attack_method  
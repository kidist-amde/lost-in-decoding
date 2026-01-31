#!/bin/sh
#SBATCH --job-name=eval_attack_document_sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=280G
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --partition=gpu_h100
#SBATCH --time=00-02:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
conda activate ir

nvidia-smi


dataset=$1
model=$2
attacked_num=$3
seed=$4

python eval_attack.py --dataset $dataset --model_name $model  --attacked_num $attacked_num --seed $seed \
    --attack_method supervised_poisoning  
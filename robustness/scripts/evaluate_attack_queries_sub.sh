#!/bin/sh
#SBATCH --job-name=evaluate_attack_queries_sub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=280G
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --partition=gpu_h100
#SBATCH --time=00-01:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda

conda activate pag-robustness

nvidia-smi

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

dataset=$1
model=$2
attack_method=$3
seed=$4

python -m robustness.evaluation.attack_eval \
  --dataset $dataset \
  --model_name $model \
  --attack_method $attack_method \
  --seed $seed

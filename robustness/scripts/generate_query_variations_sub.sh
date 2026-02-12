#!/bin/sh
#SBATCH --job-name=generate_query_variations_sub
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

conda activate pag-robustness

nvidia-smi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

dataset=$1
attack_method=$2
seed=$3

python -m robustness.query_variations.generate_penha \
  --dataset $dataset \
  --seed $seed \
  --attack_method $attack_method

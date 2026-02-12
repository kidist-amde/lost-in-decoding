#!/bin/sh
#SBATCH --job-name=generate_query_variations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

conda activate pag-robustness

nvidia-smi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

dataset_list=(
    "msmarco"
)

attack_method_list=(
    "mispelling" # All submitted
    "ordering" # All submitted
    "synonym" # All submitted
    "paraphrase" # All submitted
    "naturality" # All submitted
)

seed_list=(1999 5 27 2016 2026)

for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for attack_method in ${attack_method_list[@]}; do
            sbatch robustness/scripts/generate_query_variations_sub.sh ${dataset} ${attack_method} ${seed}
        done
    done
done

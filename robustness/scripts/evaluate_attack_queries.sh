#!/bin/sh
#SBATCH --job-name=evaluate_attack_queries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-01:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
 
conda activate pag-robustness

nvidia-smi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

dataset_list=(
    "msmarco"
)

seed_list=( 1999 5 27 2016 2026)

attack_method_list=(
    "none"
    "mispelling"
    "ordering"
    "synonym"
    "paraphrase"
    "naturality"
)

model_name_list=(
    "contriever" 
    "bge_m3"  
    "qwen3" 
    "linq"
    "gte"
    "reasonir"  
    "diver"
    "bge_reasoner"
    "qwen3_4B"
    "qwen3_0.6B"
)  

for dataset in ${dataset_list[@]}; do
    for model_name in ${model_name_list[@]}; do
        for attack_method in ${attack_method_list[@]}; do
            for seed in ${seed_list[@]}; do
                sbatch robustness/scripts/evaluate_attack_queries_sub.sh ${dataset} ${model_name} ${attack_method} ${seed}
            done
        done
    done   
done

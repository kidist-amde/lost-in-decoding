#!/bin/sh
#SBATCH --job-name=augment_query_variation_text
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

 
conda activate ir

nvidia-smi



dataset_list=(
    "nq"
    "hotpotqa"
    "fiqa"
    "msmarco"
)

attack_method_list=(
    "mispelling" # All submitted
    "ordering" # All submitted
    "synonym" # All submitted
    "paraphrase" # All submitted
    "naturality" # All submitted
)

seed_list=(1999  5  27 2016 2026 ) 


for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for attack_method in ${attack_method_list[@]}; do
            sbatch  scripts/augment_query_variation_text_sub.sh ${dataset} ${attack_method} ${seed}
        done
    done
done
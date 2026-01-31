#!/bin/sh
#SBATCH --job-name=eval_attack_document 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-02:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
 
conda activate ir

nvidia-smi

 

dataset_list=(
    "nq"
    "hotpotqa"
    "msmarco"
)

seed_list=( 1999 5 27 2016 2026)

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

attacked_num_list=(
    10
    50
)

for dataset in ${dataset_list[@]}; do
    for model_name in ${model_name_list[@]}; do
        for attacked_num in ${attacked_num_list[@]}; do
            for seed in ${seed_list[@]}; do
                sbatch scripts/eval_attack_document_sub.sh ${dataset} ${model_name} ${attacked_num} ${seed}
            done
        done
    done   
done

 
#!/bin/bash
#SBATCH --job-name=pag_bow
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=experiments/%x-%j_bow.out
#SBATCH --error=experiments/%x-%j_bow.err
#SBATCH --chdir=/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/

mkdir -p experiments

# initialize conda for non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh   
conda activate pag-env


collection_path=./data/msmarco-full/full_collection/
experiment_dir=experiments-splade

export CUDA_VISIBLE_DEVICES=0
model_dir="./data/$experiment_dir/t5-splade-0-12l"
pretrained_path=$model_dir/checkpoint

for bow_topk in 16 32 64 128
do
    out_dir=$model_dir/top_bow_$bow_topk
    python -m t5_pretrainer.evaluate \
        --task=spalde_get_bow_rep \
        --pretrained_path=$pretrained_path \
        --index_retrieve_batch_size=128 \
        --collection_path=$collection_path \
        --out_dir=$out_dir \
        --bow_topk=$bow_topk
done

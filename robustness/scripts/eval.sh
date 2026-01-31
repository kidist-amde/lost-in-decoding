#!/bin/sh
#SBATCH --job-name=eval 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-02:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
 
conda activate ir

nvidia-smi
 
# Large models need to apply for 2 GPUs to store the memory
big_dataset_list=( 
    "msmarco"
    "nq"
    "hotpotqa"
    "dbpedia-entity"
    "fever"
    "climate-fever"
)

dataset_list=(    
    "browsecomp_plus"
    "trec-covid"
    "nfcorpus"
    # "nq"
    # "hotpotqa"
    "fiqa"
    "arguana"
    "webis-touche2020"
    "quora"
    # "dbpedia-entity"
    "scidocs"
    # "fever"
    # "climate-fever"
    "scifact"
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "leetcode"
    "pony"
    "aops"
    "theoremqa_theorems"
    "theoremqa_questions"  
    # "msmarco"
)

 
model_list=(
    'bm25'
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
 
 

for  model in ${model_list[@]}; do
    for dataset in ${dataset_list[@]}; do
            #sleep 5 seconds
            sleep 5
            sbatch scripts/eval_sub.sh ${dataset} ${model}
            echo "dataset: ${dataset}, model: ${model} has been submitted"
        done
done

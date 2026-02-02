#!/bin/bash
#SBATCH --job-name=rq2_pag_robust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-06:00:00
#SBATCH --output=experiments/RQ2_robustness/%x-%j.out
#SBATCH --error=experiments/RQ2_robustness/%x-%j.err

# ──────────────────────────────────────────────────────────────────────
# RQ2 single-job runner
#
# Arguments:
#   $1 = split         (dl19 | dl20 | dev)
#   $2 = attack_method (mispelling | ordering | synonym | paraphrase | naturality)
#   $3 = seed          (1999 | 5 | 27 | 2016 | 2026)
# ──────────────────────────────────────────────────────────────────────

# Initialise environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

nvidia-smi

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

split=$1
attack_method=$2
seed=$3

echo "========================================"
echo "RQ2 PAG Robustness Evaluation"
echo "  split:         $split"
echo "  attack_method: $attack_method"
echo "  seed:          $seed"
echo "========================================"

python -m robustness.evaluation.rq2 \
    --split "$split" \
    --attack_method "$attack_method" \
    --seed "$seed" \
    --n_gpu 1 \
    --batch_size 8 \
    --lex_topk 1000 \
    --smt_topk 100 \
    --output_dir experiments/RQ2_robustness

echo "Done: split=$split attack=$attack_method seed=$seed"

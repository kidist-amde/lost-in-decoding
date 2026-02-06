#!/bin/bash
#SBATCH --job-name=rq3_crosslingual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-12:00:00
# NOTE: --output and --error are set by the launcher script
# so that logs are organized per language/split.

# ──────────────────────────────────────────────────────────────────────────
# RQ3 Cross-lingual single-job runner
#
# Arguments:
#   $1 = language   (fr | de | zh | nl)
#   $2 = split      (dl19 | dl20 | dev)
# ──────────────────────────────────────────────────────────────────────────

# Initialise environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

nvidia-smi

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

language=$1
split=$2

# Derive a unique MASTER_PORT from SLURM_JOB_ID to avoid port conflicts
# when multiple jobs land on the same node.
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))

echo "========================================"
echo "RQ3 Cross-lingual Evaluation"
echo "  language:      $language"
echo "  split:         $split"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "========================================"

python -m cross_lingual.evaluation.rq3 \
    --language "$language" \
    --split "$split" \
    --n_gpu 1 \
    --batch_size 8 \
    --lex_topk 1000 \
    --smt_topk 100 \
    --translation_model nllb \
    --output_dir experiments/RQ3_crosslingual

echo "Done: language=$language split=$split"

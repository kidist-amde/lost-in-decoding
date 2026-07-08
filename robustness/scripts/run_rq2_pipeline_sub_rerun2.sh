#!/bin/bash
#SBATCH --job-name=rq2_pag_robust_rerun2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-06:00:00
#SBATCH --output=experiments/RQ2_robustness_rerun2/logs/%x-%j.out
#SBATCH --error=experiments/RQ2_robustness_rerun2/logs/%x-%j.err

# ──────────────────────────────────────────────────────────────────────
# Second independent launch of the RQ2 clean-run pipeline, unchanged
# config, for run-to-run decoding variance verification.
#
# Identical to robustness/scripts/run_rq2_pipeline_sub.sh except
# --output_dir points at a fresh directory so the original clean run
# under experiments/RQ2_robustness is not overwritten.
#
# Arguments:
#   $1 = split         (dl19 | dl20 | dev)
#   $2 = attack_method (mispelling | ordering | synonym | paraphrase | naturality)
#   $3 = seed          (1999 | 5 | 27 | 2016 | 2026)
# ──────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

nvidia-smi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

split=$1
attack_method=$2
seed=$3

export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))

echo "========================================"
echo "RQ2 PAG Robustness Evaluation (rerun 2, verification)"
echo "  split:         $split"
echo "  attack_method: $attack_method"
echo "  seed:          $seed"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "========================================"

python -m robustness.evaluation.rq2 \
    --split "$split" \
    --attack_method "$attack_method" \
    --seed "$seed" \
    --n_gpu 1 \
    --batch_size 8 \
    --lex_topk 1000 \
    --smt_topk 100 \
    --output_dir experiments/RQ2_robustness_rerun2

echo "Done: split=$split attack=$attack_method seed=$seed"

#!/bin/bash
#SBATCH --job-name=rq2_ripor_robust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-06:00:00
# Default log paths (used when submitting this script directly via sbatch).
# The launcher script can still override these with --output/--error.
#SBATCH --output=experiments/RQ2_robustness_ripor/logs/%x-%j.out
#SBATCH --error=experiments/RQ2_robustness_ripor/logs/%x-%j.err

# ──────────────────────────────────────────────────────────────────────
# RQ2 RIPOR single-job runner
#
# Arguments:
#   $1 = split         (dl19 | dl20 | dev)
#   $2 = attack_method (mispelling | ordering | synonym | paraphrase | naturality)
#   $3 = seed          (1999 | 5 | 27 | 2016 | 2026)
# ──────────────────────────────────────────────────────────────────────

# Initialise environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
export PYTHONUNBUFFERED=1

nvidia-smi

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

split=$1
attack_method=$2
seed=$3

# Derive a unique MASTER_PORT from SLURM_JOB_ID to avoid port conflicts
# when multiple jobs land on the same node.
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))

echo "========================================"
echo "RQ2 RIPOR Robustness Evaluation"
echo "  split:         $split"
echo "  attack_method: $attack_method"
echo "  seed:          $seed"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "========================================"

# Pre-build prefix tree if not cached (only runs once, then reuses pickle)
DOCID_SMTID=/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/RIPOR/RIPOR_data/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json
PKL_PATH="$(dirname "$DOCID_SMTID")/list_smtid_to_nextids.pkl"
if [ ! -f "$PKL_PATH" ]; then
    echo "Building prefix tree (first run only)..."
    PYTHONPATH=/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/RIPOR python \
        -m t5_pretrainer.aq_preprocess.build_list_smtid_to_nextids \
        --docid_to_smtid_path="$DOCID_SMTID"
fi

python -u -m robustness.evaluation.rq2_ripor \
    --split "$split" \
    --attack_method "$attack_method" \
    --seed "$seed" \
    --n_gpu 1 \
    --batch_size 1 \
    --topk 100 \
    --output_dir experiments/RQ2_robustness_ripor

echo "Done: split=$split attack=$attack_method seed=$seed"

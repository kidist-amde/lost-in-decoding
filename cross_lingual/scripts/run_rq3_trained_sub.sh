#!/bin/bash
#SBATCH --job-name=rq3_trained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-12:00:00

# ──────────────────────────────────────────────────────────────────────────
# RQ3 Trained Extension: single-language job
#
# Runs all 3 steps (data prep, train, eval) for one language.
#
# Arguments:
#   $1 = language   (fr | de | zh | nl)
#   $2 = mode       (all | eval-only)  [default: all]
# ──────────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

nvidia-smi

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

LANG=$1
MODE="${2:-all}"    # all | eval-only
OUT_DIR="experiments/RQ3_crosslingual_trained"

export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))

echo "========================================"
echo "RQ3 Trained Extension"
echo "  language:      $LANG"
echo "  mode:          $MODE"
echo "  output_dir:    $OUT_DIR"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "========================================"

if [ "$MODE" != "eval-only" ]; then

# Step 1: Prepare parallel data
echo ""
echo "=== Step 1: Prepare parallel data ==="
python -m cross_lingual.trained_extension.data_loader_parallel \
    --languages "$LANG"

# Step 2: Train planner alignment
echo ""
echo "=== Step 2: Train planner alignment ==="
python -m cross_lingual.trained_extension.train_planner_alignment \
    --language "$LANG" \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 8 \
    --temperature 2.0 \
    --topk_union 100 \
    --finetune_last_n_layers 12

else
    echo "=== Skipping Steps 1-2 (eval-only mode) ==="
fi

# Step 3: Evaluate adapted checkpoint
echo ""
echo "=== Step 3: Evaluate adapted checkpoint ==="
CKPT_DIR="cross_lingual/trained_extension/checkpoints/${LANG}/best"
if [ ! -d "$CKPT_DIR" ]; then
    CKPT_DIR="cross_lingual/trained_extension/checkpoints/${LANG}/final"
fi

if [ -d "$CKPT_DIR" ]; then
    EVAL_ARGS=(
        --language "$LANG"
        --adapted_checkpoint "$CKPT_DIR"
        --output_dir "$OUT_DIR"
        --lex_topk 1000
        --smt_topk 100
        --batch_size 8
        --n_gpu 1
    )
    # In eval-only mode, force re-run (stale results from previous failed runs)
    if [ "$MODE" = "eval-only" ]; then
        EVAL_ARGS+=(--force)
    fi
    python -m cross_lingual.trained_extension.evaluate_adapted "${EVAL_ARGS[@]}"
else
    echo "ERROR: No checkpoint found at $CKPT_DIR"
    exit 1
fi

echo "Done: language=$LANG"

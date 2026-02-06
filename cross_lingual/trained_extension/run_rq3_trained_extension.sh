#!/usr/bin/env bash
# ============================================================
# RQ3 Trained Extension: Master runner
# ============================================================
#
# Steps:
#   1. Prepare parallel query data for all languages
#   2. Train planner alignment (per language)
#   3. Evaluate adapted checkpoint alongside baselines
#   4. Aggregate results + significance tests
#
# Usage:
#   bash cross_lingual/trained_extension/run_rq3_trained_extension.sh
#
#   # Override defaults:
#   bash cross_lingual/trained_extension/run_rq3_trained_extension.sh \
#       --languages "fr de" --epochs 3 --lr 1e-5 --batch_size 8
#
# SLURM example:
#   sbatch --gpus=1 --time=24:00:00 --mem=64G \
#       cross_lingual/trained_extension/run_rq3_trained_extension.sh
#
# ============================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LANGUAGES="zh nl fr de"
EPOCHS=5
LR="1e-5"
BATCH_SIZE=8
TEMPERATURE=2.0
TOPK_UNION=100
FINETUNE_LAST_N=12
N_GPU=1
LEX_TOPK=1000
SMT_TOPK=100
CONFIG=""
OUTPUT_DIR=""
FORCE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --languages)       LANGUAGES="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --lr)              LR="$2"; shift 2 ;;
        --batch_size)      BATCH_SIZE="$2"; shift 2 ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --topk_union)      TOPK_UNION="$2"; shift 2 ;;
        --finetune_last_n) FINETUNE_LAST_N="$2"; shift 2 ;;
        --n_gpu)           N_GPU="$2"; shift 2 ;;
        --lex_topk)        LEX_TOPK="$2"; shift 2 ;;
        --smt_topk)        SMT_TOPK="$2"; shift 2 ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --force)           FORCE="--force"; shift ;;
        *)                 echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAINED_EXT="${REPO_ROOT}/cross_lingual/trained_extension"
CKPT_DIR="${TRAINED_EXT}/checkpoints"
RESULTS_DIR="${TRAINED_EXT}/results"
EVAL_DIR="${OUTPUT_DIR:-${REPO_ROOT}/cross_lingual/results}"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# MASTER_PORT for distributed training
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
    echo "[runner] SLURM_JOB_ID=${SLURM_JOB_ID}, MASTER_PORT=${MASTER_PORT}"
else
    export MASTER_PORT=${MASTER_PORT:-29500}
fi

# ---------------------------------------------------------------------------
# Log hardware info
# ---------------------------------------------------------------------------
{
    echo "============================================================"
    echo "RQ3 Trained Extension - $(date)"
    echo "============================================================"
    echo "Hostname:   $(hostname)"
    echo "Languages:  ${LANGUAGES}"
    echo "Epochs:     ${EPOCHS}"
    echo "LR:         ${LR}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Temperature:${TEMPERATURE}"
    echo "TopK union: ${TOPK_UNION}"
    echo "Finetune N: ${FINETUNE_LAST_N}"
    echo "N GPU:      ${N_GPU}"
    echo ""
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    fi
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || true
    echo ""
} | tee "${LOG_DIR}/hardware_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Step 1: Prepare parallel data
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 1: Prepare parallel query data"
echo "============================================================"

python3 -m cross_lingual.trained_extension.data_loader_parallel \
    --languages ${LANGUAGES} \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee "${LOG_DIR}/data_prep.log"

# ---------------------------------------------------------------------------
# Step 2: Train planner alignment (per language)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 2: Train planner alignment"
echo "============================================================"

for LANG in ${LANGUAGES}; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Training for ${LANG}"
    echo "------------------------------------------------------------"

    TRAIN_ARGS=(
        --language "${LANG}"
        --epochs "${EPOCHS}"
        --lr "${LR}"
        --batch_size "${BATCH_SIZE}"
        --temperature "${TEMPERATURE}"
        --topk_union "${TOPK_UNION}"
        --finetune_last_n_layers "${FINETUNE_LAST_N}"
        --output_dir "${CKPT_DIR}"
    )

    if [ -n "${CONFIG}" ]; then
        TRAIN_ARGS+=(--config "${CONFIG}")
    fi

    python3 -m cross_lingual.trained_extension.train_planner_alignment \
        "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train_${LANG}.log"

    echo "[runner] Training complete for ${LANG}"
done

# ---------------------------------------------------------------------------
# Step 3: Evaluate adapted checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 3: Evaluate adapted planner"
echo "============================================================"

for LANG in ${LANGUAGES}; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Evaluating ${LANG}"
    echo "------------------------------------------------------------"

    # Auto-detect best checkpoint
    CKPT="${CKPT_DIR}/${LANG}/best"
    if [ ! -d "${CKPT}" ]; then
        CKPT="${CKPT_DIR}/${LANG}/final"
    fi

    if [ ! -d "${CKPT}" ]; then
        echo "[runner] No checkpoint found for ${LANG}, skipping evaluation"
        continue
    fi

    python3 -m cross_lingual.trained_extension.evaluate_adapted \
        --language "${LANG}" \
        --adapted_checkpoint "${CKPT}" \
        --output_dir "${EVAL_DIR}" \
        --lex_topk "${LEX_TOPK}" \
        --smt_topk "${SMT_TOPK}" \
        --batch_size "${BATCH_SIZE}" \
        --n_gpu "${N_GPU}" \
        ${FORCE} \
        2>&1 | tee "${LOG_DIR}/eval_${LANG}.log"

    echo "[runner] Evaluation complete for ${LANG}"
done

# ---------------------------------------------------------------------------
# Step 4: Aggregate results + significance
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 4: Aggregate results"
echo "============================================================"

python3 -m cross_lingual.trained_extension.evaluate_adapted \
    --language all \
    --output_dir "${EVAL_DIR}" \
    --significance \
    --summary_csv "${RESULTS_DIR}/summary_rq3_trained.csv" \
    2>&1 | tee "${LOG_DIR}/aggregate.log"

echo ""
echo "============================================================"
echo "  Done! Results at:"
echo "    Checkpoints: ${CKPT_DIR}"
echo "    Results:     ${RESULTS_DIR}"
echo "    Summary CSV: ${RESULTS_DIR}/summary_rq3_trained.csv"
echo "============================================================"

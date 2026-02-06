#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# RQ3 Cross-Lingual Baselines Runner
#
# Runs all baselines (naive, sequential, translate) for all languages
# (zh, nl, fr, de) against the English MS MARCO passage corpus using
# released English PAG artifacts.
#
# Usage:
#   ./cross_lingual/run_rq3_baselines.sh                   # all defaults
#   ./cross_lingual/run_rq3_baselines.sh --languages "fr de"
#   ./cross_lingual/run_rq3_baselines.sh --batch_size 4 --beam_size 50
#   ./cross_lingual/run_rq3_baselines.sh --baselines "naive translate"
#
# Environment:
#   Expects conda env 'pag-env' with torch, transformers, pytrec_eval,
#   datasets, and the NLLB-200 model for translation.
#
# On SLURM, wrap this script or invoke the individual baselines via sbatch.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="/gpfs/work4/0/prjs1037/dpo-exp/pag-repro"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
LANGUAGES="${LANGUAGES:-zh nl fr de}"
BASELINES="${BASELINES:-naive sequential translate}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BEAM_SIZE="${BEAM_SIZE:-100}"       # smt_topk (Stage 2 beam/top-k)
LEX_TOPK="${LEX_TOPK:-1000}"       # Stage 1 top-k
N_GPU="${N_GPU:-1}"
TRANSLATION_MODEL="${TRANSLATION_MODEL:-nllb}"
TRANSLATION_BATCH="${TRANSLATION_BATCH:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/cross_lingual/results}"
DEVICE="${DEVICE:-cuda}"

# ── Parse CLI overrides ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --languages)      LANGUAGES="$2";           shift 2 ;;
        --baselines)      BASELINES="$2";           shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";          shift 2 ;;
        --beam_size)      BEAM_SIZE="$2";           shift 2 ;;
        --lex_topk)       LEX_TOPK="$2";            shift 2 ;;
        --n_gpu)          N_GPU="$2";               shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";          shift 2 ;;
        --translation_model) TRANSLATION_MODEL="$2"; shift 2 ;;
        --translation_batch) TRANSLATION_BATCH="$2"; shift 2 ;;
        --device)         DEVICE="$2";              shift 2 ;;
        --force)          FORCE="--force";          shift ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done
FORCE="${FORCE:-}"

# ── Derive unique MASTER_PORT ─────────────────────────────────────────────
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 10000) ))
else
    export MASTER_PORT=29500
fi

# ── Log hardware + hyperparameters ────────────────────────────────────────
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

{
    echo "========================================"
    echo "RQ3 Cross-Lingual Baselines"
    echo "  Date:              $(date -Iseconds)"
    echo "  Hostname:          $(hostname)"
    echo "  SLURM_JOB_ID:     ${SLURM_JOB_ID:-local}"
    echo "  Languages:         $LANGUAGES"
    echo "  Baselines:         $BASELINES"
    echo "  Batch size:        $BATCH_SIZE"
    echo "  Beam size (smt):   $BEAM_SIZE"
    echo "  Lex top-k:         $LEX_TOPK"
    echo "  N GPUs:            $N_GPU"
    echo "  Translation model: $TRANSLATION_MODEL"
    echo "  Output dir:        $OUTPUT_DIR"
    echo "  MASTER_PORT:       $MASTER_PORT"
    echo "========================================"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi
    fi
    echo ""
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || true
    echo ""
} | tee "$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# ── Run baselines ─────────────────────────────────────────────────────────
for lang in $LANGUAGES; do
    for baseline in $BASELINES; do
        echo ""
        echo "============================================================"
        echo "  Language: $lang | Baseline: $baseline"
        echo "============================================================"

        case "$baseline" in
            naive)
                python3 -m cross_lingual.baseline_naive \
                    --language "$lang" \
                    --output_dir "$OUTPUT_DIR" \
                    --lex_topk "$LEX_TOPK" \
                    --smt_topk "$BEAM_SIZE" \
                    --batch_size "$BATCH_SIZE" \
                    --n_gpu "$N_GPU" \
                    $FORCE
                ;;
            sequential)
                python3 -m cross_lingual.baseline_sequential \
                    --language "$lang" \
                    --output_dir "$OUTPUT_DIR" \
                    --smt_topk "$BEAM_SIZE" \
                    --batch_size "$BATCH_SIZE" \
                    --n_gpu "$N_GPU" \
                    $FORCE
                ;;
            translate)
                python3 -m cross_lingual.baseline_translate \
                    --language "$lang" \
                    --output_dir "$OUTPUT_DIR" \
                    --lex_topk "$LEX_TOPK" \
                    --smt_topk "$BEAM_SIZE" \
                    --batch_size "$BATCH_SIZE" \
                    --n_gpu "$N_GPU" \
                    --translation_model "$TRANSLATION_MODEL" \
                    --translation_batch_size "$TRANSLATION_BATCH" \
                    --device "$DEVICE" \
                    $FORCE
                ;;
            *)
                echo "Unknown baseline: $baseline"
                ;;
        esac

        echo "[runner] $lang / $baseline done."
    done
done

# ── Plan diagnostics ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Running plan diagnostics"
echo "============================================================"
python3 -m cross_lingual.plan_diagnostics \
    --results_dir "$OUTPUT_DIR" \
    --topk 100

# ── Aggregate evaluation ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Aggregating evaluation results"
echo "============================================================"
python3 -m cross_lingual.evaluate \
    --results_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "  RQ3 baselines complete."
echo "  Results: $OUTPUT_DIR"
echo "========================================"

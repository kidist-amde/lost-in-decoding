#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# RQ3 Trained Extension Launcher
#
# Submits SLURM jobs for planner alignment training + evaluation.
#
# Usage:
#   ./cross_lingual/scripts/run_rq3_trained.sh                    # all languages, full pipeline
#   ./cross_lingual/scripts/run_rq3_trained.sh fr                 # French only
#   ./cross_lingual/scripts/run_rq3_trained.sh "fr de"            # French + German
#   ./cross_lingual/scripts/run_rq3_trained.sh all eval-only      # eval-only (skip train)
#   ./cross_lingual/scripts/run_rq3_trained.sh "fr de" eval-only  # eval-only for fr + de
# ──────────────────────────────────────────────────────────────────────────

set -e

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

LANGUAGES="${1:-fr de zh nl}"
MODE="${2:-all}"    # all | eval-only

if [ "$LANGUAGES" = "all" ]; then
    LANGUAGES="fr de zh nl"
fi

OUT_DIR="experiments/RQ3_crosslingual_trained"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "RQ3 Trained Extension Launcher"
echo "  Languages:  $LANGUAGES"
echo "  Mode:       $MODE"
echo "  Output dir: $OUT_DIR"
echo "  Log dir:    $LOG_DIR"
echo "=============================================="

for lang in $LANGUAGES; do
    job_name="rq3_trained_${lang}"
    log_file="${LOG_DIR}/trained_${lang}_%j.log"
    err_file="${LOG_DIR}/trained_${lang}_%j.err"

    echo "Submitting: $job_name (mode=$MODE)"

    EXTRA_SBATCH_ARGS=""
    if [ "$MODE" = "eval-only" ]; then
        EXTRA_SBATCH_ARGS="--time=06:00:00"
    fi

    sbatch \
        --job-name="$job_name" \
        --output="$log_file" \
        --error="$err_file" \
        $EXTRA_SBATCH_ARGS \
        cross_lingual/scripts/run_rq3_trained_sub.sh "$lang" "$MODE"

    sleep 1
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs:    $LOG_DIR"
echo "Results: $OUT_DIR"

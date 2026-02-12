#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# RQ3 Cross-lingual Evaluation Launcher
#
# Submits SLURM jobs for all language/split combinations.
#
# Usage:
#   ./run_rq3_pipeline.sh              # Run all languages, all splits
#   ./run_rq3_pipeline.sh fr           # Run French only, all splits
#   ./run_rq3_pipeline.sh fr dl19      # Run French, DL19 only
# ──────────────────────────────────────────────────────────────────────────

set -e

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Default languages and splits
LANGUAGES="${1:-fr de zh nl}"
SPLITS="${2:-dl19 dl20 dev}"

# Handle "all" shortcuts
if [ "$LANGUAGES" = "all" ]; then
    LANGUAGES="fr de zh nl"
fi
if [ "$SPLITS" = "all" ]; then
    SPLITS="dl19 dl20 dev"
fi

# Create log directory
LOG_DIR="experiments/RQ3_crosslingual/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "RQ3 Cross-lingual Evaluation Launcher"
echo "  Languages: $LANGUAGES"
echo "  Splits:    $SPLITS"
echo "  Log dir:   $LOG_DIR"
echo "=============================================="

# Submit jobs
for lang in $LANGUAGES; do
    for split in $SPLITS; do
        job_name="rq3_${lang}_${split}"
        log_file="${LOG_DIR}/${lang}_${split}_%j.log"
        err_file="${LOG_DIR}/${lang}_${split}_%j.err"

        echo "Submitting: $job_name"

        sbatch \
            --job-name="$job_name" \
            --output="$log_file" \
            --error="$err_file" \
            cross_lingual/scripts/run_rq3_pipeline_sub.sh "$lang" "$split"

        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will be written to: $LOG_DIR"

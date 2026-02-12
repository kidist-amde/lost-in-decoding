#!/bin/bash
#SBATCH --job-name=translate_queries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --output=experiments/RQ3_crosslingual/logs/translate_%j.log
#SBATCH --error=experiments/RQ3_crosslingual/logs/translate_%j.err

# ──────────────────────────────────────────────────────────────────────────
# Pre-translate mMARCO queries to English
#
# This caches all translations so the main RQ3 pipeline doesn't need
# to load the translation model.
#
# Arguments:
#   $1 = language (fr | de | zh | nl) - optional, defaults to all
# ──────────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

nvidia-smi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Create log directory
mkdir -p experiments/RQ3_crosslingual/logs

LANGUAGE="${1:-all}"

echo "========================================"
echo "Translating mMARCO queries to English"
echo "  Language: $LANGUAGE"
echo "========================================"

if [ "$LANGUAGE" = "all" ]; then
    LANGUAGES="fr de zh nl"
else
    LANGUAGES="$LANGUAGE"
fi

for lang in $LANGUAGES; do
    for split in dl19 dl20 dev; do
        echo ""
        echo "--- Translating $lang / $split ---"
        python -m cross_lingual.utils.translator \
            --language "$lang" \
            --split "$split" \
            --model m2m100 \
            --batch_size 32 \
            --output_dir experiments/RQ3_crosslingual/queries
    done
done

echo ""
echo "Done: All translations cached"

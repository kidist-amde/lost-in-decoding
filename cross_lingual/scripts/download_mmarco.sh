#!/bin/bash
#SBATCH --job-name=download_mmarco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=thin
#SBATCH --time=02:00:00
#SBATCH --output=experiments/RQ3_crosslingual/logs/download_mmarco_%j.log
#SBATCH --error=experiments/RQ3_crosslingual/logs/download_mmarco_%j.err

# ──────────────────────────────────────────────────────────────────────────
# Download mMARCO queries for RQ3 languages
# ──────────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

# Create log directory
mkdir -p experiments/RQ3_crosslingual/logs

echo "========================================"
echo "Downloading mMARCO queries"
echo "========================================"

# Download queries for French, German, Chinese, Dutch
# Pass through any extra args (e.g. --force)
python -m cross_lingual.data.mmarco_loader \
    --languages fr de zh nl "$@"

echo "Done: mMARCO queries downloaded"

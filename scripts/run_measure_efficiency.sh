#!/bin/bash
#SBATCH --job-name=table3_efficiency
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=experiments/table3_mk_sweep/logs/%x-%j.out
#SBATCH --error=experiments/table3_mk_sweep/logs/%x-%j.err
#SBATCH --chdir=/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

mkdir -p experiments/table3_mk_sweep/logs

echo "Starting efficiency measurement at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python scripts/measure_efficiency.py \
    --warmup_batches_s1 5 \
    --warmup_batches_s2 2 \
    --out experiments/table3_mk_sweep/efficiency_results.json \
    2>&1 | tee experiments/table3_mk_sweep/logs/efficiency_measurement.log

echo ""
echo "Done at $(date)"

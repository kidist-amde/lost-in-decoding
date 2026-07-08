#!/bin/bash
#SBATCH --job-name=rq2_redecode_dev
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=experiments/RQ2_redecode/logs/%x-%j.out
#SBATCH --error=experiments/RQ2_redecode/logs/%x-%j.err
#SBATCH --chdir=.

set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

mkdir -p experiments/RQ2_redecode/logs

python scripts/rq2_redecode_table3_pipeline.py --splits dev

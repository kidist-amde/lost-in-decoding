#!/bin/bash
#SBATCH --job-name=rq2_redecode_dev_gap
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=experiments/RQ2_redecode/logs/%x-%j.out
#SBATCH --error=experiments/RQ2_redecode/logs/%x-%j.err
#SBATCH --chdir=.

set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

python3 -c "
import sys
sys.path.insert(0, 'scripts')
from rq2_redecode_table3_pipeline import run_pipeline_for_condition
res = run_pipeline_for_condition('dev', 'paraphrase', 2016)
print(res)
"

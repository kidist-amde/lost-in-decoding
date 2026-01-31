#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# RQ2 Robustness Evaluation Launcher
#
# Submits one SLURM job per (split, attack_method, seed) combination.
# Each job runs the full PAG pipeline on clean + perturbed queries.
#
# Usage:
#   bash robustness/scripts/run_rq2_pipeline.sh              # all splits x attacks x seeds
#   bash robustness/scripts/run_rq2_pipeline.sh dl19          # single split, all attacks
#   bash robustness/scripts/run_rq2_pipeline.sh dl19 mispelling  # single split + attack
# ──────────────────────────────────────────────────────────────────────

cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

SPLIT_ARG="${1:-all}"
ATTACK_ARG="${2:-all}"

if [ "$SPLIT_ARG" = "all" ]; then
    split_list=("dl19" "dl20" "dev")
else
    split_list=("$SPLIT_ARG")
fi

if [ "$ATTACK_ARG" = "all" ]; then
    attack_method_list=(
        "mispelling"
        "ordering"
        "synonym"
        "paraphrase"
        "naturality"
    )
else
    attack_method_list=("$ATTACK_ARG")
fi

seed_list=(1999 5 27 2016 2026)

for split in "${split_list[@]}"; do
    for attack_method in "${attack_method_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            echo "Submitting: split=$split attack=$attack_method seed=$seed"
            sbatch robustness/scripts/run_rq2_pipeline_sub.sh \
                "$split" "$attack_method" "$seed"
        done
    done
done

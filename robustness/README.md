# Robustness Evaluation (RQ2)

This module evaluates PAG under controlled query perturbations and reports both retrieval degradation and planner-specific stability metrics.

## What This Module Covers

1. Query perturbation generation (`mispelling`, `ordering`, `synonym`, `paraphrase`, `naturality`).
2. End-to-end PAG evaluation on clean vs perturbed queries.
3. Plan-collapse diagnostics (candidate overlap, token-plan overlap, recovery behavior).
4. Aggregation into summary tables (`csv/json`).

## Directory Map

```text
robustness/
  evaluation/rq2.py                    main RQ2 pipeline
  evaluation/aggregate_results.py      summary aggregation
  evaluation/attack_eval.py            dense retriever attack evaluation
  query_variations/generate_penha.py   perturbation generation
  query_variations/loader.py           perturbation loader and split mapping
  metrics/plan_collapse.py             plan-stability metrics
  scripts/run_rq2_pipeline.sh          SLURM launcher
  scripts/run_rq2_pipeline_sub.sh      single-job SLURM script
  scripts/generate_query_variations.sh perturbation generation launcher
  scripts/evaluate_attack_queries.sh   dense baseline attack launcher
```

## Prerequisites

From repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh

# RQ2 PAG evaluation
conda activate pag-env

# Optional (query generation + dense attack baselines)
# conda activate pag-robustness
```

Required input paths are repository-relative and expected under `data/`.

## Data Expectations

Core query/qrel inputs:

```text
data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv
data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv
data/msmarco-full/dev_queries/raw.tsv
data/msmarco-full/TREC_DL_2019/qrel.json
data/msmarco-full/TREC_DL_2020/qrel.json
data/msmarco-full/dev_qrel.json
```

Expected perturbation files (25 total):

```text
data/msmarco-full/query_variations/msmarco/
  msmarco_test_attacked_queries_seed_<SEED>_attack_method_<METHOD>.json
```

Seeds used in paper pipelines: `1999`, `5`, `27`, `2016`, `2026`.

## Step-by-Step Run

### 1. Generate Query Variations (if missing)

```bash
conda activate pag-robustness
bash robustness/scripts/generate_query_variations.sh
```

Single command alternative:

```bash
python -m robustness.query_variations.generate_penha \
  --dataset msmarco \
  --split test \
  --seed 1999 \
  --attack_method mispelling
```

### 2. Run RQ2 for One Setting

```bash
conda activate pag-env
python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999 \
  --n_gpu 1 \
  --batch_size 8 \
  --lex_topk 1000 \
  --smt_topk 100 \
  --output_dir experiments/RQ2_robustness
```

### 3. Run Full Grid via SLURM

```bash
# all splits x all attacks x 5 seeds
bash robustness/scripts/run_rq2_pipeline.sh

# one split only
bash robustness/scripts/run_rq2_pipeline.sh dl19

# one split + one attack
bash robustness/scripts/run_rq2_pipeline.sh dl19 mispelling
```

### 4. Aggregate Results

```bash
python -m robustness.evaluation.aggregate_results \
  --results_dir experiments/RQ2_robustness \
  --splits dl19 dl20 dev \
  --attacks mispelling ordering synonym paraphrase naturality
```

## Metrics Reported

Retrieval metrics:

- Stage-1 metrics (planner-only retrieval quality)
- Stage-2 metrics (end-to-end PAG quality)
- Delta between clean and perturbed runs

Plan-stability metrics:

- `CandOverlap@100`: overlap of top candidate sets
- `PlanIntersect@100`: overlap of top planner token sets
- `SeqGain`: stage-2 minus stage-1 on perturbed inputs
- `PlanSwapDrop`: normal perturbed run vs plan-swapped decoding
- `Plan collapse` tail events:
  - A query is collapsed if
    `(CandOverlap@k < tau OR TokJaccard@ell < tau) AND DeltaM_SimulOnly <= -delta`
  - `DeltaM_SimulOnly = M(perturbed) - M(clean)` on Stage-1 (planner-only) metrics.
  - Default `tau` is the 10th percentile of `CandOverlap@k`; default `delta=0.05`.
  - Sensitivity sweeps over `tau` percentiles and `delta` are saved to
    `plan_collapse_sensitivity.json`.
  - Lower-tail summary stats (`p10`, `p25`) are reported for overlap metrics in `summary.csv`.

## Expected Outputs

```text
experiments/RQ2_robustness/
  logs/
  queries/
  <DATASET>/
    clean/
      planner_tokens.json
      pag/lex_ret/<DATASET>/run.json
      pag/smt_ret/<DATASET>/run.json
    perturbed/<ATTACK>_seed_<SEED>/
      planner_tokens.json
      plan_collapse_per_query.json
      plan_collapse_sensitivity.json
      per_query_metrics/
      pag/lex_ret/<DATASET>/run.json
      pag/smt_ret/<DATASET>/run.json
      pag_planswap/smt_ret/<DATASET>/run.json
  summary.json
  summary.csv
```

## Observability Checklist

1. Job logs exist in `experiments/RQ2_robustness/logs/`.
2. Each run emits both stage-1 and stage-2 `run.json` files.
3. Planner-token files and plan-collapse json files are present.
4. Aggregation creates `summary.csv` and `summary.json`.

Quick checks:

```bash
find experiments/RQ2_robustness -name "run.json" | wc -l
find experiments/RQ2_robustness -name "plan_collapse_per_query.json" | wc -l
ls experiments/RQ2_robustness/summary*.csv
```

## Evaluate-Only Mode

Use `--eval_only` when inference already finished and you only want metric recomputation:

```bash
python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999 \
  --eval_only \
  --output_dir experiments/RQ2_robustness
```

Plan-collapse controls (optional):

```bash
python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999 \
  --collapse_metric auto \
  --collapse_tau_percentile 10 \
  --collapse_delta 0.05 \
  --collapse_sensitivity_tau_percentiles 5,10,15,20,25 \
  --collapse_sensitivity_deltas 0.01,0.03,0.05,0.10
```

## Common Issues

- Missing perturbation JSON: run generation step first.
- Missing checkpoint/docid mapping: verify expected `data/experiments-*` paths.
- SLURM jobs complete but no summaries: run aggregation manually.
- CUDA OOM: lower `--batch_size`.
- Partial outputs from interrupted jobs: rerun only missing split/attack/seed combinations.

## Dense Baseline Attack Evaluation

`robustness/evaluation/attack_eval.py` and related scripts support dense retriever attack experiments using the same perturbation files. Keep these outputs separate from PAG RQ2 outputs for clean comparison.

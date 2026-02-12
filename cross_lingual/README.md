# Cross-Lingual Evaluation (RQ3)

This module evaluates PAG when queries are non-English but the retrieval corpus remains English.

## Scope

RQ3 compares four settings:

1. `naive_pag`: non-English query directly into PAG.
2. `seq_only`: planner-disabled sequential baseline.
3. `translate_pag`: query translated to English, then PAG.
4. `plan_swap`: target-language query with an English lexical plan (causal probe).

Languages currently used: `fr`, `de`, `zh`, `nl`.
Splits: `dl19`, `dl20`, `dev`.

## Directory Map

```text
cross_lingual/
  evaluation/rq3.py                 main RQ3 pipeline
  evaluation/aggregate_results.py   aggregation and summary tables
  data/mmarco_loader.py             mMARCO loader/downloader
  scripts/run_rq3_pipeline.sh       SLURM launcher (all combos)
  scripts/run_rq3_pipeline_sub.sh   single SLURM job
  scripts/download_mmarco.sh        query download wrapper
  scripts/translate_queries.sh      translation cache utility
  metrics/planner_diagnostics.py    planner-specific diagnostics
```

## Prerequisites

From repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
```

Required assets (shared with other tracks):

- PAG checkpoints and identifier mappings in `data/experiments-*`
- MS MARCO corpus and qrels in `data/msmarco-full/`

## Step-by-Step Run

### 1. Download mMARCO Queries

```bash
bash cross_lingual/scripts/download_mmarco.sh --force
```

Direct alternative:

```bash
python -m cross_lingual.data.mmarco_loader --languages fr de zh nl --force
```

### 2. (Optional) Pre-cache Translations

```bash
sbatch cross_lingual/scripts/translate_queries.sh
```

### 3. Run Evaluation

Single run:

```bash
python -m cross_lingual.evaluation.rq3 \
  --language fr \
  --split dev \
  --n_gpu 1 \
  --batch_size 8
```

Batch submission:

```bash
# all languages, all splits
bash cross_lingual/scripts/run_rq3_pipeline.sh all all

# one language, all splits
bash cross_lingual/scripts/run_rq3_pipeline.sh fr all

# one language, one split
bash cross_lingual/scripts/run_rq3_pipeline.sh fr dl19
```

### 4. Aggregate Results

```bash
python -m cross_lingual.evaluation.aggregate_results \
  --results_dir experiments/RQ3_crosslingual
```

## Expected Outputs

```text
experiments/RQ3_crosslingual/
  logs/
  queries/
  english/<split>/...
  fr/<split>/...
  de/<split>/...
  zh/<split>/...
  nl/<split>/...
  summary.json
  summary.csv
  aggregates.json
  latex/
```

Per language/split outputs include system-specific run files and planner diagnostics.

## Observability Checklist

1. `logs/*.log` and `logs/*.err` exist for each submitted job.
2. Each language/split has non-empty result directories.
3. `summary.csv` is generated after aggregation.
4. Diagnostics files (planner-token overlap, candidate recall, etc.) are present for completed runs.

Quick checks:

```bash
find experiments/RQ3_crosslingual -name "run.json" | wc -l
ls experiments/RQ3_crosslingual/summary*.csv
```

## Reproducibility Guidelines

- Keep language/split lists explicit in logs.
- Use consistent `--batch_size` and GPU count across comparisons.
- Archive command lines and commit hash with each run.
- Keep translated query caches versioned by model and language.

## Common Issues

- Missing mMARCO files: rerun `download_mmarco.sh --force`.
- Translation model download failures: check network/proxy and HuggingFace auth.
- Empty result folders: inspect SLURM `.err` files and confirm checkpoint paths.
- Inconsistent metrics between runs: verify same seed/config and no partial overwrites.

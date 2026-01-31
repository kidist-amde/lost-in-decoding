# Robustness Toolkit (RQ2/RQ3)

This package contains query-variation generators, robustness evaluation entry points, and analysis utilities for PAG experiments. It is organized so that query perturbation generation, evaluation, and metrics live in clearly named modules and can be run reproducibly from the repo root.

## Folder Structure

```
robustness/
├── evaluation/                 # Evaluation entry points
│   ├── rq2.py                  # RQ2 pipeline (PAG robustness to query perturbations)
│   └── attack_eval.py          # Dense-retrieval evaluation under query attacks
├── metrics/                    # Metrics and analysis helpers
│   └── plan_collapse.py        # Plan-collapse + retrieval metric helpers
├── query_variations/           # Query-variation generation + loaders
│   ├── generate_penha.py       # Penha-style variation generator (mispelling, ordering, etc.)
│   ├── loader.py               # Loads MSMARCO variations + split mapping
│   └── penha/                  # Penha transformation implementations
│       ├── transformations_mispelling.py
│       ├── transformations_ordering.py
│       ├── transformations_synonym.py
│       ├── transformations_paraphrase.py
│       └── transformations_naturality.py
├── scripts/                    # SLURM wrappers for batch runs
│   ├── run_rq2_pipeline.sh
│   ├── run_rq2_pipeline_sub.sh
│   ├── generate_query_variations.sh
│   ├── generate_query_variations_sub.sh
│   ├── evaluate_attack_queries.sh
│   └── evaluate_attack_queries_sub.sh
├── utils/                      # Shared helpers (data/model loading, BEIR helpers)
└── README.md                   # This file
```

## RQ2: Robustness to Query Perturbations

RQ2 evaluates how PAG handles realistic query noise by measuring retrieval quality drops and plan-collapse metrics (candidate overlap, size ratio, rank correlation) and sequential recovery.

### Required Inputs and Paths

RQ2 uses pre-generated MSMARCO query variations. By default, the loader expects:

```
data/msmarco-full/query_variations/msmarco/
```

Files should be named like:

```
msmarco_test_attacked_queries_seed_<SEED>_attack_method_<METHOD>.json
```

Evaluation queries and qrels are read from:

```
data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv
data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv
data/msmarco-full/dev_queries/raw.tsv

data/msmarco-full/TREC_DL_2019/qrel.json
data/msmarco-full/TREC_DL_2019/qrel_binary.json
data/msmarco-full/TREC_DL_2020/qrel.json
data/msmarco-full/TREC_DL_2020/qrel_binary.json
data/msmarco-full/dev_qrel.json
```

Model and docid mapping defaults are defined in `robustness/utils/pag_inference.py`:

- `MODEL_DIR` / `PRETRAINED_PATH`
- `LEX_DOCID_PATH`
- `SMT_DOCID_PATH`

If your paths differ, update those constants before running.

### Single-Run (Local) Command

```bash
# From repo root
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

# Activate environment
conda activate pag-env

python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999
```

### Evaluate-Only Mode (Skip Inference)

```bash
python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999 \
  --eval_only
```

### Batch Runs (SLURM)

```bash
# All splits x attacks x seeds
bash robustness/scripts/run_rq2_pipeline.sh

# Single split
bash robustness/scripts/run_rq2_pipeline.sh dl19

# Single split + attack
bash robustness/scripts/run_rq2_pipeline.sh dl19 mispelling
```

### Expected Outputs

By default, outputs are written to:

```
experiments/RQ2_robustness/
```

Structure:

```
experiments/RQ2_robustness/
├── summary.json                        # All results
├── summary.csv                         # Flat CSV for analysis
├── queries/                            # Prepared query TSVs
│   ├── TREC_DL_2019/
│   │   ├── clean/raw.tsv
│   │   └── perturbed/mispelling_seed_1999/raw.tsv
│   ├── TREC_DL_2020/...
│   └── MSMARCO/...
├── TREC_DL_2019/
│   ├── clean/pag/
│   │   ├── lex_ret/<dataset>/run.json   # Lexical planner output
│   │   └── smt_ret/<dataset>/run.json   # Sequential decoder output
│   └── perturbed/<attack_seed>/pag/
│       ├── lex_ret/<dataset>/run.json
│       ├── smt_ret/<dataset>/run.json
│       └── plan_collapse_per_query.json
├── TREC_DL_2020/...
└── MSMARCO/...
```

## Query Variation Generation

Penha-style generators produce JSON/CSV perturbations for downstream evaluation.

### Single Run

```bash
python -m robustness.query_variations.generate_penha \
  --dataset msmarco \
  --split test \
  --attack_method mispelling \
  --seed 1999
```

Outputs are written under:

```
robustness/output_attack/attacked_text/query/<dataset>/
```

### Batch Runs (SLURM)

```bash
bash robustness/scripts/generate_query_variations.sh
```

## Dense Retrieval Attack Evaluation

Evaluate dense retrieval models under query perturbations:

```bash
python -m robustness.evaluation.attack_eval \
  --dataset msmarco \
  --model_name contriever \
  --attack_method mispelling \
  --seed 1999
```

SLURM wrapper:

```bash
bash robustness/scripts/evaluate_attack_queries.sh
```

## Dependencies

Uses the existing PAG codebase (`t5_pretrainer/`). Additional:

- `pytrec_eval` (already required for PAG evaluation)
- `scipy` (optional, for rank correlation in plan-collapse metrics)

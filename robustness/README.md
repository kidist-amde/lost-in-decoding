# Robustness Evaluation Toolkit

This package evaluates PAG (Planning Ahead in Generative Retrieval) and dense retrieval models under realistic query perturbations. It supports two research questions:

- **RQ2**: How sensitive is PAG's lexical planner to query noise, and does the sequential decoder recover from plan degradation?
- **RQ3** (dense baselines): How do standard dense retrieval models perform under the same perturbations?

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Prerequisites](#prerequisites)
3. [Data Layout](#data-layout)
4. [RQ2: PAG Robustness Evaluation](#rq2-pag-robustness-evaluation)
   - [Overview](#overview)
   - [Metrics](#metrics)
   - [Single-Run Commands](#single-run-commands)
   - [Batch SLURM Submission](#batch-slurm-submission)
   - [Evaluate-Only Mode (Skip Inference)](#evaluate-only-mode)
   - [Result Consolidation](#result-consolidation)
   - [Output Structure](#output-structure)
5. [Running All 5 Seeds and Reporting Mean ± Std (Example: DL19)](#running-all-5-seeds-and-reporting-mean--std-example-dl19)
6. [Query Variation Generation](#query-variation-generation)
7. [Dense Retrieval Attack Evaluation (RQ3)](#dense-retrieval-attack-evaluation)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Directory Structure

```
robustness/
├── __init__.py
├── README.md                                   # This file
├── evaluation/                                 # Evaluation entry points
│   ├── __init__.py
│   ├── rq2.py                                  # RQ2 pipeline (PAG robustness)
│   ├── aggregate_results.py                    # Aggregate results -> LaTeX tables
│   └── attack_eval.py                          # Dense retrieval attack evaluation
├── metrics/                                    # Metrics and analysis helpers
│   ├── __init__.py
│   └── plan_collapse.py                        # Plan-collapse + retrieval metrics
├── query_variations/                           # Query perturbation generation + loading
│   ├── __init__.py
│   ├── generate_penha.py                       # Penha-style variation generator
│   ├── loader.py                               # Load pre-generated variations & split mapping
│   └── penha/                                  # Penha transformation implementations
│       ├── __init__.py
│       ├── transformations_mispelling.py       # QWERTY keyboard typos
│       ├── transformations_ordering.py         # Word-order shuffling
│       ├── transformations_synonym.py          # Semantic synonym replacement
│       ├── transformations_paraphrase.py       # Back-translation + seq2seq paraphrasing
│       └── transformations_naturality.py       # Stop-word removal + T5 summarisation
├── scripts/                                    # SLURM batch-job wrappers
│   ├── run_rq2_pipeline.sh                     # RQ2 launcher (all combinations)
│   ├── run_rq2_pipeline_sub.sh                 # RQ2 single-job SLURM script
│   ├── generate_query_variations.sh            # Query generation launcher
│   ├── generate_query_variations_sub.sh        # Query generation single job
│   ├── evaluate_attack_queries.sh              # Dense retrieval launcher
│   └── evaluate_attack_queries_sub.sh          # Dense retrieval single job
└── utils/                                      # Shared utilities
    ├── __init__.py
    ├── pag_inference.py                        # PAG inference pipeline wrapper
    ├── beir_exact_search.py                    # FAISS dense retrieval (BEIR)
    ├── beir_utils.py                           # Dense encoder model classes
    ├── load_model.py                           # HuggingFace / SentenceTransformer loading
    ├── load_data.py                            # BEIR data loading
    ├── data_loader.py                          # Generic corpus/query/qrel loader
    ├── normalize_text.py                       # Unicode text normalisation
    ├── logging.py                              # tqdm-compatible logging handler
    ├── dist_utils.py                           # torch.distributed gather functions
    └── utils.py                                # Text processing, metric helpers
```

---

## Prerequisites

### Environment

All RQ2 experiments must run inside the `pag-env` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
```

Query variation generation and dense retrieval evaluation (RQ3) use `pag-robustness`:

```bash
conda activate pag-robustness
```

### Python Dependencies

**RQ2 (pag-env)**:

| Package | Purpose |
|---------|---------|
| `torch` | Model inference (GPU) |
| `transformers` | Tokeniser, T5 model backbone |
| `pytrec_eval` | NDCG@10, MRR@10, Recall@100 evaluation |
| `scipy` | Spearman rank correlation (optional, for plan-collapse analysis) |
| `numpy` | Numerical operations |
| `tqdm` | Progress bars |

**Query variation generation (pag-robustness)**:

| Package | Purpose |
|---------|---------|
| `torch`, `transformers` | T5 paraphrase models |
| `textattack` | Word-order shuffling (`WordInnerSwapRandom`) |
| `pandas` | CSV output of variations |
| `wandb` | Experiment logging |

**Dense retrieval (pag-robustness)**:

| Package | Purpose |
|---------|---------|
| `beir` | BEIR benchmark framework |
| `faiss-gpu` | Approximate nearest neighbour search |
| `sentence-transformers` | Encoder model wrappers |
| `wandb` | Experiment logging |

### Hardware

- **RQ2**: 1x NVIDIA H100 GPU, 180 GB RAM, 10 CPU cores, up to 6 hours per job
- **Query generation**: 1x GPU, 70 GB RAM, up to 20 hours per job
- **Dense retrieval**: 2x GPUs, 280 GB RAM, up to 1 hour per job

---

## Data Layout

All data paths are relative to the repository root (`/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/`).

### Model Checkpoint and Document-ID Mappings

Defined in `robustness/utils/pag_inference.py`:

| Constant | Path | Description |
|----------|------|-------------|
| `PRETRAINED_PATH` | `data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/checkpoint` | Trained PAG (LexicalRipor) checkpoint |
| `LEX_DOCID_PATH` | `data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json` | Lexical document-ID to BOW token-ID mapping |
| `SMT_DOCID_PATH` | `data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json` | Semantic document-ID to semantic token-ID mapping |

### Query Files

Original queries (one per split):

```
data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv     # 43 queries
data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv     # 54 queries
data/msmarco-full/dev_queries/raw.tsv                    # 6980 queries
```

Format: `<qid>\t<query_text>` (tab-separated, no header).

### Relevance Judgements (qrels)

```
data/msmarco-full/TREC_DL_2019/qrel.json           # Graded relevance (for NDCG)
data/msmarco-full/TREC_DL_2019/qrel_binary.json    # Binary relevance (for MRR, Recall)
data/msmarco-full/TREC_DL_2020/qrel.json
data/msmarco-full/TREC_DL_2020/qrel_binary.json
data/msmarco-full/dev_qrel.json                     # Binary (MS MARCO dev)
```

Format: JSON dict `{qid: {docid: relevance_int, ...}, ...}`.

### Pre-Generated Query Variations

```
data/msmarco-full/query_variations/msmarco/
    msmarco_test_attacked_queries_seed_<SEED>_attack_method_<METHOD>.json
```

- 25 files total: 5 attack methods x 5 seeds.
- Format: JSON dict `{qid: perturbed_query_text, ...}`.
- Covers all MS MARCO test queries; the loader filters to per-split qids.

**Seeds**: `1999`, `5`, `27`, `2016`, `2026`

**Attack methods**:
| Method | Key | Description |
|--------|-----|-------------|
| Misspelling | `mispelling` | QWERTY keyboard typos (DL_Typo-compatible), avoids stopwords and short words |
| Word ordering | `ordering` | Random word-order shuffling via TextAttack's `WordInnerSwapRandom` |
| Synonym | `synonym` | Semantic synonym replacement using word embeddings (cosine > 0.8) or USE (> 0.7) |
| Paraphrase | `paraphrase` | Back-translation (EN -> DE -> EN) + T5 seq2seq paraphrasing |
| Naturality | `naturality` | Stop-word removal (20% of words) + T5 summarisation |

---

## RQ2: PAG Robustness Evaluation

### Overview

The RQ2 pipeline (`robustness/evaluation/rq2.py`) runs five steps for each (split, attack_method, seed) triple:

1. **Prepare query TSVs** -- Write clean and perturbed queries to `raw.tsv` files the PAG pipeline expects.
2. **Run PAG on clean queries** -- Stage 1 (lexical planner, top-1000) + Stage 2 (constrained sequential decoder, top-100).
3. **Run PAG on perturbed queries** -- Identical pipeline on the attacked query set.
4. **Extract planner tokens** -- Save the top-100 highest-scoring vocabulary tokens from the planner's `[bz, vocab_size]` output for each query (clean and perturbed). Used for PlanIntersect@100.
5. **Plan-swapped decoding** -- Run Stage 2 on perturbed queries but supply the lexical plan from clean queries. Used for PlanSwapDrop.

### Metrics

#### Table 1: Retrieval Quality Under Query Perturbations

| Metric | Source | Description |
|--------|--------|-------------|
| S1-NDCG@10 | Stage 1 (lexical planner only) | NDCG at cutoff 10 |
| S1-MRR@10 | Stage 1 | Mean Reciprocal Rank at cutoff 10 |
| S2-NDCG@10 | Stage 2 (full PAG pipeline) | End-to-end NDCG at cutoff 10 |
| S2-MRR@10 | Stage 2 | End-to-end MRR at cutoff 10 |
| Delta | Clean - Perturbed | Absolute performance drop |

#### Table 2: Plan Stability and Sensitivity

| Metric | Source | Description |
|--------|--------|-------------|
| CandOverlap@100 | `candidate_overlap()` | Jaccard overlap of top-100 candidate document sets (clean vs. perturbed planner output) |
| PlanIntersect@100 | `plan_intersect()` | Jaccard overlap of top-100 planner vocabulary tokens (the "plan") between clean and perturbed queries |
| SeqGain | `S2_metric - S1_metric` | Marginal gain of Stage 2 over Stage 1 on perturbed queries |
| PlanSwapDrop | `normal_PAG - planswap_PAG` | Performance drop when replacing perturbed-query plan with clean-query plan. Positive = wrong plan hurts |

Additional diagnostic metrics (saved but not in paper tables):
- **Candidate size ratio**: `|perturbed_candidates| / |clean_candidates|`
- **Spearman rank correlation**: Rank correlation of document scores in planner output
- **Recovery delta**: `drop_lex - drop_smt` (positive = sequential decoder mitigates degradation)

### Single-Run Commands

All commands assume you are at the repository root and `pag-env` is activated.

```bash
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro
conda activate pag-env

# Run full pipeline for one (split, attack, seed) triple
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

**CLI arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `dl19` | Evaluation split: `dl19`, `dl20`, `dev`, or `all` |
| `--attack_method` | `mispelling` | Attack: `mispelling`, `ordering`, `synonym`, `paraphrase`, `naturality`, or `all` |
| `--seed` | `1999` | Seed: `1999`, `5`, `27`, `2016`, `2026`, or `all` |
| `--n_gpu` | `1` | Number of GPUs for Stage 2 (`torch.distributed.launch`) |
| `--batch_size` | `8` | Batch size for Stage 1 and planner token extraction (Stage 2 uses `batch_size * 2`) |
| `--lex_topk` | `1000` | Top-K for Stage 1 (lexical planner) |
| `--smt_topk` | `100` | Top-K for Stage 2 (sequential decoder) |
| `--output_dir` | `experiments/RQ2_robustness` | Root output directory |
| `--eval_only` | `false` | Skip inference; compute metrics from existing run files only |
| `--variation_dir` | auto | Override directory for query variation JSONs |
| `--max_variants_per_qid` | `1` | Max variations per query (currently 1 per JSON) |

### Batch SLURM Submission

The launcher script submits one SLURM job per (split, attack_method, seed) combination:

```bash
# All splits x default attacks (mispelling + paraphrase) x 5 seeds = 30 jobs
bash robustness/scripts/run_rq2_pipeline.sh

# Single split, default attacks = 10 jobs
bash robustness/scripts/run_rq2_pipeline.sh dl19

# Single split + single attack = 5 jobs
bash robustness/scripts/run_rq2_pipeline.sh dl19 mispelling

# All splits x all 5 attack methods x 5 seeds = 75 jobs
bash robustness/scripts/run_rq2_pipeline.sh all all5
```

**SLURM resource request** (per job, defined in `run_rq2_pipeline_sub.sh`):

```
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=10
#SBATCH --time=00-06:00:00
```

Logs are written to `experiments/RQ2_robustness/rq2_pag_robust-<JOBID>.{out,err}`.

### Evaluate-Only Mode

When inference outputs already exist (e.g. from completed SLURM jobs), you can skip GPU inference and only recompute metrics:

```bash
python -m robustness.evaluation.rq2 \
    --split dl19 \
    --attack_method mispelling \
    --seed 1999 \
    --eval_only
```

This reads the existing `run.json` files and recomputes all Table 1 and Table 2 metrics. Requires `pytrec_eval` (available in `pag-env`).

### Result Consolidation

Since parallel SLURM jobs each write their own `summary.csv`, a consolidation step is needed after all jobs complete to produce the combined summary:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

# Consolidate all 30 results into one summary.csv
python -c "
import sys; sys.path.insert(0, '.')
from robustness.evaluation.rq2 import evaluate_single, write_summary

results = []
for split in ['dl19', 'dl20', 'dev']:
    for attack in ['mispelling', 'paraphrase']:
        for seed in [1999, 5, 27, 2016, 2026]:
            r = evaluate_single(
                split=split, attack_method=attack, seed=seed,
                output_dir='experiments/RQ2_robustness', eval_only=True,
            )
            results.append(r)

write_summary(results, 'experiments/RQ2_robustness')
"

# Aggregate over seeds and produce LaTeX-ready tables
python -m robustness.evaluation.aggregate_results --latex
```

**Aggregate results CLI** (`robustness/evaluation/aggregate_results.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | `experiments/RQ2_robustness` | Directory containing `summary.csv` |
| `--attacks` | `mispelling paraphrase` | Attack methods to include in tables |
| `--splits` | `dl19 dl20 dev` | Splits to include |
| `--latex` | `false` | Also print LaTeX-formatted values for copy-paste |

### Output Structure

```
experiments/RQ2_robustness/
├── summary.json                                          # All results (JSON)
├── summary.csv                                           # Flat CSV for analysis
├── queries/                                              # Prepared query TSV files
│   ├── TREC_DL_2019/
│   │   ├── clean/raw.tsv                                 # Clean queries
│   │   └── perturbed/
│   │       ├── mispelling_seed_1999/raw.tsv
│   │       ├── mispelling_seed_5/raw.tsv
│   │       ├── ...
│   │       └── paraphrase_seed_2026/raw.tsv
│   ├── TREC_DL_2020/
│   │   └── (same structure)
│   └── msmarco_dev/
│       └── (same structure)
│
├── TREC_DL_2019/                                         # Results per split
│   ├── clean/
│   │   ├── pag/
│   │   │   ├── lex_ret/TREC_DL_2019/
│   │   │   │   ├── run.json                              # Stage 1 lexical planner output
│   │   │   │   └── perf.json                             # Stage 1 metrics
│   │   │   └── smt_ret/TREC_DL_2019/
│   │   │       ├── run.json                              # Stage 2 sequential decoder output
│   │   │       └── perf.json                             # Stage 2 metrics
│   │   └── planner_tokens.json                           # Top-100 planner token IDs per query
│   │
│   └── perturbed/
│       └── mispelling_seed_1999/
│           ├── pag/
│           │   ├── lex_ret/TREC_DL_2019/run.json         # Perturbed Stage 1 output
│           │   └── smt_ret/TREC_DL_2019/run.json         # Perturbed Stage 2 output
│           ├── pag_planswap/
│           │   └── smt_ret/TREC_DL_2019/run.json         # Plan-swapped Stage 2 output
│           ├── planner_tokens.json                        # Perturbed planner token IDs
│           └── plan_collapse_per_query.json               # Per-query plan-collapse stats
│
├── TREC_DL_2020/                                         # Same structure as TREC_DL_2019
│   └── ...
├── MSMARCO/                                              # Same structure (dev split)
│   └── ...
│
└── rq2_pag_robust-<JOBID>.{out,err}                      # SLURM job logs
```

**Key output files**:

| File | Format | Content |
|------|--------|---------|
| `run.json` | `{qid: {docid: score, ...}}` | Ranked document lists with scores |
| `perf.json` | `{NDCG@10: float, MRR@10: float, ...}` | Aggregate retrieval metrics |
| `planner_tokens.json` | `{qid: [token_id, ...]}` | Top-100 vocabulary token IDs from planner output |
| `plan_collapse_per_query.json` | `{qid: {jaccard: float, plan_intersect: float, ...}}` | Per-query plan-collapse diagnostics |
| `summary.csv` | CSV | One row per (split, attack, seed); all Table 1 and Table 2 metrics |

**`summary.csv` columns**:

```
split, attack_method, seed, n_queries,
clean_lex_NDCG@10, clean_lex_MRR@10,
pert_lex_NDCG@10, pert_lex_MRR@10,
clean_smt_NDCG@10, clean_smt_MRR@10,
pert_smt_NDCG@10, pert_smt_MRR@10,
swap_smt_NDCG@10, swap_smt_MRR@10,
delta_lex_NDCG@10, delta_lex_MRR@10,
delta_smt_NDCG@10, delta_smt_MRR@10,
recovery_delta_NDCG@10, recovery_delta_MRR@10,
CandOverlap@100, PlanIntersect@100,
avg_rank_correlation@100, avg_size_ratio,
SeqGain_MRR@10, SeqGain_NDCG@10,
PlanSwapDrop_MRR@10, PlanSwapDrop_NDCG@10
```

---

## Running All 5 Seeds and Reporting Mean ± Std (Example: DL19)

To produce paper-ready results with variance estimates, run all 5 seeds for a given (split, attack) pair and then aggregate.

### Step 1: Run All 5 Seeds

**Option A — Single process (sequential, no SLURM):**

```bash
conda activate pag-env
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

# All 5 seeds for DL19 + misspelling (seeds: 1999, 5, 27, 2016, 2026)
python -m robustness.evaluation.rq2 \
    --split dl19 \
    --attack_method mispelling \
    --seed all \
    --n_gpu 1 \
    --batch_size 8
```

The `--seed all` flag loops over all 5 seeds sequentially. Each seed's results are appended to `experiments/RQ2_robustness/summary.csv`.

**Option B — SLURM (parallel, one job per seed):**

```bash
# Submits 5 jobs (one per seed) for DL19 + misspelling
bash robustness/scripts/run_rq2_pipeline.sh dl19 mispelling
```

To run all 5 attack methods × 5 seeds = 25 jobs:

```bash
bash robustness/scripts/run_rq2_pipeline.sh dl19
```

### Step 2: Aggregate Over Seeds (Mean ± Std)

After all seeds complete, aggregate the per-seed rows into mean ± std tables:

```bash
# Single attack method
python -m robustness.evaluation.aggregate_results \
    --splits dl19 \
    --attacks mispelling \
    --latex

# All 5 attack methods
python -m robustness.evaluation.aggregate_results \
    --splits dl19 \
    --attacks mispelling ordering synonym paraphrase naturality \
    --latex
```

This reads `experiments/RQ2_robustness/summary.csv`, groups rows by (split, attack_method), and computes sample mean and standard deviation over the 5 seeds. It prints three tables:

- **Table 1**: Retrieval performance (S1-NDCG@10, S1-MRR@10, S2-NDCG@10, S2-MRR@10) with deltas, formatted as `mean±std`
- **Table 2**: Plan stability (CandOverlap@100, PlanIntersect@100, SeqGain, PlanSwapDrop, collapse rate) as `mean±std`
- **Table 3**: Distributional statistics (percentiles) for CandOverlap and TokJaccard

With `--latex`, it additionally prints LaTeX-formatted values (e.g. `0.6543$\pm$0.0123`) ready for copy-paste into paper tables.

### Step 2b: Consolidation After Parallel SLURM Jobs

When seeds ran as separate SLURM jobs, each job writes its own 1-row `summary.csv` (overwriting the previous). **No inference data is lost.** The actual inference outputs are saved as individual `run.json` files per seed in separate directories that never conflict:

```
experiments/RQ2_robustness/TREC_DL_2019/
├── clean/pag/lex_ret/TREC_DL_2019/run.json       # Clean Stage 1 (shared)
├── clean/pag/smt_ret/TREC_DL_2019/run.json       # Clean Stage 2 (shared)
├── perturbed/mispelling_seed_1999/pag/.../run.json
├── perturbed/mispelling_seed_5/pag/.../run.json
├── perturbed/mispelling_seed_27/pag/.../run.json
├── perturbed/mispelling_seed_2016/pag/.../run.json
└── perturbed/mispelling_seed_2026/pag/.../run.json
```

Only `summary.csv` (a derived summary) gets overwritten. It can be rebuilt at any time from the `run.json` files using `--eval_only`. Before aggregating, re-consolidate all results:

**Single attack method (5 seeds):**

```bash
conda activate pag-env
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

python -m robustness.evaluation.rq2 \
    --split dl19 \
    --attack_method mispelling \
    --seed all \
    --eval_only

# Then aggregate
python -m robustness.evaluation.aggregate_results \
    --splits dl19 \
    --attacks mispelling \
    --latex
```

**All 5 attack methods × 5 seeds (25 jobs):**

When you ran all 25 jobs in parallel (`bash robustness/scripts/run_rq2_pipeline.sh dl19`), the same overwrite issue applies — all 25 jobs write to the same `summary.csv`. After all jobs finish, consolidate everything in one command:

```bash
conda activate pag-env
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

python -m robustness.evaluation.rq2 \
    --split dl19 \
    --attack_method all \
    --seed all \
    --eval_only

# Then aggregate over all 5 attack methods
python -m robustness.evaluation.aggregate_results \
    --splits dl19 \
    --attacks mispelling ordering synonym paraphrase naturality \
    --latex
```

This rebuilds `summary.csv` with all 25 rows (5 attacks × 5 seeds), then produces mean ± std tables for each attack method.

The `--eval_only` flag skips GPU inference and only recomputes metrics from existing `run.json` files, rebuilding the combined `summary.csv` with all seed rows.

---

## Query Variation Generation

Pre-generated variations already exist at `data/msmarco-full/query_variations/msmarco/`. To regenerate or generate for a new dataset:

### Single Run

```bash
conda activate pag-robustness
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

python -m robustness.query_variations.generate_penha \
    --dataset msmarco \
    --split test \
    --attack_method mispelling \
    --seed 1999
```

**CLI arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `fiqa` | BEIR dataset name (`msmarco`, `nq`, `fiqa`, etc.) |
| `--split` | `test` | Dataset split |
| `--seed` | `1999` | Random seed for reproducible variations |
| `--attack_method` | `naturality` | `mispelling`, `ordering`, `synonym`, `paraphrase`, `naturality` |
| `--attack_word_change_rate` | `0.2` | Word change rate (used by some generators) |
| `--uqv_model_path` | `./` | Path for custom UQV paraphrase checkpoints |

Outputs are saved to:

```
robustness/output_attack/attacked_text/query/<dataset>/
    <dataset>_<split>_attacked_queries_seed_<SEED>_attack_method_<METHOD>.json
    <dataset>_<split>_attacked_queries_seed_<SEED>_attack_method_<METHOD>.csv
```

### Batch SLURM Submission

```bash
bash robustness/scripts/generate_query_variations.sh
```

Submits 25 jobs (5 attack methods x 5 seeds) for the MS MARCO dataset.

**SLURM resource request** (per job):

```
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=9
#SBATCH --time=00-20:00:00
```

---

## Dense Retrieval Attack Evaluation

Evaluates dense retrieval models under query perturbations (and optionally corpus poisoning) using BEIR.

### Supported Models

Defined in `robustness/scripts/evaluate_attack_queries.sh`:

| Model key | Description |
|-----------|-------------|
| `contriever` | Contriever-MSMARCO (dot product scoring) |
| `bge_m3` | BGE-M3 |
| `qwen3` | Qwen3 embedding |
| `linq` | LinQ embedding |
| `gte` | GTE embedding |
| `reasonir` | ReasonIR |
| `diver` | Diver |
| `bge_reasoner` | BGE-Reasoner |
| `qwen3_4B` | Qwen3-4B |
| `qwen3_0.6B` | Qwen3-0.6B |

### Single Run

```bash
conda activate pag-robustness
cd /gpfs/work4/0/prjs1037/dpo-exp/pag-repro

python -m robustness.evaluation.attack_eval \
    --dataset msmarco \
    --model_name contriever \
    --attack_method mispelling \
    --seed 1999
```

**CLI arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `nq` | BEIR dataset name |
| `--model_name` | `linq` | Model key (see table above) |
| `--split` | `test` | Dataset split |
| `--seed` | `1999` | Seed (`1999`, `5`, `27`, `2016`, `2026`, or `all`) |
| `--attack_method` | `none` | `none`, `mispelling`, `ordering`, `synonym`, `paraphrase`, `naturality`, `supervised_poisoning` |
| `--per_gpu_eval_batch_size` | `64` | Batch size |
| `--attacked_num` | `50` | Number of adversarial documents (for corpus poisoning) |
| `--no_wandb` | `false` | Disable W&B logging |

### Batch SLURM Submission

```bash
bash robustness/scripts/evaluate_attack_queries.sh
```

Submits jobs for all (dataset, model, attack, seed) combinations. Default: 1 dataset x 10 models x 6 attacks x 5 seeds = 300 jobs.

**SLURM resource request** (per job):

```
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --mem=280G
#SBATCH --cpus-per-task=10
#SBATCH --time=00-01:00:00
```

Results are saved to:

```
robustness/output_attack/scores/<model_name>/<dataset>_<split>/
    attack_method_<METHOD>_seed_<SEED>_attacked_num_<N>/
        retrieval_results.json       # Full ranked lists
        merged_scores.json           # Per-query metric scores
        metrics_scores_and_asr.json  # Aggregate metrics + ASR@K
```

---

## Configuration Reference

### Key Constants

**`robustness/utils/pag_inference.py`**:

| Constant | Value |
|----------|-------|
| `MODEL_DIR` | `data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/` |
| `PRETRAINED_PATH` | `{MODEL_DIR}/checkpoint` |
| `LEX_DOCID_PATH` | `data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json` |
| `SMT_DOCID_PATH` | `data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json` |

**`robustness/query_variations/loader.py`**:

| Constant | Value |
|----------|-------|
| `ATTACK_METHODS` | `["mispelling", "ordering", "synonym", "paraphrase", "naturality"]` |
| `SEEDS` | `[1999, 5, 27, 2016, 2026]` |
| `VARIATION_DIR` | `data/msmarco-full/query_variations/msmarco/` |

### Split-to-Dataset Mapping

The `t5_pretrainer` pipeline uses specific dataset directory names internally. The mapping is:

| Split key | Dataset name | Query path directory |
|-----------|-------------|---------------------|
| `dl19` | `TREC_DL_2019` | `queries/TREC_DL_2019/` |
| `dl20` | `TREC_DL_2020` | `queries/TREC_DL_2020/` |
| `dev` | `MSMARCO` | `queries/msmarco_dev/` |

The query directory names must contain strings that `t5_pretrainer.utils.utils.get_dataset_name()` recognises (case-sensitive): `TREC_DL_2019`, `TREC_DL_2020`, or lowercase `msmarco`.

### PAG Pipeline Internals

The PAG pipeline runs as subprocesses to avoid polluting the main process with torch/GPU state:

1. **Stage 1**: `python -m t5_pretrainer.evaluate --task=lexical_constrained_retrieve_and_rerank`
   - Model runs in mode `lex_retrieval`
   - Produces `[bz, vocab_size]` scores via `log(1 + ReLU(logit))`
   - Scores documents by summing planner token scores over each document's BOW token set (`TermEncoderRetriever`)
   - Output: `lex_ret/<dataset>/run.json`

2. **Stage 2**: `python -m torch.distributed.launch -m t5_pretrainer.evaluate --task=lexical_constrained_retrieve_and_rerank_2`
   - Model runs in mode `smt_retrieval`
   - Uses `BatchPrefixerForLexInc` to constrain beam search with lexical plan
   - Uses `BatchLexTmpReScorer` for re-scoring
   - Outputs per-GPU shards `run_*.json`, merged into `run.json`
   - Output: `smt_ret/<dataset>/run.json`

3. **Planner token extraction**: Runs as inline subprocess (`_EXTRACT_TOKENS_SCRIPT`).
   - Loads `LexicalRipor` model
   - Calls `model.encode()` in `lex_retrieval` mode
   - Extracts `torch.topk(batch_preds, k=100)` per query
   - Output: `planner_tokens.json`

4. **Plan-swapped decoding**: Runs Stage 2 but supplies `lex_out_dir` from clean queries instead of perturbed queries. This tests sensitivity to plan correctness.

---

## Troubleshooting

### `pytrec_eval` not found

`pytrec_eval` is only installed in `pag-env`. If running consolidation from a login shell:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
```

### SLURM jobs overwrite `summary.csv`

Each parallel SLURM job writes its own 1-row `summary.csv`, overwriting the previous one. This is expected. After all jobs complete, run the [consolidation step](#result-consolidation) to produce the combined summary.

### Stage 2 shard merge assertion error

The upstream `t5_pretrainer` has an assertion `len(sub_paths) == torch.cuda.device_count()` that can fail when visible GPUs differ from `nproc_per_node`. The `merge_and_evaluate()` function in `pag_inference.py` reimplements the merge logic without this assertion.

### Dataset name not recognised

The `t5_pretrainer` pipeline determines the dataset name from the query directory path. Ensure query directories contain the exact strings `TREC_DL_2019`, `TREC_DL_2020`, or `msmarco` (lowercase). The `loader.py` module handles this automatically via the `split_label` mapping.

### W&B initialisation errors (query generation)

`generate_penha.py` initialises `wandb.init()` unconditionally. To disable, either set `WANDB_MODE=disabled` or modify the script.

```bash
WANDB_MODE=disabled python -m robustness.query_variations.generate_penha ...
```

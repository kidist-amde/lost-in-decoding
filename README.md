[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](https://sigir2026.org/en-AU/pages/submissions/reproducibility-track)
[![Submission](https://img.shields.io/badge/submission-298-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)

# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval

This repository contains code and experiment pipelines for three evaluation tracks:

1. `RQ1`: artifact-level PAG reproduction (released checkpoints + released identifiers).
2. `RQ2`: robustness under query perturbations.
3. `RQ3`: cross-lingual query shift and mitigation.

## Repository Layout

- `t5_pretrainer/`: core PAG model/training/inference codepaths.
- `full_scripts/`: legacy/full pipeline scripts used by RQ1-style runs.
- `robustness/`: RQ2 evaluation, perturbation generation, and aggregation.
- `cross_lingual/`: RQ3 evaluation and diagnostics.
- `scripts/`, `tools/`: utility scripts for efficiency and plotting.
- `data/`: datasets, checkpoints, and intermediate artifacts (not fully versioned).
- `experiments/`: run outputs (metrics, logs, summaries).

## Quick Start

### 1. Environment

Use Conda (recommended):

```bash
source ~/miniconda3/etc/profile.d/conda.sh

# Main evaluation env
conda activate pag-env

# Optional (query-variation generation / dense-attack tooling)
# conda env create -f environment.yml
# conda activate pag-robustness
```

### 2. Verify Required Inputs

Expected core inputs:

```text
data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/checkpoint/
data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json
data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json
data/msmarco-full/full_collection/
data/msmarco-full/dev_queries/raw.tsv
data/msmarco-full/dev_qrel.json
data/msmarco-full/TREC_DL_2019/
data/msmarco-full/TREC_DL_2020/
```

Quick check:

```bash
for p in \
  data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/checkpoint \
  data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json \
  data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json \
  data/msmarco-full/full_collection \
  data/msmarco-full/dev_queries/raw.tsv \
  data/msmarco-full/dev_qrel.json

do
  [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"
done
```

## Reproduction Entry Points

### RQ1: PAG Artifact Reproduction

Run evaluation script:

```bash
bash full_scripts/full_lexical_ripor_evaluate.sh
```

Typical outputs:

- `data/experiments-full-lexical-ripor/.../run.json`
- evaluation json files (`perf*.json`)

### RQ2: Robustness to Query Perturbations

Single run:

```bash
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

Batch launcher:

```bash
bash robustness/scripts/run_rq2_pipeline.sh
```

Aggregate:

```bash
python -m robustness.evaluation.aggregate_results \
  --results_dir experiments/RQ2_robustness \
  --splits dl19 dl20 dev \
  --attacks mispelling ordering synonym paraphrase naturality
```

See `robustness/README.md` for full details.

### RQ3: Cross-Lingual Query Shift

Download multilingual queries:

```bash
bash cross_lingual/scripts/download_mmarco.sh --force
```

Run RQ3 (single language/split):

```bash
python -m cross_lingual.evaluation.rq3 \
  --language fr \
  --split dev \
  --n_gpu 1 \
  --batch_size 8
```

Batch launcher:

```bash
bash cross_lingual/scripts/run_rq3_pipeline.sh all all
```

Aggregate:

```bash
python -m cross_lingual.evaluation.aggregate_results \
  --results_dir experiments/RQ3_crosslingual
```

See `cross_lingual/README.md` for full details.

## Observability and Run Validation

Use these checks to confirm runs progressed correctly:

1. Log files are created under each task's `experiments/.../logs/` directory.
2. Per-run metric files exist (e.g., `metrics_scores_and_asr.json`, `run.json`, `planner_tokens.json`).
3. Aggregation commands emit `summary.csv` and `summary.json`.
4. Re-running aggregation does not change results unless upstream outputs changed.

Useful checks:

```bash
# Count generated run files
find experiments -name "run.json" | wc -l

# Confirm RQ2 summary exists
ls experiments/RQ2_robustness/summary*.csv

# Confirm RQ3 summary exists
ls experiments/RQ3_crosslingual/summary*.csv
```

## Reproducibility Notes

- Use fixed seeds where scripts provide them (`1999`, `5`, `27`, `2016`, `2026`).
- Keep `lex_topk` / `smt_topk` consistent when comparing runs.
- Prefer writing outputs to fresh subdirectories when testing modifications.
- Record environment versions (`conda env export > env_snapshot.yml`) for archival.

## Troubleshooting

- `ModuleNotFoundError`: ensure you run from repository root and activate the correct conda env.
- Missing `run.json`: inspect SLURM stderr logs first; common causes are missing checkpoint/data paths.
- Empty aggregation output: verify all expected split/attack/seed combinations completed.
- OOM in evaluation: reduce `--batch_size` or run with fewer GPUs/processes.

## Submodule Guides

- `robustness/README.md`: perturbation generation, RQ2 pipeline, dense-attack evaluation.
- `cross_lingual/README.md`: mMARCO setup, RQ3 execution, diagnostics, aggregation.
- `cross_lingual/trained_extension/README.md`: Trained extension workflow.

## License

Apache 2.0. See `LICENSE`.

## Upstream Work

This repository reproduces and stress-tests the **PAG (Planner-Assisted Generative retrieval)** system.

**Paper:**
> Zeng, H., & Zamani, H. (2024). *Planning Ahead in Generative Retrieval: Guiding Autoregressive
Generation through Simultaneous Decoding.*
> [arXiv:2404.14600](https://arxiv.org/pdf/2404.14600)

**Upstream repository:**
> [https://github.com/HansiZeng/PAG/tree/main/t5_pretrainer](https://github.com/HansiZeng/PAG/tree/main/t5_pretrainer)

Our work builds directly on the released checkpoints and document identifiers from the upstream PAG repository. All three research questions (RQ1–RQ3) use the original PAG model as their baseline.

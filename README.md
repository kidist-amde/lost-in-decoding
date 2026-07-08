[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](https://sigir2026.org/en-AU/pages/submissions/reproducibility-track)
[![Submission](https://img.shields.io/badge/submission-298-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.23396)

# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval

> ## Evaluation Correction Notice for RQ2

> [!IMPORTANT]
> **The camera-ready paper has already been finalized, so this README serves as
> the public correction notice for RQ2.**

> [!WARNING]
> **The RQ2 numbers reported in the camera-ready Table 6 should be replaced by
> the corrected values below.** The original RQ2 run used a decoding/evaluation
> path that gave lower clean baselines than the verified PAG reproduction path.
> Re-running RQ2 with the verified setup raises the clean TREC-DL baselines from
> **0.6688 to 0.7003** on DL19 and from **0.6214 to 0.7021** on DL20.

> [!NOTE]
> **The conclusion is unchanged.** PAG remains brittle to surface-form and
> semantic perturbations: misspelling, synonym replacement, and paraphrasing still
> cause large drops, while word reordering is small or near-neutral. What changes
> is the exact RQ2 table, not the interpretation.

> [!CAUTION]
> **Status:** DL19 and DL20 are complete and independently re-scored. Dev is
> still incomplete, so Dev entries are left as `--` until all seeds finish.

---

This repository contains code and experiment pipelines for three evaluation tracks:

1. `RQ1`: artifact-level PAG reproduction (released checkpoints + released identifiers).
2. `RQ2`: robustness under query perturbations.
3. `RQ3`: cross-lingual query shift and mitigation.

## Corrected RQ2 Results

The tables below replace the camera-ready RQ2 values. They report mean ± std
over five seeds; Δ means `clean − perturbed`, using the clean run from the same
verified setup. Metrics follow the upstream PAG convention: NDCG@10 on graded
qrels and Recall@10 on the binary qrels.

### TREC-DL 2019 — clean: NDCG@10 = 0.7003, Recall@10 = 0.2650

| Variation    | NDCG@10 (mean ± std) | Δ vs clean       | Recall@10 (mean ± std) |
| ------------ | -------------------- | ---------------- | ---------------------- |
| Misspelling  | 0.5067 ± 0.0245      | 0.1937           | 0.1897 ± 0.0130        |
| Reordering   | 0.7013 ± 0.0057      | −0.0010 (≈ none) | 0.2683 ± 0.0050        |
| Synonym      | 0.5745 ± 0.0316      | 0.1258           | 0.2049 ± 0.0145        |
| Paraphrase   | 0.6374 ± 0.0215      | 0.0630           | 0.2475 ± 0.0046        |
| Naturality\* | 0.6756 ± 0.0000      | 0.0247           | 0.2592 ± 0.0000        |

### TREC-DL 2020 — clean: NDCG@10 = 0.7021, Recall@10 = 0.2370

| Variation    | NDCG@10 (mean ± std) | Δ vs clean | Recall@10 (mean ± std) |
| ------------ | -------------------- | ---------- | ---------------------- |
| Misspelling  | 0.5196 ± 0.0170      | 0.1825     | 0.1663 ± 0.0157        |
| Reordering   | 0.6809 ± 0.0104      | 0.0212     | 0.2199 ± 0.0101        |
| Synonym      | 0.5592 ± 0.0218      | 0.1430     | 0.1894 ± 0.0098        |
| Paraphrase   | 0.5881 ± 0.0233      | 0.1140     | 0.2079 ± 0.0048        |
| Naturality\* | 0.6789 ± 0.0000      | 0.0232     | 0.2262 ± 0.0000        |

### MS MARCO Dev — clean: MRR@10 = --, Recall@10 = -- (re-decode in progress)

| Variation   | MRR@10 (mean ± std) | Δ vs clean | Recall@10 (mean ± std) |
| ----------- | ------------------- | ---------- | ---------------------- |
| Misspelling | --                  | --         | --                     |
| Reordering  | --                  | --         | --                     |
| Synonym     | --                  | --         | --                     |
| Paraphrase  | --                  | --         | --                     |
| Naturality  | --                  | --         | --                     |

> \*Naturality has zero seed variance because the naturality perturbation is
> identical across seeds. This was verified separately and is not a scoring bug.

**Takeaway:** the corrected numbers still support the paper's RQ2 claim. PAG is
most sensitive to misspellings, synonyms, and paraphrases, while reordering has a
much smaller effect. The camera-ready numerical table should be updated, but the
planner-brittleness conclusion remains the same.

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

> [!TIP]
> To reproduce the corrected RQ2 values, use the same verified PAG setup for the
> clean and perturbed runs, then compute each delta within that matched run.

Single run:

```bash
python -m robustness.evaluation.rq2 \
  --split dl19 \
  --attack_method mispelling \
  --seed 1999 \
  --n_gpu 1 \
  --batch_size 16 \
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

> [!TIP]
> Use fresh output directories for new experiments so old and corrected results
> are not mixed.

## Reproducibility Notes

> [!TIP]
> - Use fixed seeds where scripts provide them (`1999`, `5`, `27`, `2016`, `2026`).
> - Keep `lex_topk` / `smt_topk` consistent when comparing runs.
> - Record environment versions (`conda env export > env_snapshot.yml`) for archival.
> - Always compare clean and perturbed runs from the same matched setup.

### Metric Convention

> [!NOTE]
> Metrics match upstream PAG. NDCG@10 uses graded qrels, Recall@10 uses binary
> qrels, and rankings are evaluated at the top 10 using the upstream tie-handling
> convention.

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
> [https://github.com/HansiZeng/PAG](https://github.com/HansiZeng/PAG/tree/main)

Our work builds directly on the released checkpoints and document identifiers from the upstream PAG repository. All three research questions (RQ1–RQ3) use the original PAG model as their baseline.

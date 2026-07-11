[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](https://sigir2026.org/en-AU/pages/submissions/reproducibility-track)
[![Submission](https://img.shields.io/badge/submission-298-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.23396)

# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval

This repository contains code and experiment pipelines for three evaluation tracks:

1. `RQ1`: artifact-level PAG reproduction (released checkpoints + released identifiers).
2. `RQ2`: robustness under query perturbations.
3. `RQ3`: cross-lingual query shift and mitigation.

## Correction to the RQ2 Results

> [!IMPORTANT]
> **The RQ2 results reported in Table 6 of the camera-ready paper have been corrected.**
>
> The original RQ2 experiments did not use the same decoding and evaluation path as the verified Table 3 reproduction. Consequently, the clean baselines in Table 6 were lower than the corresponding Table 3 results.
>
> To correct this inconsistency, we re-decoded every RQ2 condition—both clean and perturbed—using the exact Table 3 pipeline:
>
> * the same model checkpoint;
> * the same lexical and semantic identifier files;
> * the same `--topk` and `--max_new_token_for_docid` settings;
> * the same `--lex_constrained` configuration; and
> * the same two-stage constrained-decoding procedure.
>
> All reported differences are calculated as
>
> `Δ = clean − perturbed`
>
> using the clean baseline produced by the corrected pipeline for the corresponding evaluation split. No clean value is copied from a different experiment or configuration.

> [!NOTE]
> **The correction does not change the findings or conclusions of the paper.**
>
> PAG remains sensitive to changes in query wording. Misspellings, synonym replacement, and paraphrasing consistently produce substantial retrieval-quality degradation, whereas word reordering and naturality normalization have considerably smaller effects.
>
> The corrected values below have not yet been incorporated into the arXiv version of the paper. They will be included in a future revision.

### Correction scope

The corrected evaluation covers:

* three evaluation splits: MS MARCO Dev, TREC-DL 2019, and TREC-DL 2020;
* five query variations: misspelling, reordering, synonym replacement, paraphrasing, and naturality;
* five perturbation seeds: `1999`, `5`, `27`, `2016`, and `2026`.

In total, we re-decoded and re-evaluated:

* 75 perturbed runs: 3 splits × 5 variations × 5 seeds; and
* 3 clean runs, one for each split.

This gives **78 runs in total**. The corrected clean results reproduce the Table 3 headline values within the small variation observed across repeated constrained-decoding runs.

---

## Corrected RQ2 Results

Unless otherwise stated, the tables report the mean and standard deviation over five perturbation seeds.

For each split:

`Δ = corrected clean score − mean perturbed score`

Before evaluation, each run is stably sorted by descending retrieval score and truncated to the top 10 results. NDCG@10 is computed using the released graded qrels, while Recall@10 and MRR@10 use the released binary qrels.

<!-- ### Note on the TREC-DL 2020 binary qrels

> [!NOTE]
> NIST defines passage-level relevance for TREC-DL 2019 and 2020 using grades of at least 2. Grade 1, labelled *Related*, is not considered relevant for passage retrieval; the grade-at-least-1 threshold applies to document retrieval.
>
> However, the `qrel_binary.json` file distributed with the upstream PAG data package for TREC-DL 2020 contains binary values corresponding to a grade-at-least-1 threshold. This repository contains an unmodified copy of the upstream data package.
>
> For direct comparability with the original PAG results, we therefore compute TREC-DL 2020 Recall@10 and MRR@10 using the distributed file as provided. We document the discrepancy here rather than silently modifying or regenerating the qrels. -->

### Note on decoding variability

> [!NOTE]
> The constrained beam-search implementation used by this pipeline does not expose a decoding seed. As a result, repeated runs with identical inputs and settings may produce small differences in individual query rankings. Aggregate scores are generally stable within a few thousandths.
>
> For this reason, all differences below use the clean score produced by the same corrected pipeline and configuration, rather than the clean score from the original Table 6 experiments.

---

### TREC-DL 2019

**Corrected clean baseline:** NDCG@10 = **0.7003**, Recall@10 = **0.2650**

| Query variation           | NDCG@10, mean ± std |          Δ vs. clean | Recall@10, mean ± std |
| ------------------------- | ------------------: | -------------------: | --------------------: |
| Misspelling               |     0.5067 ± 0.0245 |               0.1937 |       0.1897 ± 0.0130 |
| Reordering                |     0.7013 ± 0.0057 | −0.0010 (negligible) |       0.2683 ± 0.0050 |
| Synonym replacement       |     0.5745 ± 0.0316 |               0.1258 |       0.2049 ± 0.0145 |
| Paraphrasing              |     0.6374 ± 0.0215 |               0.0630 |       0.2475 ± 0.0046 |
| Naturality normalization* |     0.6756 ± 0.0000 |               0.0247 |       0.2592 ± 0.0000 |

---

### TREC-DL 2020

**Corrected clean baseline:** NDCG@10 = **0.7021**, Recall@10 = **0.2370**

| Query variation           | NDCG@10, mean ± std | Δ vs. clean | Recall@10, mean ± std |
| ------------------------- | ------------------: | ----------: | --------------------: |
| Misspelling               |     0.5196 ± 0.0170 |      0.1825 |       0.1663 ± 0.0157 |
| Reordering                |     0.6809 ± 0.0104 |      0.0212 |       0.2199 ± 0.0101 |
| Synonym replacement       |     0.5592 ± 0.0218 |      0.1430 |       0.1894 ± 0.0098 |
| Paraphrasing              |     0.5881 ± 0.0233 |      0.1140 |       0.2079 ± 0.0048 |
| Naturality normalization* |     0.6789 ± 0.0000 |      0.0232 |       0.2262 ± 0.0000 |

---

### MS MARCO Dev

**Corrected clean baseline:** MRR@10 = **0.3856**, Recall@10 = **0.6706**

| Query variation           | MRR@10, mean ± std | Δ vs. clean | Recall@10, mean ± std |
| ------------------------- | -----------------: | ----------: | --------------------: |
| Misspelling               |    0.2456 ± 0.0016 |      0.1400 |       0.4528 ± 0.0021 |
| Reordering                |    0.3771 ± 0.0005 |      0.0085 |       0.6582 ± 0.0015 |
| Synonym replacement       |    0.2888 ± 0.0021 |      0.0968 |       0.5210 ± 0.0036 |
| Paraphrasing              |    0.3190 ± 0.0028 |      0.0666 |       0.5688 ± 0.0041 |
| Naturality normalization* |    0.3660 ± 0.0000 |      0.0196 |       0.6438 ± 0.0000 |

* The released naturality perturbation is identical across the five seed directories. It is a deterministic style-normalization transformation rather than a randomized perturbation. Consequently, the five evaluations produced identical aggregate scores in these experiments, resulting in zero measured seed variance.

---

## Summary of the Correction

The camera-ready RQ2 experiments reported a clean MS MARCO Dev MRR@10 of **0.362**. After running RQ2 through the verified Table 3 pipeline, the corrected clean score is **0.386**, consistent with the headline reproduction result.

The corrected results support the same overall conclusion as the camera-ready paper:

* misspellings cause the largest degradation across all three splits;
* synonym replacement and paraphrasing also produce substantial losses;
* word reordering has little or no effect on TREC-DL 2019, a small but consistent effect on MS MARCO Dev, and a modest effect on TREC-DL 2020, comparable in size to naturality normalization on that split; and
* naturality normalization produces relatively small degradation.

The numerical values in the camera-ready RQ2 table should therefore be replaced by the corrected results above. The interpretation of the experiment and the paper’s conclusion regarding PAG’s sensitivity to query variation remain unchanged.

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
> To reproduce the corrected RQ2 values, decode the clean and perturbed query
> sets through the same verified Table 3 setup, then compute each Δ within
> that matched (clean, perturbed) pair.

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
> Use fresh output directories for new experiments so old and corrected
> results are not mixed. When re-decoding into an existing experiment tree,
> guard against overwriting a prior run's output directory — reusing an
> output path across two different jobs can silently overwrite a prior run's
> outputs. A path-reuse issue of this kind affected the original RQ2
> artifacts, so we re-decoded every condition into a fresh, isolated output
> tree.

## Reproducibility Notes

> [!TIP]
> - Use fixed seeds where scripts provide them (`1999`, `5`, `27`, `2016`, `2026`).
> - Keep `lex_topk` / `smt_topk` consistent when comparing runs.
> - Record environment versions (`conda env export > env_snapshot.yml`) for archival.
> - Always compare clean and perturbed runs from the same matched, re-decoded setup.
> - Constrained beam decoding is unseeded; expect small run-to-run wobble in
>   per-query rankings even with an identical config. Aggregate metrics are
>   stable to within a few thousandths across repeats.

### Metric Convention

> [!NOTE]
> Metrics match the upstream PAG convention: a stable descending-score sort of
> the run truncated to the top 10, NDCG@10 computed on the graded qrels, and
> Recall@10 computed on the released binary qrels. The binary-qrel relevance
> threshold is per split as released: TREC-DL 2019 uses grade ≥ 2, TREC-DL
> 2020 uses grade ≥ 1. MS MARCO Dev uses its single released (binary) qrel
> file for both MRR@10 and Recall@10.

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

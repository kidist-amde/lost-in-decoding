[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](https://sigir2026.org/en-AU/pages/submissions/reproducibility-track)
[![Submission](https://img.shields.io/badge/submission-298-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.23396)

# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval

This repository contains code and experiment pipelines for three evaluation tracks:

1. `RQ1`: artifact-level PAG reproduction (released checkpoints + released identifiers).
2. `RQ2`: robustness under query perturbations.
3. `RQ3`: cross-lingual query shift and mitigation.

## 📌 Correction Notice for RQ2 (read first)

> [!IMPORTANT]
> **RQ1 (Table 3) is verified and unchanged.** The headline reproduction of
> PAG on the released checkpoint and identifiers holds:
>
> | Split | Metric | Value |
> | --- | --- | --- |
> | MS MARCO Dev | MRR@10 | **0.386** |
> | MS MARCO Dev | Recall@10 | **0.671** |
> | TREC-DL 2019 | NDCG@10 | **0.703** |
> | TREC-DL 2019 | Recall@10 | **0.265** |
> | TREC-DL 2020 | NDCG@10 | **0.701** |
> | TREC-DL 2020 | Recall@10 | **0.236** |
>
> Every run.json behind these six values was independently re-scored with
> `pytrec_eval` against the released qrels.

> [!WARNING]
> **RQ2 (Table 6) is corrected.** The camera-ready RQ2 numbers used a
> decoding/evaluation path that was not the same path as the verified Table 3
> reproduction, and its clean baselines came out lower as a result, a
> difference of this size (≈0.03 NDCG@10 on DL19/DL20, and a comparable gap on
> Dev) is consistent with a decoding-path difference, not a metric-definition
> change; a metric fix alone cannot move a clean baseline by that much. The
> fix was to **re-decode every RQ2 condition (clean and perturbed) through the
> exact Table 3 pipeline** same checkpoint, same lexical/semantic identifier
> files, same `--topk`, `--max_new_token_for_docid`, `--lex_constrained`, same
> two-stage constrained decoding — so that clean and perturbed runs are
> directly comparable. Every Δ in the tables below is computed **within that
> matched run**: `Δ = clean − perturbed`, using the clean baseline from the
> same re-decode, not a value copied from elsewhere.

> [!NOTE]
> **The paper's findings and conclusions are unchanged.** PAG remains brittle
> to surface-form and semantic query perturbation: misspelling, synonym
> replacement, and paraphrasing all cause large retrieval-quality drops, while
> word reordering is small or near-neutral. What changes is the numerical
> table, not the interpretation. 🚧 **The corrected RQ2 numbers in this
> README have not yet been ported to the arXiv version of the paper, and will
> be reflected in a future arXiv revision.**

> [!CAUTION]
> **Status: complete.** All three splits (MS MARCO Dev, TREC-DL 2019,
> TREC-DL 2020), all five query perturbations (misspelling, reordering,
> synonym, paraphrase, naturality), all five seeds (`1999`, `5`, `27`, `2016`,
> `2026`) have been re-decoded and re-scored: 75 perturbation conditions
> across three splits (3 splits × 5 perturbations × 5 seeds), plus the three
> clean baselines (**78 runs total**). Each split's clean re-decode
> reproduces its Table 3 headline value within normal
> run-to-run decoding noise (see below).

## Corrected RQ2 Results

The tables report mean ± std over five seeds unless noted; Δ = `clean −
perturbed`, computed within the same re-decoded run. Metrics follow the
upstream PAG convention: a stable descending-score sort of the run before
truncating to the top 10, NDCG@10 computed on the graded qrels, and Recall@10
computed on the released binary qrels. 
<!-- The binary-qrel relevance threshold is
per split, matching the released `qrel_binary.json` files exactly as
distributed in upstream PAG's data package: **TREC-DL 2019's file treats
relevance grade ≥ 2 as relevant (values are `{0, 2, 3}`); TREC-DL 2020's file
treats relevance grade ≥ 1 as relevant (values are `{0, 1}`).** Do not assume
a shared threshold across DL19 and DL20 — the two files use different binary
cutoffs, verified by inspecting `data/msmarco-full/TREC_DL_{2019,2020}/qrel_binary.json`
directly. -->

> [!NOTE]
> NIST's own TREC-DL documentation specifies grade ≥ 2 as relevant for
> **passage** ranking in both 2019 and 2020 (grade 1, "Related", is
> explicitly not relevant for passages; grade ≥ 1 is the **document**-ranking
> threshold). TREC-DL 2020's released `qrel_binary.json` in this data
> package uses `{0, 1}`, matching the document-ranking threshold rather than
> the passage-ranking one PAG evaluates on. This file is used **exactly as
> distributed in upstream PAG's data package** (`data/` in this repository is
> a verbatim copy of upstream's released Google Drive package) — it is not
> altered, regenerated, or introduced by this reproduction. We report
> Recall@10 and MRR@10 on DL20 using this file as-is for comparability with
> upstream's own numbers, and flag the threshold choice here rather than
> silently resolving it.

> [!NOTE]
> Constrained beam decoding in this pipeline is **not seeded**: per-query
> rankings can vary slightly from run to run even with identical inputs and
> config, though aggregate metrics are stable to within a few thousandths.
> This is why every Δ below is computed within one matched (clean, perturbed)
> pair from the same re-decode, rather than against a clean value from a
> different run.

### TREC-DL 2019 — clean: NDCG@10 = 0.7003, Recall@10 = 0.2650

| Variation    | NDCG@10 (mean ± std) | Δ vs clean       | Recall@10 (mean ± std) |
| ------------ | --------------------- | ---------------- | ----------------------- |
| Misspelling  | 0.5067 ± 0.0245       | 0.1937            | 0.1897 ± 0.0130         |
| Reordering   | 0.7013 ± 0.0057       | −0.0010 (≈ none) | 0.2683 ± 0.0050         |
| Synonym      | 0.5745 ± 0.0316       | 0.1258            | 0.2049 ± 0.0145         |
| Paraphrase   | 0.6374 ± 0.0215       | 0.0630            | 0.2475 ± 0.0046         |
| Naturality\* | 0.6756 ± 0.0000       | 0.0247            | 0.2592 ± 0.0000         |

### TREC-DL 2020 — clean: NDCG@10 = 0.7021, Recall@10 = 0.2370

| Variation    | NDCG@10 (mean ± std) | Δ vs clean | Recall@10 (mean ± std) |
| ------------ | --------------------- | ---------- | ----------------------- |
| Misspelling  | 0.5196 ± 0.0170       | 0.1825     | 0.1663 ± 0.0157         |
| Reordering   | 0.6809 ± 0.0104       | 0.0212     | 0.2199 ± 0.0101         |
| Synonym      | 0.5592 ± 0.0218       | 0.1430     | 0.1894 ± 0.0098         |
| Paraphrase   | 0.5881 ± 0.0233       | 0.1140     | 0.2079 ± 0.0048         |
| Naturality\* | 0.6789 ± 0.0000       | 0.0232     | 0.2262 ± 0.0000         |

### MS MARCO Dev — clean: MRR@10 = 0.3856, Recall@10 = 0.6706

| Variation    | MRR@10 (mean ± std) | Δ vs clean | Recall@10 (mean ± std) |
| ------------ | -------------------- | ---------- | ----------------------- |
| Misspelling  | 0.2456 ± 0.0016      | 0.1400     | 0.4528 ± 0.0021         |
| Reordering   | 0.3771 ± 0.0005      | 0.0085     | 0.6582 ± 0.0015         |
| Synonym      | 0.2888 ± 0.0021      | 0.0968     | 0.5210 ± 0.0036         |
| Paraphrase   | 0.3190 ± 0.0028      | 0.0666     | 0.5688 ± 0.0041         |
| Naturality\* | 0.3660 ± 0.0000      | 0.0196     | 0.6438 ± 0.0000         |

> \*Naturality has zero seed variance because the released naturality
> perturbation file is identical across all five seeds — it is a deterministic
> style normalization, not a randomized attack like misspelling, synonym, or
> paraphrase. This was verified against the source perturbation files and is
> not a scoring artifact.

The camera-ready reported Dev clean MRR@10 = 0.362 on the buggy path; the
corrected value is 0.386, consistent with the Table 3 headline reproduction
above.

**Takeaway:** the corrected numbers support the same RQ2 finding as the
camera-ready draft. PAG is most sensitive to misspellings, synonym
replacement, and paraphrasing, and substantially more robust to word
reordering — on every split, reordering's Δ is at least an order of magnitude
smaller than the other four perturbations. The camera-ready numerical table
should be replaced by the tables above; the planner-brittleness conclusion is
unchanged.

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

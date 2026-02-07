[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](#)
[![Submission](https://img.shields.io/badge/submission-298-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

# Lost in Decoding? Reproducing and Stress-Testing the Lexical Planner in Generative Retrieval

This repository contains the code and experiment pipelines for:
1. Reproducing inference-time PAG results with released artifacts (RQ1).
2. Stress-testing planner robustness under query perturbations (RQ2).
3. Evaluating cross-lingual query shift and mitigation strategies (RQ3).

Primary paper artifact in this repo: `root/PAG_Repro (3).pdf`.

## What This Repo Contains

- `t5_pretrainer/`: core PAG training/inference codepaths used by the original model family.
- `full_scripts/`: legacy/full training and evaluation shell scripts for PAG and ablations.
- `robustness/`: RQ2 pipelines for perturbation generation, plan-collapse metrics, and aggregation.
- `cross_lingual/`: RQ3 pipelines (naive, sequential-only, translate-at-inference, plan-swap), diagnostics, and aggregation.
- `data/`: checkpoints, docid mappings, corpora/queries/qrels, and generated intermediate artifacts.
- `experiments/`: output artifacts from RQ1/RQ2/RQ3 runs.

## Step-by-Step Guides by Folder

For detailed reproduction steps, each main repro area has its own guide:
1. Top-level full pipeline notes: `follow_the_steps.md`
2. Robustness pipeline (RQ2): `robustness/README.md`
3. Cross-lingual pipeline (RQ3): `cross_lingual/README.md`
4. Trained cross-lingual extension: `cross_lingual/trained_extension/README.md`

## Reproducibility Matrix

| Scope | Entry point | Output root | Main outputs |
|---|---|---|---|
| RQ1 artifact-level reproduction | `full_scripts/full_lexical_ripor_evaluate.sh` | `data/experiments-full-lexical-ripor/...` | run files + `perf*.json` for MSMARCO / DL19 / DL20 |
| RQ2 robustness | `python -m robustness.evaluation.rq2` or `robustness/scripts/run_rq2_pipeline.sh` | `experiments/RQ2_robustness` | per-run metrics, plan-collapse JSONs, `summary*.csv/json` |
| RQ3 cross-lingual | `python -m cross_lingual.evaluation.rq3` or `cross_lingual/scripts/run_rq3_pipeline.sh` | `experiments/RQ3_crosslingual` | per-system metrics, planner-token diagnostics, `summary*.csv/json` |
| RQ3 trained extension (optional) | `cross_lingual/trained_extension/run_rq3_trained_extension.sh` | `cross_lingual/trained_extension` | adapted checkpoints + trained-extension summaries |

## Environment and Prerequisites

This codebase uses two environments in practice:
1. `pag-env` for core PAG, RQ2, and RQ3 evaluation scripts.
2. `pag-robustness` for query-variation generation and dense baseline attack evaluations.

The checked-in `environment.yml` is named `pag-robustness` and includes CUDA/FAISS-heavy dependencies. There is also a lightweight `requirements.txt` for legacy PAG dependencies.

### Minimal activation used by main pipelines

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env
```

### Optional environment creation from file

```bash
conda env create -f environment.yml
conda activate pag-robustness
```

## Required Data and Checkpoints

The repository assumes the following are present (inference/reproduction path):

```text
data/
+-- experiments-full-lexical-ripor/
|   +-- lexical_ripor_direct_lng_knp_seq2seq_1/checkpoint/
|   +-- t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json
+-- experiments-splade/
|   +-- t5-splade-0-12l/top_bow/docid_to_tokenids.json
+-- msmarco-full/
    +-- full_collection/
    +-- dev_queries/raw.tsv
    +-- dev_qrel.json
    +-- TREC_DL_2019/queries_2019/raw.tsv
    +-- TREC_DL_2019/qrel.json
    +-- TREC_DL_2019/qrel_binary.json
    +-- TREC_DL_2020/queries_2020/raw.tsv
    +-- TREC_DL_2020/qrel.json
    +-- TREC_DL_2020/qrel_binary.json
```

Reference release source for PAG artifacts: `root/README.md` and the linked PAG data bundle.

### Quick integrity check

```bash
for p in \
  data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/checkpoint \
  data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json \
  data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json \
  data/msmarco-full/full_collection \
  data/msmarco-full/dev_queries/raw.tsv \
  data/msmarco-full/TREC_DL_2019/queries_2019/raw.tsv \
  data/msmarco-full/TREC_DL_2020/queries_2020/raw.tsv \
  data/msmarco-full/dev_qrel.json
do
  [ -e "$p" ] && echo "OK  $p" || echo "MISS $p"
done
```

## RQ1: Reproducing PAG Inference Results

RQ1 in this repository is artifact-level/inference-level reproduction: evaluate released checkpoint + released identifiers/trie under matching decoding settings.

### Run

Edit task/preset near the top of `full_scripts/full_lexical_ripor_evaluate.sh` and execute:

```bash
bash full_scripts/full_lexical_ripor_evaluate.sh
```

Common presets in that script:
1. `task=lexical_constrained_retrieve_and_rerank` for standard PAG evaluation.
2. `task=constrained_beam_search_for_doc_ret_by_sub_tokens` for beam/subtoken sweeps.
3. `task=constrained_beam_search_for_qid_rankdata` for sequential-only style runs.

### Expected artifacts

- `data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/sub_tokenid_8_out_100/*/run.json`
- `data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/doc_ret_by_sub_tokens/ret_*/...`
- `perf*.json` files generated by evaluation branches in `t5_pretrainer.evaluate`.

### Verify

```bash
find data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1 -name "run.json" | head
```

## RQ2: Query Perturbation Robustness and Plan Collapse

RQ2 evaluates robustness under perturbations: `mispelling`, `ordering`, `synonym`, `paraphrase`, `naturality` across seeds `1999, 5, 27, 2016, 2026`.

### 1) Ensure perturbation files exist

Expected location:
- `data/msmarco-full/query_variations/msmarco/msmarco_test_attacked_queries_seed_<seed>_attack_method_<method>.json`

Generate if missing:

```bash
conda activate pag-robustness
bash robustness/scripts/generate_query_variations.sh
```

Single-file generation:

```bash
python -m robustness.query_variations.generate_penha \
  --dataset msmarco \
  --split test \
  --seed 1999 \
  --attack_method mispelling
```

### 2) Run RQ2 evaluation

Single run:

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

SLURM batch launcher:

```bash
bash robustness/scripts/run_rq2_pipeline.sh
```

### 3) Aggregate over seeds/splits

```bash
python -m robustness.evaluation.aggregate_results \
  --results_dir experiments/RQ2_robustness \
  --splits dl19 dl20 dev \
  --attacks mispelling ordering synonym paraphrase naturality
```

### RQ2 expected outputs

Per run:
- `experiments/RQ2_robustness/<DATASET>/perturbed/<attack>_seed_<seed>/plan_collapse_per_query.json`
- `experiments/RQ2_robustness/<DATASET>/perturbed/<attack>_seed_<seed>/plan_collapse_sensitivity.json`
- `experiments/RQ2_robustness/<DATASET>/perturbed/<attack>_seed_<seed>/planner_tokens.json`
- `experiments/RQ2_robustness/<DATASET>/perturbed/<attack>_seed_<seed>/per_query_metrics/*.json`

Global summaries:
- `experiments/RQ2_robustness/summary.json`
- `experiments/RQ2_robustness/summary.csv`
- split-tagged versions (for split-specific runs), e.g. `summary_dl19.csv`.

## RQ3: Cross-Lingual Query Shift (mMARCO -> English MS MARCO)

RQ3 evaluates language mismatch with three inference systems and plan-swap probe:
1. Naive PAG on non-English queries.
2. Sequential-only (planner disabled).
3. Translate-at-inference (NLLB/M2M100).
4. Plan-swap (target query with English lexical plan).

### 1) Download mMARCO queries

```bash
conda activate pag-env
bash cross_lingual/scripts/download_mmarco.sh --force
```

Direct command:

```bash
python -m cross_lingual.data.mmarco_loader --languages fr de zh nl --force
```

### 2) Optional: pre-cache translations

```bash
bash cross_lingual/scripts/translate_queries.sh
```

### 3) Run RQ3 evaluation

Single language/split:

```bash
python -m cross_lingual.evaluation.rq3 \
  --language fr \
  --split dev \
  --translation_model m2m100 \
  --n_gpu 1 \
  --batch_size 8 \
  --lex_topk 1000 \
  --smt_topk 100 \
  --output_dir experiments/RQ3_crosslingual
```

All language/split combinations via SLURM launcher:

```bash
bash cross_lingual/scripts/run_rq3_pipeline.sh
```

### 4) Aggregate RQ3 outputs

```bash
python -m cross_lingual.evaluation.aggregate_results \
  --results_dir experiments/RQ3_crosslingual
```

### RQ3 expected outputs

- `experiments/RQ3_crosslingual/queries/.../raw.tsv`
- `experiments/RQ3_crosslingual/english/<split>/pag/{lex_ret,smt_ret}/...`
- `experiments/RQ3_crosslingual/<lang>/<split>/{naive_pag,seq_only,translate_pag,plan_swap}/...`
- `experiments/RQ3_crosslingual/<lang>/<split>/planner_tokens/*.json`
- `experiments/RQ3_crosslingual/summary.json`
- `experiments/RQ3_crosslingual/summary.csv`
- `experiments/RQ3_crosslingual/latex/*.tex` (after aggregation)

## Optional: RQ3 Trained Extension

This is explicitly an extension (not artifact-only reproduction). It fine-tunes query-side planner alignment while keeping document-side artifacts fixed.

Runner:

```bash
bash cross_lingual/trained_extension/run_rq3_trained_extension.sh
```

Outputs:
- `cross_lingual/trained_extension/checkpoints/<lang>/{best,final}/`
- `cross_lingual/trained_extension/results/summary_rq3_trained.csv`

## Full Training Pipeline (Original PAG-Style Stages)

The full end-to-end training scripts are in `full_scripts/` and follow:
1. Stage 1 lexical/set-based docid path (SPLADE + term encoder).
2. Stage 2 sequential/semantic docid path (dense + AQ + RIPOR).
3. Stage 3 merged joint lexical+sequential PAG training.

Canonical script sequence (high-level):
1. `full_scripts/t5_splade_train.sh`
2. `full_scripts/t5_splade_get_bow_rep.sh`
3. `full_scripts/t5_full_term_encoder_train.sh`
4. `full_scripts/t5_full_term_encoder_evaluate.sh`
5. `full_scripts/t5_full_dense_train.sh`
6. `full_scripts/t5_full_dense_evaluate.sh`
7. `full_scripts/full_ripor_initial_train.sh`
8. `full_scripts/full_ripor_direct_lng_knp_train.sh`
9. `python -m t5_pretrainer.full_preprocess.merge_model_weights`
10. `full_scripts/full_lexical_ripor_train.sh`

For script-specific knobs (`task`, `finetune_step`, run names, paths), verify values at the top of each script before execution.

## Observability Checklist

Use this checklist to confirm runs are healthy and complete.

### Before starting
1. `conda activate <env>` succeeds.
2. `python -c "import torch; print(torch.cuda.is_available())"` returns `True`.
3. Required checkpoint/docid/qrel paths exist.

### During run
1. SLURM logs are written under `experiments/.../logs/` or `logs/`.
2. For distributed jobs, no persistent `run_*.json` shard leftovers once merge completes.

### After run
1. `run.json` exists for target datasets.
2. `perf_all_datasets.json` (or split `perf.json`) exists for evaluated systems.
3. Summary files (`summary*.csv/json`) exist for RQ2/RQ3.
4. Aggregation scripts complete without missing-key errors.

## Notes and Common Pitfalls

1. Many shell scripts are SLURM-oriented and assume cluster partitions like `gpu_h100`.
2. `full_scripts/full_lexical_ripor_evaluate.sh` is multi-purpose; ensure the correct `task` block is active.
3. RQ2 expects perturbation JSON filenames with `attack_method_<method>` and seed suffixes exactly matching loader logic.
4. RQ3 translation model can be set to `nllb` or `m2m100`; caching is stored under `data/translation_cache`.
5. For reproducibility of reported means/stds, keep the canonical 5 seeds.

## References

- PAG paper (original): https://arxiv.org/pdf/2404.14600.pdf

## Citation

```bibtex
@inproceedings{pag_repro2026,
  title={Lost in Decoding? Reproducing and Stress-Testing the Lexical Planner in Generative Retrieval},
  author={Anonymous},
  booktitle={SIGIR},
  year={2026}
}

@inproceedings{zeng2024planning,
  title={Planning Ahead in Generative Retrieval: Guiding Autoregressive Generation through Simultaneous Decoding},
  author={Zeng, Hansi and Luo, Chen and Zamani, Hamed},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={469--480},
  year={2024}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

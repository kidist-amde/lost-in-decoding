[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](#)
[![Submission](https://img.shields.io/badge/submission-%23XXX-informational)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](#)

# Lost in Translation? Reproducing and Stress-Testing the Lexical Planner in Generative Retrieval

> **Reproducibility and stress-test study of PAG (Planning Ahead in Generative Retrieval)**
>
> Submitted to SIGIR '26

## Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Download Data and Checkpoints](#download-data-and-checkpoints)
- [Reproducing the Main Results](#reproducing-the-main-results)
  - [RQ1: Reproduction](#rq1-can-we-reproduce-pags-reported-effectiveness-efficiency-and-ablation-findings)
  - [RQ2: Query Noise Sensitivity](#rq2-how-sensitive-is-the-lexical-planner-to-realistic-query-noise)
  - [RQ3: Multilingual Generalization](#rq3-does-lexical-planning-generalize-across-languages)
- [Evaluated Models](#evaluated-models)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

This repository contains the code for our reproducibility and stress-test study of [Planning Ahead in Generative Retrieval (PAG)](https://arxiv.org/pdf/2404.14600.pdf) by Zeng et al. (SIGIR '24). We go beyond replication to mechanistically interrogate PAG's lexical planning module, identifying failure modes under query perturbations, and analyzing generalization under multilingual lexical variation.

PAG introduces a dual-stage decoding framework for generative retrieval: before generating a sequential document ID, the model performs a one-step **simultaneous decoding** to produce a **set-based lexical plan** (bag-of-words tokens) that guides subsequent autoregressive generation. This mitigates the *prefix pruning* problem in constrained beam search.

Our study validates PAG's effectiveness and then stress-tests the planning mechanism:

```
Query ──────────────────────────────────────────────────────────────
(Original / Perturbed)
       │
       ▼
  Tokenization (En / Zh / De ...)
       │
       ▼
  Transformer Encoder
       │
       ├──────────────────────────┐
       ▼                         ▼
  Set-based Planner         Plan Collapse
  (Simultaneous Decoding)     Source 1: Query Noise
       │                       Source 2: Segmentation Shift
       │  Lexical Plan          Metric: Intersection@k
       │  (Top-k Tokens)
       │
       ▼
  Sequential Decoder
  (Guided Beam Search)
       │
       ▼
  Generated DocIDs
```

## Repository Structure

```
PAG/
├── t5_pretrainer/                 # PAG core implementation (reproduction)
│   ├── main.py                    # Training entry point
│   ├── arguments.py               # Configuration and arguments
│   ├── evaluate.py                # Evaluation pipeline
│   ├── modeling/                  # Model architectures
│   │   ├── t5_generative_retriever.py   # RIPOR & LexicalRipor models
│   │   ├── t5_term_encoder.py           # SPLADE & term encoder models
│   │   └── customized_modeling_t5.py    # Custom T5 modifications
│   ├── dataset/                   # Data loading and collation
│   ├── tasks/                     # Training and evaluation logic
│   ├── losses/                    # Loss functions and regularization
│   ├── utils/                     # Utilities (metrics, prefix trie, etc.)
│   └── full_preprocess/           # Data preprocessing scripts
├── robustness/                    # Stress-test extensions (RQ2 & RQ3)
│   ├── query_variations/          # Query-variation generation + loaders
│   │   ├── generate_penha.py      # Main Penha-style variation generator
│   │   └── penha/
│   │       ├── transformations_mispelling.py    # Typo generation (QWERTY, char swap, etc.)
│   │       ├── transformations_paraphrase.py    # Paraphrase (T5, back-translation)
│   │       ├── transformations_synonym.py       # Synonym replacement (WordNet, embeddings)
│   │       ├── transformations_ordering.py      # Word order shuffling
│   │       └── transformations_naturality.py    # Stop word removal, summarization
│   ├── evaluation/               # Robustness evaluation entry points
│   │   ├── rq2.py                 # RQ2 pipeline
│   │   └── attack_eval.py         # Attack evaluation on dense retrieval
│   ├── metrics/                   # Robustness metrics
│   │   └── plan_collapse.py       # Plan-collapse metrics
│   ├── scripts/                   # SLURM evaluation scripts
│   │   ├── eval.sh                # Standard BEIR evaluation
│   │   ├── evaluate_attack_queries.sh   # Evaluation on perturbed queries
│   │   ├── generate_query_variations.sh  # Batch query variation generation
│   │   └── run_rq2_pipeline.sh    # RQ2 robustness pipeline
│   └── utils/                     # Evaluation utilities
│       ├── beir_custom_metrics.py       # MRR, Recall_cap, Hole@k, Accuracy@k
│       ├── beir_custom_evaluation.py    # Full evaluation pipeline with oracle NDCG
│       ├── beir_exact_search.py         # Dense retrieval with exact search
│       ├── load_data.py                 # BEIR dataset loading
│       └── load_model.py               # Model loading (Contriever, BGE-M3, etc.)
├── full_scripts/                  # Shell scripts for PAG training/evaluation
├── data/                          # Datasets and checkpoints/
└── requirements.txt               # Python dependencies
```

## Requirements

- Python 3.10
- PyTorch (CUDA 12.1+)
- 4--8 NVIDIA GPUs (A100/H100 40/80GB recommended)

### Installation

```bash
pip install -r requirements.txt
conda install -c conda-forge faiss-gpu
```

**Core dependencies:** Transformers, Accelerate, FAISS-GPU, Datasets, pytrec-eval, SentencePiece, TextAttack, wandb

## Download Data and Checkpoints

All necessary files and checkpoints for PAG reproduction are available at [PAG-data (Google Drive)](https://drive.google.com/drive/folders/1q8FeHQ6nxPYpl1Thqw8mS-2ndzf7VZ9y?usp=sharing).

For **inference only**, download:

| File | Description |
|------|-------------|
| `experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/` | Trained PAG model checkpoint |
| `experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json` | Sequential (AQ) DocIDs |
| `experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json` | Set-based (BOW) DocIDs |
| `msmarco-full/` | MS MARCO collection, queries, and relevance judgments |

### Data Directory Structure

```
data/
├── experiments-full-lexical-ripor/
│   ├── lexical_ripor_direct_lng_knp_seq2seq_1/
│   │   └── checkpoint/
│   └── t5-full-dense-1-5e-4-12l/
│       └── aq_smtid/
│           └── docid_to_tokenids.json
├── experiments-splade/
│   └── t5-splade-0-12l/
│       └── top_bow/
│           └── docid_to_tokenids.json
└── msmarco-full/
    ├── full_collection/
    ├── dev_queries/
    ├── dev_qrel.json
    ├── TREC_DL_2019/
    │   ├── queries_2019/
    │   ├── qrel.json
    │   └── qrel_binary.json
    └── TREC_DL_2020/
        ├── queries_2020/
        ├── qrel.json
        └── qrel_binary.json
```

# Reproducing the Main Results

## RQ1: Can we reproduce PAG's reported effectiveness, efficiency, and ablation findings?

The full PAG training pipeline consists of three stages. Stages 1 and 2 can be trained in parallel. See [follow_the_steps.md](follow_the_steps.md) for the complete step-by-step guide.

> **Note:** Several scripts require setting a `task=` or `finetune_step=` variable at the top before running.

### Stage 1: Set-based (Lexical) DocIDs

1. **Train SPLADE sparse encoder:**
   ```bash
   bash full_scripts/t5_splade_train.sh       # 8 GPUs, 200K steps → M_sp
   ```

2. **Extract set-based DocIDs (bag-of-words):**
   ```bash
   bash full_scripts/t5_splade_get_bow_rep.sh
   ```
   This produces `experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json`.

3. **Term encoder fine-tuning, step 1** (set `finetune_step=bm25_neg`):
   ```bash
   bash full_scripts/t5_full_term_encoder_train.sh   # 8 GPUs, 50 epochs
   ```

4. **Mine hard negatives for step 2:**
   ```bash
   # Set task=retrieve_train_queries in full_scripts/t5_full_term_encoder_evaluate.sh
   bash full_scripts/t5_full_term_encoder_evaluate.sh
   python t5_pretrainer/full_preprocess/add_qrel_to_rerank_run.py
   ```
   Set `experiment_names = ["t5-term-encoder-0-bow-12l"]` in the `.py` script.

5. **Term encoder fine-tuning, step 2** (set `finetune_step=self_neg`):
   ```bash
   bash full_scripts/t5_full_term_encoder_train.sh   # 8 GPUs, 50 epochs → M_set
   ```

### Stage 2: Sequential (Semantic) DocIDs

1. **Dense encoder fine-tuning, step 1** (set `finetune_step=bm25_neg`):
   ```bash
   bash full_scripts/t5_full_dense_train.sh   # 4 GPUs, 50 epochs
   ```

2. **Mine hard negatives for step 2:**
   ```bash
   # Set task=retrieve_train_queries in full_scripts/t5_full_dense_evaluate.sh
   bash full_scripts/t5_full_dense_evaluate.sh
   bash full_scripts/rerank_for_create_trainset.sh
   python t5_pretrainer/full_preprocess/add_qrel_to_rerank_run.py
   ```
   Set `experiment_names = ["t5-full-dense-0-5e-4-12l"]` in the `.py` script.

3. **Dense encoder fine-tuning, step 2** (set `finetune_step=self_neg`):
   ```bash
   bash full_scripts/t5_full_dense_train.sh   # 4 GPUs, 50 epochs → M_ds
   ```

4. **Build sequential DocIDs via Additive Quantization** (set `task=all_aq_pipline`):
   ```bash
   bash full_scripts/t5_full_dense_evaluate.sh
   ```
   This produces `experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json`.

5. **Seq2seq pre-training (RIPOR):**
   ```bash
   bash full_scripts/full_ripor_initial_train.sh   # 8 GPUs, 250K steps → M_s2s
   ```

6. **Rank-oriented fine-tuning:**
   ```bash
   bash full_scripts/full_ripor_direct_lng_knp_train.sh   # 8 GPUs, 150 epochs → M_seq
   ```

### Stage 3: Unified PAG

1. **Merge lexical and semantic model weights:**
   ```bash
   cd t5_pretrainer
   python -m full_preprocess.merge_model_weights
   cd ..
   ```

2. **Final joint fine-tuning:**
   ```bash
   bash full_scripts/full_lexical_ripor_train.sh   # 4 GPUs, 120 epochs → PAG final
   ```

3. **Run inference and evaluation** (set `task=lexical_constrained_retrieve_and_rerank`):
   ```bash
   bash full_scripts/full_lexical_ripor_evaluate.sh
   ```
   We use a single 80GB A100 GPU for inference. This produces the effectiveness results for **Table 1**, the ablation analysis for **Table 2**, and the efficiency/latency analysis for **Table 3**. Evaluation is performed on MS MARCO Dev, TREC-DL 2019, and TREC-DL 2020 using MRR@10, Recall@k, and NDCG@k.

## RQ2: How sensitive is the lexical planner to realistic query noise?

We stress-test PAG's lexical planner under controlled query perturbations using the Penha et al. framework. This measures *plan collapse*: perturbation-induced degradation of (i) the planner's top-n candidate set and (ii) token-plan consistency, alongside the sequential decoder's recovery rate from corrupted plans.

Five perturbation types are supported:

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Misspelling** | Character-level typos (QWERTY errors, char swap, insertion, deletion) | TextAttack transformations |
| **Synonym** | Word-level replacement via WordNet or embedding similarity | TextAttack + Universal Sentence Encoder constraint |
| **Paraphrase** | Sequence-level rewrites via T5 paraphraser or back-translation (M2M100) | Pretrained seq2seq models |
| **Ordering** | Random word order shuffling | Custom transformation |
| **Naturality** | Stop word removal, random word deletion, T5 summarization | NLTK + T5-base |

1. **Generate query variations:**
   ```bash
   bash robustness/scripts/generate_query_variations.sh
   ```
   Query variations are saved to `output_attack/attacked_text/query/`. To generate a single perturbation type:
   ```bash
   python -m robustness.query_variations.generate_penha \
       --attack_method mispelling \
       --dataset msmarco \
       --split test
   ```

2. **Evaluate on perturbed queries:**
   ```bash
   bash robustness/scripts/evaluate_attack_queries.sh
   ```
   Evaluation runs across multiple random seeds (1999, 5, 27, 2016, 2026) for statistical robustness.

## RQ3: Does lexical planning generalize across languages?

We evaluate whether PAG's lexical planning generalizes across languages using mMARCO. We analyze how planner quality correlates with lexical segmentation characteristics -- e.g., ambiguous segmentation in Chinese, compounding and inflection in German -- that change the effective notion of a lexical unit.

1. **Evaluate on mMARCO across target languages:**
   ```bash
   bash robustness/scripts/eval.sh
   ```

## Evaluated Models

The robustness evaluation framework (RQ2 and RQ3) supports multiple retrieval baselines for comparison:

| Model | Type |
|-------|------|
| BM25 | Sparse |
| Contriever | Dense |
| BGE-M3 | Dense (multilingual) |
| Qwen3 (0.6B, 4B) | Dense |
| GTE | Dense |
| LiNQ | Dense |
| ReasonIR | Dense |
| DiVER | Dense |
| BGE-Reasoner | Dense |

## Citation

If you find this work useful, please cite our paper and the original PAG paper:

```bibtex
@inproceedings{pag_repro2026,
  title={Lost in Translation? Reproducing and Stress-Testing the Lexical Planner in Generative Retrieval},
  author={Anonymous},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
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

## Acknowledgments

This work builds upon [PAG](https://doi.org/10.1145/3626772.3657746) (Zeng et al., SIGIR '24), [RIPOR](https://arxiv.org/pdf/2311.09134.pdf), and the [SPLADE](https://arxiv.org/abs/2107.05720) family of sparse retrieval models. Query perturbation methods follow [Penha et al.](https://doi.org/10.1007/978-3-030-99736-6_37). The evaluation infrastructure is built on [BEIR](https://github.com/beir-cellar/beir).

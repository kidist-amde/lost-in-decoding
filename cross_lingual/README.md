# RQ3: Cross-lingual Evaluation of PAG's Lexical Planner

This module evaluates whether **PAG's lexical planner still helps** when **queries are non-English** but **documents are English**, where lexical overlap is reduced.

## Goal

Test PAG under cross-lingual mismatch to understand:
1. How much does the planner degrade when queries don't share vocabulary with English docids?
2. Does the sequential decoder compensate for planner failures?
3. Can simple interventions (translation, plan swap) restore performance?

## Languages Tested

| Language | Code | Mismatch Level | Notes |
|----------|------|----------------|-------|
| French   | `fr` | Moderate | Latin script, some cognates |
| German   | `de` | Moderate | Latin script, compound words |
| Chinese  | `zh` | Hard | Different script, no lexical overlap |
| Dutch    | `nl` | Moderate | Latin script, inflectional variation |

## Systems Evaluated

### System A: Naive Cross-lingual PAG
- Input: Non-English query
- Planner: Runs as-is (expected to underperform due to lexical mismatch)
- Decoder: Standard constrained beam search
- **Purpose**: Measure baseline cross-lingual degradation

### System B: Sequential-only (Planner Disabled)
- Input: Non-English query
- Planner: Disabled (no lexical constraint)
- Decoder: Unconstrained beam search
- **Purpose**: Isolate whether planner is helping or hurting

### System C: Translate-at-Inference
- Input: Query translated to English via NLLB-200
- Planner: Runs on English translation
- Decoder: Standard constrained beam search
- **Purpose**: Test if restoring overlap restores planner usefulness

### Plan Swap (Causal Probe)
- Input: Non-English query
- Planner: Uses plan from parallel English query
- Decoder: Constrained by English plan
- **Purpose**: Direct evidence that planner mismatch is the bottleneck

## Metrics

### End-to-End Retrieval
- NDCG@10, MRR@10, Recall@100

### Planner Diagnostics
- **Candidate Recall@n**: Is the relevant doc in planner's top-n?
- **SimulOnly Quality**: Planner-only ranking effectiveness
- **Token Overlap**: Jaccard similarity of planner tokens (English vs target)
- **PAG Gain**: Stage2 improvement over Stage1
- **Plan Swap Gain**: Improvement when using English plan

## Directory Structure

```
cross_lingual/
├── data/
│   └── mmarco_loader.py      # Load mMARCO multilingual queries
├── evaluation/
│   ├── rq3.py                # Main evaluation pipeline
│   └── aggregate_results.py  # Results aggregation & LaTeX tables
├── metrics/
│   └── planner_diagnostics.py # Planner-level analysis metrics
├── scripts/
│   ├── download_mmarco.sh    # Download mMARCO from HuggingFace
│   ├── translate_queries.sh  # Pre-cache translations
│   ├── run_rq3_pipeline.sh   # SLURM job launcher
│   └── run_rq3_pipeline_sub.sh # Single-job SLURM script
└── utils/
    ├── pag_inference.py      # PAG pipeline wrappers
    └── translator.py         # NLLB/M2M100 translation with caching
```

## Quick Start

### 1. Download mMARCO Queries

```bash
# Via SLURM
bash cross_lingual/scripts/download_mmarco.sh --force


# Or directly
python -m cross_lingual.data.mmarco_loader --languages fr de zh nl
```

### 2. (Optional) Pre-cache Translations

```bash
# Caches all translations to avoid loading model during main run
sbatch cross_lingual/scripts/translate_queries.sh
```

### 3. Run Evaluation

```bash
# All languages, all splits (submits 12 SLURM jobs)
./cross_lingual/scripts/run_rq3_pipeline.sh

# Single language/split
./cross_lingual/scripts/run_rq3_pipeline.sh fr dev

# Or run directly
python -m cross_lingual.evaluation.rq3 \
    --language fr \
    --split dev \
    --n_gpu 1 \
    --batch_size 8
```

### 4. Aggregate Results

```bash
python -m cross_lingual.evaluation.aggregate_results \
    --results_dir experiments/RQ3_crosslingual
```

## Output Structure

```
experiments/RQ3_crosslingual/
├── queries/                    # Prepared query files
│   └── TREC_DL_2019/
│       ├── english/raw.tsv
│       ├── fr/raw.tsv
│       ├── fr_translated_nllb/raw.tsv
│       ├── nl/raw.tsv
│       └── nl_translated_nllb/raw.tsv
├── english/                    # English baseline results
│   └── dl19/pag/{lex_ret,smt_ret}/
├── fr/                         # French results
│   └── dl19/
│       ├── naive_pag/          # System A
│       ├── seq_only/           # System B
│       ├── translate_pag/      # System C
│       ├── plan_swap/          # Causal probe
│       └── planner_tokens/     # Token analysis
├── nl/                         # Dutch results
│   └── dl19/
│       ├── naive_pag/          # System A
│       ├── seq_only/           # System B
│       ├── translate_pag/      # System C
│       ├── plan_swap/          # Causal probe
│       └── planner_tokens/     # Token analysis
├── summary.json                # All results
├── summary.csv                 # Flattened for analysis
├── aggregates.json             # Statistics
└── latex/                      # Paper tables
    ├── table_main_results.tex
    ├── table_planner_diagnostics.tex
    └── table_summary_by_language.tex
```

## Expected Findings

Based on the RQ3 hypothesis:

1. **Naive PAG degrades**: Cross-lingual queries cause planner recall to drop
2. **Chinese worst**: Hard mismatch (different script) shows largest degradation
3. **Translation helps**: Restoring English overlap improves planner effectiveness
4. **Plan swap confirms**: Using English plan with target query improves results, proving planner mismatch is the bottleneck

## Data Sources

- **Corpus**: MS MARCO passages (English, same as RQ1/RQ2)
- **DocIDs**: Released English PAG artifacts (sequential + set-based)
- **Queries**: [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) multilingual queries
- **Qrels**: Original MS MARCO relevance judgments

## Dependencies

Additional packages beyond base PAG requirements:
```
datasets          # For loading mMARCO from HuggingFace
transformers      # For NLLB/M2M100 translation models
sentencepiece     # For NLLB tokenization
```

## Citation

If using this evaluation framework:

```bibtex
@inproceedings{pag-stress-test,
  title={Lost in Translation? Reproducing and Stress-Testing the Lexical Planner in Generative Retrieval},
  author={...},
  booktitle={SIGIR},
  year={2026}
}
```

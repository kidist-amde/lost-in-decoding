# RQ3 Trained Extension: Query-Side Planner Alignment

**Status**: EXTENSION (not reproduction). Evaluation protocol matches inference-only baselines.

## Goal

Train a student model so that a non-English query `q_lang` produces planning-token
weights aligned to the teacher model's weights for the paired English query `q_en`
(same qid), **without any re-indexing**.

## Frozen vs. Trained Components

### Frozen (MUST remain unchanged)
- Released sequential DocIDs, trie (L=8, V=2048)
- `docid_to_tokenids.json` (top-64 tokens per docid) for both lexical and semantic
- All index/identifier artifacts
- Doc-side cached planner artifacts
- **Teacher model** = released English PAG checkpoint

### Trained (student)
- T5 encoder (query-side) — optionally only last N layers via `finetune_last_n_layers`
- `lm_head` linear projection (shared vocabulary projection)

### Always frozen in student
- T5 decoder (all layers)
- Embedding layer (when `finetune_last_n_layers < total_layers`)

## Training Objective

**KL distillation over planner-token distributions:**

For each parallel query pair `(q_en, q_lang)` sharing the same qid:

1. Extract planner tokens using the **exact** RQ2 `extract_planner_tokens` code path:
   - T5 encoder → T5 decoder (mode=`lex_retrieval`) → `lm_head` → `log(1 + relu(logits)) * mask` → max-pool → `[bz, vocab_size]`
   - Top-K=100 tokens per query

2. Compute union `U = T_ids ∪ S_ids` (teacher top-K ∪ student top-K)

3. KL divergence with temperature scaling:
   ```
   p = softmax(z_teacher[U] / τ)
   q = softmax(z_student[U] / τ)
   L_align = KL(p || q) * τ²
   ```

4. Fine-tune only query-side encoder + lm_head parameters.

## Data

- **Languages**: zh, nl, fr, de
- **Parallel pairs**: mMARCO queries (HuggingFace `unicamp-dl/mmarco`)
  - `q_lang`: mMARCO `queries-{lang}` split
  - `q_en`: mMARCO `queries-en` split (preferred); fallback to MS MARCO dev queries
- **Training split**: mMARCO qids NOT in dev evaluation set
- **Evaluation split**: dev qids matching inference-only baselines (intersection with qrels)
- **Ground truth**: English MS MARCO `dev_qrel.json`

### QID Alignment Validation
- Asserts `|Q_en ∩ Q_lang| / |Q_lang| ≈ 1.0`
- Fails loudly if mismatch > 0.1% with example qids

## Evaluation

After training, the adapted checkpoint is evaluated **identically** to inference-only baselines:

| System | Description |
|--------|-------------|
| naive | English PAG on non-English queries (original checkpoint) |
| sequential-only | Unconstrained beam search (original checkpoint) |
| translate | Translate query to English then run PAG (original checkpoint) |
| **adapted** | Adapted checkpoint on non-English queries (NO translation) |

### Metrics (per language)
1. **Retrieval**: MRR@10, nDCG@10, Recall@10
2. **Planner alignment**: TokJaccard@100 (adapted vs teacher-English), CandOverlap@100
3. **Latency**: avg_latency_ms_per_query (same measurement as baselines)
4. **Significance**: paired bootstrap test vs naive baseline

Even if retrieval does not improve, all diagnostics and tables are reported.

## Directory Structure

```
cross_lingual/
├── scripts/
│   ├── run_rq3_trained.sh          # SLURM launcher (submits one job per language)
│   └── run_rq3_trained_sub.sh      # Per-language SLURM job (data prep → train → eval)
└── trained_extension/
    ├── __init__.py
    ├── data_loader_parallel.py      # Parallel (q_en, q_lang) pairs with alignment validation
    ├── extract_planner_tokens.py    # Reuses exact RQ2 extraction code path
    ├── train_planner_alignment.py   # KL distillation training loop
    ├── evaluate_adapted.py          # Evaluation matching baseline protocol
    ├── run_rq3_trained_extension.sh # Sequential master runner (alternative)
    ├── configs/
    │   └── default.yaml             # Default training config
    ├── checkpoints/                 # Saved model checkpoints (per language)
    │   └── {lang}/best/             # Best checkpoint by TokJaccard@100
    ├── results/                     # Training logs, metrics, summary CSV
    │   ├── summary_rq3_trained.csv  # Combined results table
    │   └── {lang}/training_log.json
    └── README.md
```

## How to Run

### SLURM — parallel per-language jobs (recommended)

Submits one H100 job per language (4 jobs total). Each job runs all 3 steps
(data prep → train → evaluate) independently. Results land in
`experiments/RQ3_crosslingual_trained/<lang>/adapted/`.

```bash
# All languages (fr, de, zh, nl)
./cross_lingual/scripts/run_rq3_trained.sh all

# Single language
./cross_lingual/scripts/run_rq3_trained.sh fr

# Two languages
./cross_lingual/scripts/run_rq3_trained.sh "fr de"
```

Monitor with `squeue -u $USER`. Logs at `experiments/RQ3_crosslingual_trained/logs/trained_<lang>_<jobid>.{log,err}`.

### SLURM — sequential single job (alternative)

Runs all languages sequentially in one job. Includes a Step 4 (aggregate + significance).

```bash
sbatch --gpus=1 --time=24:00:00 --mem=180G \
    cross_lingual/trained_extension/run_rq3_trained_extension.sh

# With overrides
sbatch --gpus=1 --time=24:00:00 --mem=180G \
    cross_lingual/trained_extension/run_rq3_trained_extension.sh \
    --languages "fr de" --epochs 5 --lr 1e-5 --batch_size 8 --temperature 2.0
```

### Quick start (single language, no SLURM)
```bash
python -m cross_lingual.trained_extension.train_planner_alignment \
    --language fr --epochs 3 --lr 1e-5 --temperature 2.0

python -m cross_lingual.trained_extension.evaluate_adapted \
    --language fr --significance
```

### Compute-bounded mode
For limited compute, use fewer layers and smaller batch:
```bash
python -m cross_lingual.trained_extension.train_planner_alignment \
    --language fr --finetune_last_n_layers 2 --epochs 3 --batch_size 4
```

## Dependencies

Same as the main repo, plus:
- `pyyaml` (for config loading)
- `datasets` (for mMARCO download from HuggingFace)

## Attribution

This extension references architectural patterns from:
- **MGR-CSC** (Multilingual Generative Retrieval via Cross-lingual Semantic Compression) —
  reference repository at `MGR-CSC/` in this project. No runtime dependency; ideas only.
  No explicit license found in the MGR-CSC repository.
- **PAG** (Planning Ahead in Generative Retrieval) — the base model and all frozen artifacts.
  Planner token extraction reuses the exact RQ2 code path from
  `robustness/utils/pag_inference.py:extract_planner_tokens`.

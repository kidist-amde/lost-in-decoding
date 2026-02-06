#!/usr/bin/env python3
"""
RQ3 Cross-lingual Evaluation for PAG (Planning Ahead in Generative Retrieval).

This script evaluates PAG's cross-lingual robustness by:
1. Loading mMARCO queries in target language (fr, de, zh, nl)
2. Running three systems:
   - System A: Naive cross-lingual PAG (non-English query, English docids)
   - System B: Sequential-only (planner disabled)
   - System C: Translate-at-inference (query translated to English)
3. Computing retrieval metrics (NDCG@10, MRR@10, Recall@100)
4. Computing planner diagnostics (candidate recall, SimulOnly, token overlap)
5. Running plan swap causal probe
6. Saving results to experiments/RQ3_crosslingual/

Usage
-----
    # Single language, single split
    python -m cross_lingual.evaluation.rq3 \\
        --language fr \\
        --split dl19

    # All languages, all splits
    python -m cross_lingual.evaluation.rq3 \\
        --language all \\
        --split all

    # Skip inference (evaluate only from existing runs)
    python -m cross_lingual.evaluation.rq3 \\
        --language fr --split dl19 \\
        --eval_only
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data.mmarco_loader import (
    RQ3_LANGUAGES,
    SPLIT_QREL_PATHS,
    load_qrels,
    load_english_queries,
    load_mmarco_queries,
    prepare_mmarco_queries_for_pag,
    get_parallel_queries,
)
from cross_lingual.utils.translator import (
    translate_mmarco_queries_to_english,
    prepare_translated_queries_for_pag,
)
from cross_lingual.utils.pag_inference import (
    run_naive_crosslingual_pag,
    run_sequential_only,
    run_translate_at_inference_pag,
    run_plan_swap_crosslingual,
    extract_crosslingual_planner_tokens,
    load_lexical_run,
    load_sequential_run,
    load_planner_tokens,
    _get_dataset_name,
)
from cross_lingual.metrics.planner_diagnostics import (
    aggregate_planner_diagnostics,
    compare_planner_recall,
    compare_simul_only,
    crosslingual_token_overlap,
    pag_gain,
    plan_swap_impact,
)
from robustness.metrics.plan_collapse import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_per_query,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGUAGES = RQ3_LANGUAGES  # ["fr", "de", "zh", "nl"]
SPLITS = ["dl19", "dl20", "dev"]

SPLIT_TO_DATASET = {
    "dl19": "TREC_DL_2019",
    "dl20": "TREC_DL_2020",
    "dev": "MSMARCO",
}

TRANSLATION_MODEL = "nllb"  # or "m2m100"


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_single(
    language: str,
    split: str,
    output_dir: str,
    translation_model: str = TRANSLATION_MODEL,
    n_gpu: int = 1,
    batch_size: int = 8,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    eval_only: bool = False,
    download_mmarco: bool = True,
) -> Dict:
    """
    Run the full RQ3 evaluation for one (language, split) pair.

    Returns a dict with results from all systems and diagnostics.
    """
    dataset_name = SPLIT_TO_DATASET[split]
    qrel_path = str(SPLIT_QREL_PATHS[split])

    print(f"\n{'='*70}")
    print(f"[RQ3] Language={language}, Split={split}")
    print(f"{'='*70}")

    result = {
        "language": language,
        "split": split,
        "dataset": dataset_name,
        "translation_model": translation_model,
    }

    # ── 1. Prepare query files ────────────────────────────────────────────
    queries_base = Path(output_dir) / "queries"

    # Prepare English and target language queries
    print(f"\n[RQ3] Preparing query files...")
    english_dir, target_dir, n_queries = prepare_mmarco_queries_for_pag(
        language=language,
        split=split,
        output_base=queries_base,
        download=download_mmarco,
    )
    result["n_queries"] = n_queries

    # Prepare translated queries (for System C)
    print(f"\n[RQ3] Preparing translated queries ({language} -> en)...")
    translated_dir, _ = prepare_translated_queries_for_pag(
        language=language,
        split=split,
        output_base=queries_base,
        model_type=translation_model,
        batch_size=batch_size,
    )

    # Output directories
    lang_output = Path(output_dir) / language / split

    # Load qrels
    with open(qrel_path) as f:
        qrels = json.load(f)

    if not eval_only:
        # ── 2. Run English baseline ───────────────────────────────────────
        print(f"\n[RQ3] === English baseline ({split}) ===")
        english_out = Path(output_dir) / "english" / split
        english_lex_dir = english_out / "pag" / "lex_ret"
        english_smt_dir = english_out / "pag" / "smt_ret"

        english_run_path = english_smt_dir / dataset_name / "run.json"
        if english_run_path.exists():
            print(f"[RQ3] English baseline exists, skipping inference")
        else:
            from robustness.utils.pag_inference import (
                run_lexical_retrieval,
                run_sequential_decoding,
                merge_and_evaluate,
            )

            english_lex_dir.mkdir(parents=True, exist_ok=True)
            english_smt_dir.mkdir(parents=True, exist_ok=True)

            rc = run_lexical_retrieval(
                query_dir=str(english_dir),
                lex_out_dir=str(english_lex_dir),
                topk=lex_topk,
                batch_size=batch_size,
            )
            if rc == 0:
                rc = run_sequential_decoding(
                    query_dir=str(english_dir),
                    smt_out_dir=str(english_smt_dir),
                    lex_out_dir=str(english_lex_dir),
                    topk=smt_topk,
                    batch_size=batch_size * 2,
                    n_gpu=n_gpu,
                )
            if rc == 0:
                merge_and_evaluate(
                    query_dir=str(english_dir),
                    smt_out_dir=str(english_smt_dir),
                    eval_qrel_path=qrel_path,
                )

        # ── 3. System A: Naive cross-lingual PAG ──────────────────────────
        print(f"\n[RQ3] === System A: Naive cross-lingual PAG ({language}/{split}) ===")
        naive_out = lang_output / "naive_pag"
        naive_lex_dir = lang_output / "naive_pag" / "lex_ret"
        naive_smt_dir = lang_output / "naive_pag" / "smt_ret"

        naive_run_path = naive_smt_dir / dataset_name / "run.json"
        if naive_run_path.exists():
            print(f"[RQ3] System A exists, skipping inference")
        else:
            result["naive_pag_result"] = run_naive_crosslingual_pag(
                query_dir=str(target_dir),
                output_dir=str(lang_output),
                eval_qrel_path=qrel_path,
                lex_topk=lex_topk,
                smt_topk=smt_topk,
                batch_size=batch_size,
                n_gpu=n_gpu,
            )

        # ── 4. System B: Sequential-only ──────────────────────────────────
        print(f"\n[RQ3] === System B: Sequential-only ({language}/{split}) ===")
        seq_only_dir = lang_output / "seq_only" / "smt_ret"

        seq_only_run_path = seq_only_dir / dataset_name / "run.json"
        if seq_only_run_path.exists():
            print(f"[RQ3] System B exists, skipping inference")
        else:
            result["seq_only_result"] = run_sequential_only(
                query_dir=str(target_dir),
                output_dir=str(lang_output),
                eval_qrel_path=qrel_path,
                topk=smt_topk,
                batch_size=batch_size * 2,
                n_gpu=n_gpu,
            )

        # ── 5. System C: Translate-at-inference ───────────────────────────
        print(f"\n[RQ3] === System C: Translate-at-inference ({language}/{split}) ===")
        translate_out = lang_output / "translate_pag"
        translate_smt_dir = lang_output / "translate_pag" / "smt_ret"

        translate_run_path = translate_smt_dir / dataset_name / "run.json"
        if translate_run_path.exists():
            print(f"[RQ3] System C exists, skipping inference")
        else:
            result["translate_pag_result"] = run_translate_at_inference_pag(
                original_query_dir=str(target_dir),
                translated_query_dir=str(translated_dir),
                output_dir=str(lang_output),
                eval_qrel_path=qrel_path,
                lex_topk=lex_topk,
                smt_topk=smt_topk,
                batch_size=batch_size,
                n_gpu=n_gpu,
            )

        # ── 6. Plan swap: target query + English plan ─────────────────────
        print(f"\n[RQ3] === Plan swap: {language} query + English plan ({split}) ===")
        swap_smt_dir = lang_output / "plan_swap" / "smt_ret"

        swap_run_path = swap_smt_dir / dataset_name / "run.json"
        if swap_run_path.exists():
            print(f"[RQ3] Plan swap exists, skipping inference")
        else:
            result["plan_swap_result"] = run_plan_swap_crosslingual(
                target_query_dir=str(target_dir),
                source_lex_dir=str(english_lex_dir),
                output_dir=str(lang_output),
                eval_qrel_path=qrel_path,
                topk=smt_topk,
                batch_size=batch_size * 2,
                n_gpu=n_gpu,
            )

        # ── 7. Extract planner tokens ─────────────────────────────────────
        print(f"\n[RQ3] === Extracting planner tokens ===")
        tokens_dir = lang_output / "planner_tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)

        english_tokens, target_tokens = extract_crosslingual_planner_tokens(
            english_query_dir=str(english_dir),
            target_query_dir=str(target_dir),
            output_dir=str(tokens_dir),
            topk=100,
            batch_size=batch_size,
        )

    # ── 8. Compute all metrics from run files ─────────────────────────────
    print(f"\n[RQ3] === Computing metrics ===")

    try:
        # Define paths
        english_out = Path(output_dir) / "english" / split
        english_lex_dir = english_out / "pag" / "lex_ret"
        english_smt_dir = english_out / "pag" / "smt_ret"

        naive_lex_dir = lang_output / "naive_pag" / "lex_ret"
        naive_smt_dir = lang_output / "naive_pag" / "smt_ret"
        seq_only_dir = lang_output / "seq_only" / "smt_ret"
        translate_lex_dir = lang_output / "translate_pag" / "lex_ret"
        translate_smt_dir = lang_output / "translate_pag" / "smt_ret"
        swap_smt_dir = lang_output / "plan_swap" / "smt_ret"

        # Load runs
        english_lex_run = load_lexical_run(str(english_lex_dir), dataset_name)
        english_smt_run = load_sequential_run(str(english_smt_dir), dataset_name)
        naive_lex_run = load_lexical_run(str(naive_lex_dir), dataset_name)
        naive_smt_run = load_sequential_run(str(naive_smt_dir), dataset_name)
        seq_only_run = load_sequential_run(str(seq_only_dir), dataset_name)
        translate_lex_run = load_lexical_run(str(translate_lex_dir), dataset_name)
        translate_smt_run = load_sequential_run(str(translate_smt_dir), dataset_name)
        swap_smt_run = load_sequential_run(str(swap_smt_dir), dataset_name)

        # Load planner tokens
        tokens_dir = lang_output / "planner_tokens"
        english_tokens = load_planner_tokens(str(tokens_dir / "english_planner_tokens.json"))
        target_tokens = load_planner_tokens(str(tokens_dir / "target_planner_tokens.json"))

        # --- English baseline metrics ---
        if english_lex_run:
            result["english_lex_metrics"] = compute_retrieval_metrics(
                english_lex_run, qrels, k=10
            )
        if english_smt_run:
            result["english_smt_metrics"] = compute_retrieval_metrics(
                english_smt_run, qrels, k=10
            )

        # --- System A: Naive PAG metrics ---
        if naive_lex_run:
            result["naive_lex_metrics"] = compute_retrieval_metrics(
                naive_lex_run, qrels, k=10
            )
        if naive_smt_run:
            result["naive_smt_metrics"] = compute_retrieval_metrics(
                naive_smt_run, qrels, k=10
            )

        # --- System B: Sequential-only metrics ---
        if seq_only_run:
            result["seq_only_metrics"] = compute_retrieval_metrics(
                seq_only_run, qrels, k=10
            )

        # --- System C: Translate-at-inference metrics ---
        if translate_lex_run:
            result["translate_lex_metrics"] = compute_retrieval_metrics(
                translate_lex_run, qrels, k=10
            )
        if translate_smt_run:
            result["translate_smt_metrics"] = compute_retrieval_metrics(
                translate_smt_run, qrels, k=10
            )

        # --- Plan swap metrics ---
        if swap_smt_run:
            result["swap_smt_metrics"] = compute_retrieval_metrics(
                swap_smt_run, qrels, k=10
            )

        # --- Planner diagnostics ---
        if english_lex_run and naive_lex_run:
            # Candidate recall comparison
            result["candidate_recall"] = compare_planner_recall(
                english_lex_run, naive_lex_run, qrels
            )

            # SimulOnly comparison
            result["simul_only_comparison"] = compare_simul_only(
                english_lex_run, naive_lex_run, qrels, k=10
            )

        # Token overlap
        if english_tokens and target_tokens:
            result["token_overlap"] = crosslingual_token_overlap(
                english_tokens, target_tokens
            )

        # PAG gain comparison
        if all(k in result for k in [
            "english_lex_metrics", "english_smt_metrics",
            "naive_lex_metrics", "naive_smt_metrics"
        ]):
            english_gain = pag_gain(
                result["english_lex_metrics"],
                result["english_smt_metrics"]
            )
            naive_gain = pag_gain(
                result["naive_lex_metrics"],
                result["naive_smt_metrics"]
            )
            result["english_pag_gain"] = english_gain
            result["naive_pag_gain"] = naive_gain
            result["pag_gain_delta"] = {
                k: naive_gain[k] - english_gain[k]
                for k in english_gain
            }

        # Plan swap impact
        if "naive_smt_metrics" in result and "swap_smt_metrics" in result:
            result["plan_swap_impact"] = plan_swap_impact(
                result["naive_smt_metrics"],
                result["swap_smt_metrics"]
            )

        # Translate vs Naive comparison
        if "translate_smt_metrics" in result and "naive_smt_metrics" in result:
            result["translate_vs_naive"] = {
                f"delta_{m}": (
                    result["translate_smt_metrics"].get(m, 0) -
                    result["naive_smt_metrics"].get(m, 0)
                )
                for m in ["NDCG@10", "MRR@10"]
            }

    except Exception as e:
        print(f"[RQ3] Warning: metric computation failed: {e}")
        import traceback
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(results: List[Dict], output_dir: str, tag: str = ""):
    """Write summary.json and summary.csv."""
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{tag}" if tag else ""

    # JSON
    summary_json = os.path.join(output_dir, f"summary{suffix}.json")
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[RQ3] Wrote {summary_json}")

    # CSV: flatten key metrics
    csv_rows = []
    for r in results:
        row = {
            "language": r.get("language"),
            "split": r.get("split"),
            "n_queries": r.get("n_queries"),
        }

        # English baseline
        for prefix, key in [
            ("en_lex", "english_lex_metrics"),
            ("en_smt", "english_smt_metrics"),
        ]:
            metrics = r.get(key, {})
            for m in ["NDCG@10", "MRR@10"]:
                row[f"{prefix}_{m}"] = metrics.get(m)

        # System A: Naive PAG
        for prefix, key in [
            ("naive_lex", "naive_lex_metrics"),
            ("naive_smt", "naive_smt_metrics"),
        ]:
            metrics = r.get(key, {})
            for m in ["NDCG@10", "MRR@10"]:
                row[f"{prefix}_{m}"] = metrics.get(m)

        # System B: Sequential-only
        metrics = r.get("seq_only_metrics", {})
        for m in ["NDCG@10", "MRR@10"]:
            row[f"seq_only_{m}"] = metrics.get(m)

        # System C: Translate-at-inference
        for prefix, key in [
            ("trans_lex", "translate_lex_metrics"),
            ("trans_smt", "translate_smt_metrics"),
        ]:
            metrics = r.get(key, {})
            for m in ["NDCG@10", "MRR@10"]:
                row[f"{prefix}_{m}"] = metrics.get(m)

        # Plan swap
        metrics = r.get("swap_smt_metrics", {})
        for m in ["NDCG@10", "MRR@10"]:
            row[f"swap_smt_{m}"] = metrics.get(m)

        # Deltas
        for m in ["NDCG@10", "MRR@10"]:
            # SimulOnly drop (English -> Naive)
            en_lex = row.get(f"en_lex_{m}")
            naive_lex = row.get(f"naive_lex_{m}")
            if en_lex is not None and naive_lex is not None:
                row[f"delta_simul_{m}"] = naive_lex - en_lex

            # PAG drop (English -> Naive)
            en_smt = row.get(f"en_smt_{m}")
            naive_smt = row.get(f"naive_smt_{m}")
            if en_smt is not None and naive_smt is not None:
                row[f"delta_pag_{m}"] = naive_smt - en_smt

            # Translation improvement
            trans_smt = row.get(f"trans_smt_{m}")
            if naive_smt is not None and trans_smt is not None:
                row[f"trans_gain_{m}"] = trans_smt - naive_smt

            # Plan swap improvement
            swap_smt = row.get(f"swap_smt_{m}")
            if naive_smt is not None and swap_smt is not None:
                row[f"swap_gain_{m}"] = swap_smt - naive_smt

        # PAG gain
        english_gain = r.get("english_pag_gain", {})
        naive_gain = r.get("naive_pag_gain", {})
        for m in ["PAG_Gain_NDCG@10", "PAG_Gain_MRR@10"]:
            row[f"en_{m}"] = english_gain.get(m)
            row[f"naive_{m}"] = naive_gain.get(m)

        # Planner diagnostics
        cr = r.get("candidate_recall", {})
        if "aggregate_delta" in cr:
            for m in ["recall@10", "recall@100", "recall@1000"]:
                row[f"delta_{m}"] = cr["aggregate_delta"].get(f"delta_{m}")

        so = r.get("simul_only_comparison", {})
        if "aggregate_delta" in so:
            row["delta_SimulOnly_NDCG@10"] = so["aggregate_delta"].get("delta_SimulOnly_NDCG@10")
            row["delta_SimulOnly_MRR@10"] = so["aggregate_delta"].get("delta_SimulOnly_MRR@10")

        to = r.get("token_overlap", {})
        if "aggregate" in to:
            row["token_jaccard_mean"] = to["aggregate"].get("mean_jaccard")
            row["token_jaccard_median"] = to["aggregate"].get("median_jaccard")

        ps = r.get("plan_swap_impact", {})
        row["PlanSwap_Gain_NDCG@10"] = ps.get("PlanSwap_Gain_NDCG@10")
        row["PlanSwap_Gain_MRR@10"] = ps.get("PlanSwap_Gain_MRR@10")

        csv_rows.append(row)

    if csv_rows:
        summary_csv = os.path.join(output_dir, f"summary{suffix}.csv")
        fieldnames = list(csv_rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[RQ3] Wrote {summary_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Cross-lingual Evaluation for PAG"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help=f"Language to evaluate: {RQ3_LANGUAGES}, or 'all'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dl19",
        help="Evaluation split: dl19, dl20, dev, or 'all'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "RQ3_crosslingual"),
        help="Root output directory",
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="nllb",
        choices=["nllb", "m2m100"],
        help="Translation model for System C",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs for sequential decoding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--lex_topk",
        type=int,
        default=1000,
        help="Top-K for lexical planner (Stage 1)",
    )
    parser.add_argument(
        "--smt_topk",
        type=int,
        default=100,
        help="Top-K for sequential decoder (Stage 2)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip inference; only compute metrics from existing run files",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Do not download mMARCO if not present (fail instead)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve 'all' values
    languages = LANGUAGES if args.language == "all" else [args.language]
    splits = SPLITS if args.split == "all" else [args.split]

    all_results = []
    total = len(languages) * len(splits)
    i = 0

    for lang in languages:
        for split in splits:
            i += 1
            print(f"\n{'='*70}")
            print(f"[RQ3] Experiment {i}/{total}: language={lang}, split={split}")
            print(f"{'='*70}")

            result = evaluate_single(
                language=lang,
                split=split,
                output_dir=args.output_dir,
                translation_model=args.translation_model,
                n_gpu=args.n_gpu,
                batch_size=args.batch_size,
                lex_topk=args.lex_topk,
                smt_topk=args.smt_topk,
                eval_only=args.eval_only,
                download_mmarco=not args.no_download,
            )
            all_results.append(result)

            # Write intermediate summary
            write_summary(all_results, args.output_dir, f"{args.language}_{args.split}")

    # Final summary
    write_summary(all_results, args.output_dir)
    print(f"\n[RQ3] All experiments complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()

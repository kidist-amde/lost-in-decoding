#!/usr/bin/env python3
"""
Baseline A: Naive cross-lingual PAG.

Runs the standard English PAG pipeline directly on non-English mMARCO
queries without any translation or adaptation.  The planner operates
on non-English input but the trie / docids are English.

Produces:
- Stage 1 run file (planner-only / SimulOnly) in TREC format
- Stage 2 run file (full PAG) in TREC format
- Metrics JSON (MRR@10, Recall@10, nDCG@10, latency, error_rate)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data_loader import (
    RQ3_LANGUAGES,
    prepare_query_files,
    save_coverage_report,
    save_filtered_qrels,
    get_evaluation_qids,
)
from cross_lingual.retrieval_engine import (
    run_stage1,
    run_stage2,
    run_full_pag,
    count_queries,
    load_lexical_run,
    load_sequential_run,
    run_json_to_trec,
    _get_dataset_name,
)
from cross_lingual.evaluate import compute_metrics, save_metrics


def run_naive_baseline(
    language: str,
    output_dir: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
    skip_existing: bool = True,
) -> dict:
    """Run Baseline A (naive) for one language.

    Returns dict with metrics and timing.
    """
    out = Path(output_dir) / language / "naive"
    out.mkdir(parents=True, exist_ok=True)

    # Prepare data
    queries_dir = Path(output_dir) / "queries"
    en_dir, tgt_dir, n_queries, coverage = prepare_query_files(
        language, queries_dir, download=True,
    )
    save_coverage_report(coverage, out)

    eval_qids = set(coverage["intersection_qids"])
    qrel_path = out / "qrels.json"
    save_filtered_qrels(eval_qids, qrel_path)

    dataset_name = _get_dataset_name(str(tgt_dir))
    result = {
        "baseline": "naive",
        "language": language,
        "n_queries": n_queries,
    }

    lex_out_dir = str(out / "lex_ret")
    smt_out_dir = str(out / "smt_ret")

    # --- Stage 1 (planner / SimulOnly) ---
    lex_run_path = os.path.join(lex_out_dir, dataset_name, "run.json")
    if os.path.exists(lex_run_path) and skip_existing:
        print(f"[naive] Stage 1 exists, skipping")
        result["stage1_time_s"] = 0
    else:
        rc1, t1 = run_stage1(str(tgt_dir), lex_out_dir, lex_topk, batch_size)
        result["stage1_time_s"] = t1
        if rc1 != 0:
            result["error"] = f"stage1_failed_rc={rc1}"
            return result

    # --- Stage 2 (full PAG) ---
    smt_run_path = os.path.join(smt_out_dir, dataset_name, "run.json")
    if os.path.exists(smt_run_path) and skip_existing:
        print(f"[naive] Stage 2 exists, skipping")
        result["stage2_time_s"] = 0
    else:
        rc2, t2 = run_stage2(
            str(tgt_dir), smt_out_dir, lex_out_dir,
            smt_topk, batch_size * 2, n_gpu,
        )
        result["stage2_time_s"] = t2
        if rc2 != 0:
            result["error"] = f"stage2_failed_rc={rc2}"
            return result

    # --- Merge & evaluate ---
    from robustness.utils.pag_inference import merge_and_evaluate

    merge_and_evaluate(str(tgt_dir), smt_out_dir, str(qrel_path))

    # Load runs and save TREC files
    lex_run = load_lexical_run(lex_out_dir, dataset_name)
    smt_run = load_sequential_run(smt_out_dir, dataset_name)

    if lex_run:
        trec_s1 = str(out / "stage1_run.trec")
        run_json_to_trec(lex_run, trec_s1, f"naive_{language}_s1")

    if smt_run:
        trec_s2 = str(out / "stage2_run.trec")
        run_json_to_trec(smt_run, trec_s2, f"naive_{language}_s2")

    # Compute metrics
    qrels = json.load(open(qrel_path))
    total_time = result.get("stage1_time_s", 0) + result.get("stage2_time_s", 0)

    if lex_run:
        s1_metrics = compute_metrics(lex_run, qrels, n_queries, total_time)
        result["stage1_metrics"] = s1_metrics
        save_metrics(s1_metrics, out / "stage1_metrics.json")

    if smt_run:
        s2_metrics = compute_metrics(smt_run, qrels, n_queries, total_time)
        result["stage2_metrics"] = s2_metrics
        save_metrics(s2_metrics, out / "stage2_metrics.json")

    # Save full result
    with open(out / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 Baseline A: Naive PAG")
    parser.add_argument("--language", type=str, required=True,
                        help=f"Language: {RQ3_LANGUAGES} or 'all'")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "cross_lingual" / "results"))
    parser.add_argument("--lex_topk", type=int, default=1000)
    parser.add_argument("--smt_topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = RQ3_LANGUAGES if args.language == "all" else [args.language]

    for lang in languages:
        print(f"\n{'='*60}")
        print(f"[naive] Running Baseline A for {lang}")
        print(f"{'='*60}")
        result = run_naive_baseline(
            language=lang,
            output_dir=args.output_dir,
            lex_topk=args.lex_topk,
            smt_topk=args.smt_topk,
            batch_size=args.batch_size,
            n_gpu=args.n_gpu,
            skip_existing=not args.force,
        )
        print(f"[naive] {lang} done. Errors: {result.get('error', 'none')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Baseline B: Sequential-only (planner disabled).

Runs the sequential decoder without the lexical planner constraint
on non-English queries.  This isolates whether the planner helps or
hurts under cross-lingual mismatch.

Produces:
- Stage 2 run file (unconstrained beam search) in TREC format
- Metrics JSON (MRR@10, Recall@10, nDCG@10, latency, error_rate)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data_loader import (
    RQ3_LANGUAGES,
    prepare_query_files,
    save_coverage_report,
    save_filtered_qrels,
)
from cross_lingual.retrieval_engine import (
    run_sequential_unconstrained,
    count_queries,
    load_sequential_run,
    run_json_to_trec,
    _get_dataset_name,
)
from cross_lingual.evaluate import compute_metrics, save_metrics


def run_sequential_baseline(
    language: str,
    output_dir: str,
    smt_topk: int = 100,
    batch_size: int = 16,
    n_gpu: int = 1,
    skip_existing: bool = True,
) -> dict:
    """Run Baseline B (sequential-only) for one language."""
    out = Path(output_dir) / language / "sequential"
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
    smt_out_dir = str(out / "smt_ret")

    result = {
        "baseline": "sequential",
        "language": language,
        "n_queries": n_queries,
    }

    # Check existing
    smt_run_path = os.path.join(smt_out_dir, dataset_name, "run.json")
    if os.path.exists(smt_run_path) and skip_existing:
        print(f"[seq] Run exists, skipping inference")
        result["inference_time_s"] = 0
    else:
        rc, elapsed = run_sequential_unconstrained(
            query_dir=str(tgt_dir),
            smt_out_dir=smt_out_dir,
            topk=smt_topk,
            batch_size=batch_size,
            n_gpu=n_gpu,
        )
        result["inference_time_s"] = elapsed
        if rc != 0:
            result["error"] = f"seq_failed_rc={rc}"
            return result

    # Merge
    from robustness.utils.pag_inference import merge_and_evaluate
    merge_and_evaluate(str(tgt_dir), smt_out_dir, str(qrel_path))

    # Load run, save TREC
    smt_run = load_sequential_run(smt_out_dir, dataset_name)
    if smt_run:
        trec_path = str(out / "stage2_run.trec")
        run_json_to_trec(smt_run, trec_path, f"seq_{language}")

    # Metrics
    qrels = json.load(open(qrel_path))
    if smt_run:
        metrics = compute_metrics(
            smt_run, qrels, n_queries,
            result.get("inference_time_s", 0),
        )
        result["stage2_metrics"] = metrics
        save_metrics(metrics, out / "stage2_metrics.json")

    with open(out / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Baseline B: Sequential-only"
    )
    parser.add_argument("--language", type=str, required=True,
                        help=f"Language: {RQ3_LANGUAGES} or 'all'")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "cross_lingual" / "results"))
    parser.add_argument("--smt_topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = RQ3_LANGUAGES if args.language == "all" else [args.language]

    for lang in languages:
        print(f"\n{'='*60}")
        print(f"[seq] Running Baseline B for {lang}")
        print(f"{'='*60}")
        result = run_sequential_baseline(
            language=lang,
            output_dir=args.output_dir,
            smt_topk=args.smt_topk,
            batch_size=args.batch_size,
            n_gpu=args.n_gpu,
            skip_existing=not args.force,
        )
        print(f"[seq] {lang} done. Errors: {result.get('error', 'none')}")


if __name__ == "__main__":
    main()

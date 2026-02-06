#!/usr/bin/env python3
"""
Evaluation module for RQ3 cross-lingual experiments.

Primary metrics:
- MRR@10
- Recall@10
- nDCG@10 (optional)

Additional:
- avg_latency_ms_per_query
- error_rate

Reuses pytrec_eval for metric computation.
Supports optional paired bootstrap significance testing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    n_total_queries: int = 0,
    total_time_s: float = 0.0,
) -> Dict[str, float]:
    """Compute MRR@10, Recall@10, nDCG@10, latency, error_rate.

    Args:
        run: {qid: {pid: score}}
        qrels: {qid: {pid: relevance}}
        n_total_queries: total queries attempted (for error_rate)
        total_time_s: total wall-clock time in seconds

    Returns:
        Dict with metric values
    """
    import pytrec_eval

    # Filter run to queries present in qrels
    str_run = {
        str(qid): {str(pid): float(s) for pid, s in docs.items()}
        for qid, docs in run.items()
        if str(qid) in qrels
    }
    str_qrels = {
        str(qid): {str(pid): int(s) for pid, s in docs.items()}
        for qid, docs in qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        str_qrels,
        {"ndcg_cut_10", "recip_rank", "recall_10"},
    )
    per_query = evaluator.evaluate(str_run)
    n_evaluated = len(per_query)

    if n_evaluated == 0:
        return {
            "MRR@10": 0.0,
            "Recall@10": 0.0,
            "nDCG@10": 0.0,
            "n_evaluated": 0,
            "avg_latency_ms_per_query": 0.0,
            "error_rate": 1.0 if n_total_queries > 0 else 0.0,
        }

    mrr10 = np.mean([v["recip_rank"] for v in per_query.values()])
    recall10 = np.mean([v["recall_10"] for v in per_query.values()])
    ndcg10 = np.mean([v["ndcg_cut_10"] for v in per_query.values()])

    # Latency
    avg_latency_ms = 0.0
    if total_time_s > 0 and n_evaluated > 0:
        avg_latency_ms = (total_time_s * 1000) / n_evaluated

    # Error rate
    n_total = max(n_total_queries, n_evaluated)
    n_errors = max(0, n_total - n_evaluated)
    error_rate = n_errors / n_total if n_total > 0 else 0.0

    return {
        "MRR@10": float(mrr10),
        "Recall@10": float(recall10),
        "nDCG@10": float(ndcg10),
        "n_evaluated": n_evaluated,
        "avg_latency_ms_per_query": float(avg_latency_ms),
        "error_rate": float(error_rate),
    }


def compute_per_query_metrics(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Per-query MRR@10, Recall@10, nDCG@10."""
    import pytrec_eval

    str_run = {
        str(qid): {str(pid): float(s) for pid, s in docs.items()}
        for qid, docs in run.items()
        if str(qid) in qrels
    }
    str_qrels = {
        str(qid): {str(pid): int(s) for pid, s in docs.items()}
        for qid, docs in qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        str_qrels,
        {"ndcg_cut_10", "recip_rank", "recall_10"},
    )
    per_query = evaluator.evaluate(str_run)

    result = {}
    for qid, vals in per_query.items():
        result[qid] = {
            "MRR@10": vals["recip_rank"],
            "Recall@10": vals["recall_10"],
            "nDCG@10": vals["ndcg_cut_10"],
        }
    return result


# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict, path) -> Path:
    """Save metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Saved metrics -> {path}")
    return path


# ---------------------------------------------------------------------------
# Significance testing (paired bootstrap)
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    per_query_a: Dict[str, float],
    per_query_b: Dict[str, float],
    metric: str = "nDCG@10",
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap significance test.

    Tests H0: mean(system_a) == mean(system_b).

    Args:
        per_query_a: {qid: metric_value} for system A
        per_query_b: {qid: metric_value} for system B
        metric: name (for reporting)
        n_resamples: number of bootstrap resamples

    Returns:
        {"mean_a": float, "mean_b": float, "delta": float,
         "p_value": float, "significant_at_05": bool}
    """
    common = sorted(set(per_query_a.keys()) & set(per_query_b.keys()))
    if not common:
        return {"error": "no_common_queries"}

    vals_a = np.array([per_query_a[q] for q in common])
    vals_b = np.array([per_query_b[q] for q in common])

    observed_delta = np.mean(vals_a) - np.mean(vals_b)

    rng = np.random.RandomState(seed)
    n = len(common)
    count_ge = 0

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_a = vals_a[idx]
        boot_b = vals_b[idx]
        boot_delta = np.mean(boot_a) - np.mean(boot_b)
        if abs(boot_delta) >= abs(observed_delta):
            count_ge += 1

    p_value = count_ge / n_resamples

    return {
        "metric": metric,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "delta": float(observed_delta),
        "p_value": float(p_value),
        "significant_at_05": p_value < 0.05,
        "n_queries": n,
    }


# ---------------------------------------------------------------------------
# Aggregate results across baselines and languages
# ---------------------------------------------------------------------------

def aggregate_rq3_results(results_dir: str) -> Dict:
    """Load and aggregate all RQ3 results.

    Expects:
      results_dir/{lang}/{baseline}/stage2_metrics.json

    Returns summary dict.
    """
    from cross_lingual.data_loader import RQ3_LANGUAGES

    base = Path(results_dir)
    baselines = ["naive", "sequential", "translate"]
    all_results = {}

    for lang in RQ3_LANGUAGES:
        lang_results = {}
        for bl in baselines:
            s1_path = base / lang / bl / "stage1_metrics.json"
            s2_path = base / lang / bl / "stage2_metrics.json"

            entry = {"baseline": bl, "language": lang}
            if s1_path.exists():
                with open(s1_path) as f:
                    entry["stage1"] = json.load(f)
            if s2_path.exists():
                with open(s2_path) as f:
                    entry["stage2"] = json.load(f)

            lang_results[bl] = entry

        all_results[lang] = lang_results

    return all_results


def print_results_table(all_results: Dict):
    """Print a formatted results table to stdout."""
    header = (
        f"{'Lang':>4s} {'Baseline':>12s} | "
        f"{'MRR@10':>8s} {'Recall@10':>10s} {'nDCG@10':>8s} | "
        f"{'S1-MRR@10':>10s} {'S1-R@10':>8s}"
    )
    print("\n" + "=" * len(header))
    print("RQ3 Cross-lingual Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for lang, lang_results in sorted(all_results.items()):
        for bl_name, entry in sorted(lang_results.items()):
            s2 = entry.get("stage2", {})
            s1 = entry.get("stage1", {})
            print(
                f"{lang:>4s} {bl_name:>12s} | "
                f"{s2.get('MRR@10', 0)*100:>7.2f}% "
                f"{s2.get('Recall@10', 0)*100:>9.2f}% "
                f"{s2.get('nDCG@10', 0)*100:>7.2f}% | "
                f"{s1.get('MRR@10', 0)*100:>9.2f}% "
                f"{s1.get('Recall@10', 0)*100:>7.2f}%"
            )
        print("-" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 Evaluate")
    parser.add_argument(
        "--results_dir", type=str,
        default=str(REPO_ROOT / "cross_lingual" / "results"),
    )
    parser.add_argument("--significance", action="store_true",
                        help="Run bootstrap significance tests")
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = aggregate_rq3_results(args.results_dir)
    print_results_table(all_results)

    # Save aggregated summary
    summary_path = Path(args.results_dir) / "rq3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Summary saved -> {summary_path}")

    # Significance testing
    if args.significance:
        print("\n[eval] Running bootstrap significance tests...")
        _run_significance_tests(args.results_dir, all_results)


def _run_significance_tests(results_dir: str, all_results: Dict):
    """Run pairwise significance tests between baselines."""
    from cross_lingual.data_loader import RQ3_LANGUAGES

    base = Path(results_dir)
    sig_results = []

    for lang in RQ3_LANGUAGES:
        if lang not in all_results:
            continue

        # Load per-query metrics for each baseline
        pq_by_bl = {}
        for bl in ["naive", "sequential", "translate"]:
            result_path = base / lang / bl / "result.json"
            if not result_path.exists():
                continue

            # Compute per-query from run files
            qrel_path = base / lang / bl / "qrels.json"
            if not qrel_path.exists():
                continue

            smt_out = base / lang / bl / "smt_ret"
            # Find dataset
            for ds in ["MSMARCO", "TREC_DL_2019", "TREC_DL_2020", "other_dataset"]:
                run_path = smt_out / ds / "run.json"
                if run_path.exists():
                    with open(run_path) as f:
                        run = json.load(f)
                    with open(qrel_path) as f:
                        qrels = json.load(f)
                    pq = compute_per_query_metrics(run, qrels)
                    pq_by_bl[bl] = pq
                    break

        # Compare pairs
        pairs = [("naive", "sequential"), ("naive", "translate")]
        for bl_a, bl_b in pairs:
            if bl_a not in pq_by_bl or bl_b not in pq_by_bl:
                continue

            for metric in ["MRR@10", "nDCG@10"]:
                vals_a = {q: v[metric] for q, v in pq_by_bl[bl_a].items()}
                vals_b = {q: v[metric] for q, v in pq_by_bl[bl_b].items()}
                test = paired_bootstrap_test(vals_a, vals_b, metric)
                test["language"] = lang
                test["system_a"] = bl_a
                test["system_b"] = bl_b
                sig_results.append(test)

                p = test["p_value"]
                sig = "*" if test["significant_at_05"] else ""
                print(f"  {lang} {bl_a} vs {bl_b} ({metric}): "
                      f"delta={test['delta']:.4f}, p={p:.4f}{sig}")

    sig_path = base / "significance_tests.json"
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2)
    print(f"[eval] Significance results -> {sig_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plan diagnostics for RQ3: planner stability between queries.

Compares the planner's behaviour on:
- non-English query  q_non_en
- translated query   q_trans_en    (same qid)

Metrics per qid:
  CandOverlap@100  - Jaccard of top-100 planner candidate sets
  TokJaccard@100   - Jaccard of top-100 planner vocabulary tokens

Reported summary stats per language:
  mean, median, p10, p25, p75, p90

Saves as JSON and CSV for paper inclusion.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data_loader import RQ3_LANGUAGES
from cross_lingual.retrieval_engine import (
    load_lexical_run,
    load_planner_tokens,
    _get_dataset_name,
)


# ---------------------------------------------------------------------------
# CandOverlap@K: Jaccard of top-K planner candidate sets
# ---------------------------------------------------------------------------

def cand_overlap_at_k(
    run_a: Dict[str, Dict[str, float]],
    run_b: Dict[str, Dict[str, float]],
    topk: int = 100,
) -> Dict[str, float]:
    """Per-query Jaccard overlap of top-K candidate document sets.

    Args:
        run_a, run_b: {qid: {docid: score}}  (planner outputs)
        topk: cutoff

    Returns:
        {qid: jaccard}
    """
    common = set(run_a.keys()) & set(run_b.keys())
    results = {}
    for qid in common:
        top_a = set(_topk_docs(run_a[qid], topk))
        top_b = set(_topk_docs(run_b[qid], topk))
        inter = top_a & top_b
        union = top_a | top_b
        results[qid] = len(inter) / len(union) if union else 1.0
    return results


def _topk_docs(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Return top-k document IDs by score."""
    return [d for d, _ in sorted(doc_scores.items(),
                                  key=lambda x: x[1], reverse=True)[:k]]


# ---------------------------------------------------------------------------
# TokJaccard@K: Jaccard of top-K planner vocabulary tokens
# ---------------------------------------------------------------------------

def tok_jaccard_at_k(
    tokens_a: Dict[str, List[int]],
    tokens_b: Dict[str, List[int]],
    topk: int = 100,
) -> Dict[str, float]:
    """Per-query Jaccard of top-K planner token sets.

    Args:
        tokens_a, tokens_b: {qid: [token_ids]} sorted by descending score
        topk: cutoff

    Returns:
        {qid: jaccard}
    """
    common = set(tokens_a.keys()) & set(tokens_b.keys())
    results = {}
    for qid in common:
        set_a = set(tokens_a[qid][:topk])
        set_b = set(tokens_b[qid][:topk])
        inter = set_a & set_b
        union = set_a | set_b
        results[qid] = len(inter) / len(union) if union else 1.0
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, p10, p25, p75, p90."""
    if not values:
        return {k: 0.0 for k in [
            "mean", "median", "p10", "p25", "p75", "p90",
        ]}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Main diagnostic computation
# ---------------------------------------------------------------------------

def compute_diagnostics(
    nonen_lex_run: Dict[str, Dict[str, float]],
    trans_lex_run: Dict[str, Dict[str, float]],
    nonen_tokens: Optional[Dict[str, List[int]]] = None,
    trans_tokens: Optional[Dict[str, List[int]]] = None,
    topk: int = 100,
) -> Dict:
    """Compute all plan diagnostics for one language.

    Args:
        nonen_lex_run: Planner output on non-English queries
        trans_lex_run: Planner output on translated-to-English queries
        nonen_tokens: Planner tokens for non-English queries
        trans_tokens: Planner tokens for translated queries
        topk: cutoff for overlap computation

    Returns:
        {
          "CandOverlap@100": {"per_query": {qid: val}, "summary": {...}},
          "TokJaccard@100":  {"per_query": {qid: val}, "summary": {...}},
          "n_queries": int,
        }
    """
    result = {}

    # CandOverlap
    cand_pq = cand_overlap_at_k(nonen_lex_run, trans_lex_run, topk)
    result[f"CandOverlap@{topk}"] = {
        "per_query": cand_pq,
        "summary": summary_stats(list(cand_pq.values())),
    }

    # TokJaccard (if tokens available)
    if nonen_tokens and trans_tokens:
        tok_pq = tok_jaccard_at_k(nonen_tokens, trans_tokens, topk)
        result[f"TokJaccard@{topk}"] = {
            "per_query": tok_pq,
            "summary": summary_stats(list(tok_pq.values())),
        }

    result["n_queries"] = len(cand_pq)
    return result


# ---------------------------------------------------------------------------
# Save as JSON and CSV
# ---------------------------------------------------------------------------

def save_diagnostics(
    diagnostics: Dict,
    language: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save diagnostics as JSON and flat CSV.

    Returns (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full)
    json_path = output_dir / f"plan_diagnostics_{language}.json"
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    # CSV summary table
    csv_path = output_dir / f"plan_diagnostics_{language}.csv"
    rows = []
    for metric_key in sorted(diagnostics.keys()):
        if metric_key == "n_queries":
            continue
        entry = diagnostics[metric_key]
        if "summary" not in entry:
            continue
        row = {"language": language, "metric": metric_key}
        row.update(entry["summary"])
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"[diag] Saved {json_path}")
    print(f"[diag] Saved {csv_path}")
    return json_path, csv_path


# ---------------------------------------------------------------------------
# Runner for all languages
# ---------------------------------------------------------------------------

def run_all_diagnostics(
    results_dir: str,
    topk: int = 100,
) -> Dict[str, Dict]:
    """Run diagnostics for all languages found in results_dir.

    Expects the directory layout:
      results_dir/{lang}/naive/lex_ret/{dataset}/run.json
      results_dir/{lang}/translate/lex_ret/{dataset}/run.json
      results_dir/{lang}/naive/planner_tokens/...
      results_dir/{lang}/translate/planner_tokens/...

    Returns {language: diagnostics_dict}.
    """
    all_diag = {}
    base = Path(results_dir)
    diag_dir = base / "diagnostics"

    for lang in RQ3_LANGUAGES:
        naive_lex_base = base / lang / "naive" / "lex_ret"
        trans_lex_base = base / lang / "translate" / "lex_ret"

        # Auto-detect dataset name from existing directories
        dataset_name = None
        for candidate in ["MSMARCO", "TREC_DL_2019", "TREC_DL_2020", "other_dataset"]:
            if (naive_lex_base / candidate / "run.json").exists():
                dataset_name = candidate
                break

        if dataset_name is None:
            print(f"[diag] No lexical runs found for {lang}, skipping")
            continue

        nonen_run = load_lexical_run(str(naive_lex_base), dataset_name)
        trans_run = load_lexical_run(str(trans_lex_base), dataset_name)

        if not nonen_run or not trans_run:
            print(f"[diag] Missing runs for {lang}, skipping")
            continue

        # Load planner tokens if available
        nonen_tok_path = base / lang / "naive" / "planner_tokens" / "tokens.json"
        trans_tok_path = base / lang / "translate" / "planner_tokens" / "tokens.json"
        nonen_tokens = load_planner_tokens(str(nonen_tok_path)) if nonen_tok_path.exists() else None
        trans_tokens = load_planner_tokens(str(trans_tok_path)) if trans_tok_path.exists() else None

        diag = compute_diagnostics(
            nonen_run, trans_run,
            nonen_tokens, trans_tokens,
            topk=topk,
        )
        save_diagnostics(diag, lang, diag_dir)
        all_diag[lang] = diag

    # Combined summary CSV
    if all_diag:
        _save_combined_csv(all_diag, diag_dir)

    return all_diag


def _save_combined_csv(all_diag: Dict[str, Dict], output_dir: Path):
    """Save a combined CSV with one row per (language, metric)."""
    rows = []
    for lang, diag in sorted(all_diag.items()):
        for metric_key in sorted(diag.keys()):
            if metric_key == "n_queries":
                continue
            entry = diag[metric_key]
            if "summary" not in entry:
                continue
            row = {"language": lang, "metric": metric_key}
            row.update(entry["summary"])
            rows.append(row)

    if rows:
        csv_path = output_dir / "plan_diagnostics_all.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[diag] Combined CSV -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Plan diagnostics: planner stability analysis"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(REPO_ROOT / "cross_lingual" / "results"),
        help="Directory containing per-language results",
    )
    parser.add_argument("--topk", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[diag] Running plan diagnostics on {args.results_dir}")
    all_diag = run_all_diagnostics(args.results_dir, topk=args.topk)

    for lang, diag in all_diag.items():
        print(f"\n{lang}:")
        for key in sorted(diag.keys()):
            if key == "n_queries":
                print(f"  n_queries: {diag[key]}")
                continue
            entry = diag[key]
            if "summary" in entry:
                s = entry["summary"]
                print(f"  {key}: mean={s['mean']:.4f}  "
                      f"median={s['median']:.4f}  "
                      f"p10={s['p10']:.4f}  p90={s['p90']:.4f}")


if __name__ == "__main__":
    main()

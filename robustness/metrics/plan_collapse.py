"""
Metrics for RQ2 "plan collapse" analysis.

Plan collapse measures how much the PAG lexical planner's candidate set
degrades when queries are perturbed, and how much the sequential decoder
can recover from that degradation.

Metrics
-------
1. **Candidate-set overlap** (Jaccard / Intersection-over-Union):
   Overlap of top-K planner candidate lists between clean and perturbed.

2. **Candidate-set size ratio**:
   |candidates_perturbed| / |candidates_clean| for each qid.

3. **Score correlation** (Spearman rank correlation):
   Rank correlation of document scores in the planner output.

4. **Sequential recovery delta**:
   Δ = (metric_clean_lex - metric_perturbed_lex) -
       (metric_clean_smt - metric_perturbed_smt)
   Positive values indicate the sequential decoder mitigates degradation.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Plan-level metrics  (lexical planner runs)
# ---------------------------------------------------------------------------

def candidate_overlap(
    clean_run: Dict[str, Dict[str, float]],
    perturbed_run: Dict[str, Dict[str, float]],
    topk: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Per-query candidate-set overlap between clean and perturbed lexical runs.

    Returns {qid: {"jaccard": float, "intersection_size": int,
                    "clean_size": int, "perturbed_size": int}}.
    """
    results = {}
    common_qids = set(clean_run.keys()) & set(perturbed_run.keys())

    for qid in common_qids:
        clean_docs = set(
            _top_k_docs(clean_run[qid], topk)
        )
        perturbed_docs = set(
            _top_k_docs(perturbed_run[qid], topk)
        )
        inter = clean_docs & perturbed_docs
        union = clean_docs | perturbed_docs
        jaccard = len(inter) / len(union) if union else 1.0

        results[qid] = {
            "jaccard": jaccard,
            "intersection_size": len(inter),
            "clean_size": len(clean_docs),
            "perturbed_size": len(perturbed_docs),
        }
    return results


def candidate_size_ratio(
    clean_run: Dict[str, Dict[str, float]],
    perturbed_run: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Per-query ratio of |perturbed candidates| / |clean candidates|.
    Values < 1 indicate the perturbed planner returns fewer candidates.
    """
    ratios = {}
    for qid in set(clean_run) & set(perturbed_run):
        c = len(clean_run[qid])
        p = len(perturbed_run[qid])
        ratios[qid] = p / c if c > 0 else float("nan")
    return ratios


def plan_intersect(
    clean_tokens: Dict[str, List[int]],
    perturbed_tokens: Dict[str, List[int]],
    topk: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Per-query overlap of top-K planner *tokens* (vocabulary IDs) between
    clean and perturbed queries.

    This measures token-plan drift: whether the planner selects similar
    vocabulary terms for the clean vs. perturbed query.

    Parameters
    ----------
    clean_tokens : {qid: [token_id, ...]}
        Top-K planner token IDs for clean queries (sorted by descending score).
    perturbed_tokens : {qid: [token_id, ...]}
        Top-K planner token IDs for perturbed queries.

    Returns
    -------
    {qid: {"jaccard": float, "intersection_size": int,
            "clean_size": int, "perturbed_size": int}}
    """
    results = {}
    common_qids = set(clean_tokens.keys()) & set(perturbed_tokens.keys())

    for qid in common_qids:
        clean_set = set(clean_tokens[qid][:topk])
        perturbed_set = set(perturbed_tokens[qid][:topk])
        inter = clean_set & perturbed_set
        union = clean_set | perturbed_set
        jaccard = len(inter) / len(union) if union else 1.0

        results[qid] = {
            "jaccard": jaccard,
            "intersection_size": len(inter),
            "clean_size": len(clean_set),
            "perturbed_size": len(perturbed_set),
        }
    return results


def rank_correlation(
    clean_run: Dict[str, Dict[str, float]],
    perturbed_run: Dict[str, Dict[str, float]],
    topk: int = 100,
) -> Dict[str, float]:
    """
    Per-query Spearman rank correlation of the top-K document scores.
    """
    from scipy.stats import spearmanr

    correlations = {}
    for qid in set(clean_run) & set(perturbed_run):
        clean_top = _top_k_docs(clean_run[qid], topk)
        perturbed_top = _top_k_docs(perturbed_run[qid], topk)

        # Build a common set and compare ranks
        all_docs = list(set(clean_top) | set(perturbed_top))
        if len(all_docs) < 2:
            correlations[qid] = float("nan")
            continue

        clean_rank = {d: i for i, d in enumerate(clean_top)}
        perturbed_rank = {d: i for i, d in enumerate(perturbed_top)}

        max_rank = len(all_docs)
        r_clean = [clean_rank.get(d, max_rank) for d in all_docs]
        r_perturbed = [perturbed_rank.get(d, max_rank) for d in all_docs]

        corr, _ = spearmanr(r_clean, r_perturbed)
        correlations[qid] = corr
    return correlations


# ---------------------------------------------------------------------------
# Sequential recovery delta
# ---------------------------------------------------------------------------

def recovery_delta(
    clean_lex_metric: float,
    perturbed_lex_metric: float,
    clean_smt_metric: float,
    perturbed_smt_metric: float,
) -> float:
    """
    Δ_recovery = drop_lex - drop_smt.
    Positive means the sequential decoder mitigates degradation.
    """
    drop_lex = clean_lex_metric - perturbed_lex_metric
    drop_smt = clean_smt_metric - perturbed_smt_metric
    return drop_lex - drop_smt


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_plan_collapse(
    clean_lex_run: Dict[str, Dict[str, float]],
    perturbed_lex_run: Dict[str, Dict[str, float]],
    clean_smt_run: Optional[Dict[str, Dict[str, float]]] = None,
    perturbed_smt_run: Optional[Dict[str, Dict[str, float]]] = None,
    topk: int = 100,
    clean_planner_tokens: Optional[Dict[str, List[int]]] = None,
    perturbed_planner_tokens: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate plan-collapse statistics.

    Returns a dict with:
      - avg_jaccard@{topk}          (CandOverlap@K)
      - avg_size_ratio
      - avg_rank_correlation@{topk}
      - avg_plan_intersect@{topk}   (PlanIntersect@K, if tokens provided)
      - n_queries
    """
    overlap = candidate_overlap(clean_lex_run, perturbed_lex_run, topk)
    size_ratio = candidate_size_ratio(clean_lex_run, perturbed_lex_run)

    n = len(overlap)
    stats = {
        f"avg_jaccard@{topk}": _safe_mean([v["jaccard"] for v in overlap.values()]),
        f"avg_intersection@{topk}": _safe_mean(
            [v["intersection_size"] for v in overlap.values()]
        ),
        "avg_size_ratio": _safe_mean(list(size_ratio.values())),
        "n_queries": n,
    }

    # Optional: rank correlation (scipy may not be available)
    try:
        corr = rank_correlation(clean_lex_run, perturbed_lex_run, topk)
        stats[f"avg_rank_correlation@{topk}"] = _safe_mean(list(corr.values()))
    except ImportError:
        stats[f"avg_rank_correlation@{topk}"] = None

    # PlanIntersect: token-level plan overlap
    if clean_planner_tokens and perturbed_planner_tokens:
        pi = plan_intersect(clean_planner_tokens, perturbed_planner_tokens, topk)
        stats[f"avg_plan_intersect@{topk}"] = _safe_mean(
            [v["jaccard"] for v in pi.values()]
        )
    else:
        stats[f"avg_plan_intersect@{topk}"] = None

    return stats


# ---------------------------------------------------------------------------
# Evaluate and compare  (retrieval metrics: NDCG@10, MRR@10)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute NDCG@k and MRR@k using pytrec_eval.
    """
    import pytrec_eval

    # Filter run to only queries present in qrels
    filtered_run = {qid: docs for qid, docs in run.items() if qid in qrels}

    # Convert scores to the format pytrec_eval expects
    str_run = {}
    for qid, docs in filtered_run.items():
        str_run[str(qid)] = {str(did): float(s) for did, s in docs.items()}

    str_qrels = {}
    for qid, docs in qrels.items():
        str_qrels[str(qid)] = {str(did): int(s) for did, s in docs.items()}

    evaluator = pytrec_eval.RelevanceEvaluator(str_qrels, {f"ndcg_cut_{k}", "recip_rank"})
    results = evaluator.evaluate(str_run)

    ndcg_values = [v[f"ndcg_cut_{k}"] for v in results.values()]
    mrr_values = [v["recip_rank"] for v in results.values()]

    return {
        f"NDCG@{k}": _safe_mean(ndcg_values),
        f"MRR@{k}": _safe_mean(mrr_values),
        "n_evaluated": len(results),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_k_docs(
    rankdata: Dict[str, float], k: int
) -> List[str]:
    """Return the top-k document IDs by score."""
    sorted_docs = sorted(rankdata.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in sorted_docs[:k]]


def _safe_mean(values: List[float]) -> float:
    """Mean that handles empty lists and NaN values."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else 0.0

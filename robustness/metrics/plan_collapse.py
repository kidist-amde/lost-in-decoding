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

    Returns {qid: {"jaccard": float, "overlap_at_n": float,
                    "intersection_size": int,
                    "clean_size": int, "perturbed_size": int}}.

    ``jaccard`` is the standard Jaccard: |C∩C̃| / |C∪C̃|.
    ``overlap_at_n`` is the alternative overlap: |C∩C̃| / n.
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
        overlap_at_n = len(inter) / topk if topk > 0 else 0.0

        results[qid] = {
            "jaccard": jaccard,
            "overlap_at_n": overlap_at_n,
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


def _extract_token_ids(token_data) -> List[int]:
    """Extract token ID list from either new or legacy format.

    New format: ``{"token_ids": [...], "scores": [...]}``
    Legacy format: ``[int, ...]``
    """
    if isinstance(token_data, dict) and "token_ids" in token_data:
        return token_data["token_ids"]
    return token_data


def plan_intersect(
    clean_tokens: Dict,
    perturbed_tokens: Dict,
    topk: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Per-query overlap of top-K planner *tokens* (vocabulary IDs) between
    clean and perturbed queries.

    This measures token-plan drift: whether the planner selects similar
    vocabulary terms for the clean vs. perturbed query.

    Parameters
    ----------
    clean_tokens : {qid: [token_id, ...]} or {qid: {"token_ids": [...], "scores": [...]}}
        Top-K planner token IDs for clean queries (sorted by descending score).
    perturbed_tokens : {qid: [token_id, ...]} or {qid: {"token_ids": [...], "scores": [...]}}
        Top-K planner token IDs for perturbed queries.

    Returns
    -------
    {qid: {"jaccard": float, "overlap_at_ell": float,
            "intersection_size": int,
            "clean_size": int, "perturbed_size": int}}

    ``jaccard`` is the standard Jaccard: |P∩Q̃| / |P∪Q̃|.
    ``overlap_at_ell`` is the alternative overlap: |P∩Q̃| / ℓ.
    """
    results = {}
    common_qids = set(clean_tokens.keys()) & set(perturbed_tokens.keys())

    for qid in common_qids:
        clean_set = set(_extract_token_ids(clean_tokens[qid])[:topk])
        perturbed_set = set(_extract_token_ids(perturbed_tokens[qid])[:topk])
        inter = clean_set & perturbed_set
        union = clean_set | perturbed_set
        jaccard = len(inter) / len(union) if union else 1.0
        overlap_at_ell = len(inter) / topk if topk > 0 else 0.0

        results[qid] = {
            "jaccard": jaccard,
            "overlap_at_ell": overlap_at_ell,
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

    Returns a dict with distributional stats (mean, median, p10/p25/p75/p90)
    for each stability metric, plus backward-compatible ``avg_*`` keys.

    Keys include:
      - CandOverlap@{topk}_mean/median/p10/p25/p75/p90
      - CandOverlapAtN@{topk}_mean/...       (|C∩C̃| / n)
      - TokJaccard@{topk}_mean/...           (if tokens provided)
      - TokOverlapAtEll@{topk}_mean/...      (|P∩P̃| / ℓ, if tokens provided)
      - avg_jaccard@{topk}                   (backward compat = CandOverlap mean)
      - avg_plan_intersect@{topk}            (backward compat = TokJaccard mean)
      - avg_size_ratio, avg_rank_correlation@{topk}, n_queries
    """
    overlap = candidate_overlap(clean_lex_run, perturbed_lex_run, topk)
    size_ratio = candidate_size_ratio(clean_lex_run, perturbed_lex_run)

    n = len(overlap)
    jaccard_vals = [v["jaccard"] for v in overlap.values()]
    overlap_at_n_vals = [v["overlap_at_n"] for v in overlap.values()]

    stats: Dict[str, float] = {"n_queries": n}

    # CandOverlap distributional stats
    stats.update(_distribution_stats(jaccard_vals, f"CandOverlap@{topk}"))
    stats.update(_distribution_stats(overlap_at_n_vals, f"CandOverlapAtN@{topk}"))

    # Backward-compatible keys
    stats[f"avg_jaccard@{topk}"] = stats[f"CandOverlap@{topk}_mean"]
    stats[f"avg_intersection@{topk}"] = _safe_mean(
        [v["intersection_size"] for v in overlap.values()]
    )
    stats["avg_size_ratio"] = _safe_mean(list(size_ratio.values()))

    # Optional: rank correlation (scipy may not be available)
    try:
        corr = rank_correlation(clean_lex_run, perturbed_lex_run, topk)
        stats[f"avg_rank_correlation@{topk}"] = _safe_mean(list(corr.values()))
    except ImportError:
        stats[f"avg_rank_correlation@{topk}"] = None

    # TokJaccard: token-level plan overlap distributional stats
    if clean_planner_tokens and perturbed_planner_tokens:
        pi = plan_intersect(clean_planner_tokens, perturbed_planner_tokens, topk)
        tok_jaccard_vals = [v["jaccard"] for v in pi.values()]
        tok_overlap_vals = [v["overlap_at_ell"] for v in pi.values()]
        stats.update(_distribution_stats(tok_jaccard_vals, f"TokJaccard@{topk}"))
        stats.update(_distribution_stats(tok_overlap_vals, f"TokOverlapAtEll@{topk}"))
        # Backward compat
        stats[f"avg_plan_intersect@{topk}"] = stats[f"TokJaccard@{topk}_mean"]
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


def compute_retrieval_metrics_per_query(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Per-query NDCG@k and MRR@k using pytrec_eval.

    Returns {qid: {"NDCG@k": float, "MRR@k": float}}.
    """
    import pytrec_eval

    filtered_run = {qid: docs for qid, docs in run.items() if qid in qrels}

    str_run = {}
    for qid, docs in filtered_run.items():
        str_run[str(qid)] = {str(did): float(s) for did, s in docs.items()}

    str_qrels = {}
    for qid, docs in qrels.items():
        str_qrels[str(qid)] = {str(did): int(s) for did, s in docs.items()}

    evaluator = pytrec_eval.RelevanceEvaluator(str_qrels, {f"ndcg_cut_{k}", "recip_rank"})
    results = evaluator.evaluate(str_run)

    per_query = {}
    for qid, vals in results.items():
        per_query[qid] = {
            f"NDCG@{k}": vals[f"ndcg_cut_{k}"],
            f"MRR@{k}": vals["recip_rank"],
        }
    return per_query


# ---------------------------------------------------------------------------
# Plan collapse classification
# ---------------------------------------------------------------------------

def classify_plan_collapse(
    cand_overlap_per_query: Dict[str, Dict[str, float]],
    tok_jaccard_per_query: Optional[Dict[str, Dict[str, float]]],
    clean_lex_pq: Dict[str, Dict[str, float]],
    pert_lex_pq: Dict[str, Dict[str, float]],
    metric_name: str = "NDCG@10",
    tau: Optional[float] = None,
    delta: float = 0.05,
    tau_percentile: float = 10.0,
) -> Dict[str, object]:
    """
    Classify per-query plan-collapse tail events.

    A query is *collapsed* when:
        (CandOverlap@n < τ  OR  TokJaccard@ℓ < τ)
        AND  ΔM_SimulOnly(q, q̃) ≤ -δ

    where ΔM_SimulOnly = M(q̃) - M(q).

    Parameters
    ----------
    cand_overlap_per_query : per-query output of ``candidate_overlap()``.
    tok_jaccard_per_query  : per-query output of ``plan_intersect()`` (may be None).
    clean_lex_pq : per-query metrics for clean lexical run (from
                   ``compute_retrieval_metrics_per_query``).
    pert_lex_pq  : per-query metrics for perturbed lexical run.
    metric_name  : which metric to use for the performance-drop test
                   (``"NDCG@10"`` or ``"MRR@10"``).
    tau          : explicit threshold for overlap / Jaccard.  If ``None``
                   (default), τ is set to the ``tau_percentile``-th percentile
                   of the observed CandOverlap distribution.
    delta        : absolute-drop threshold (positive).  Collapse requires
                   ΔM ≤ -delta.
    tau_percentile : percentile used to derive τ when ``tau is None``.

    Returns
    -------
    dict with:
      - ``"collapsed"``      : {qid: bool}
      - ``"low_stability"``  : {qid: bool}
      - ``"delta_simulonly"``: {qid: float}, where ΔM = M(q̃)-M(q)
      - ``"tau"``            : float (threshold used)
      - ``"delta"``          : float
      - ``"n_collapsed"``    : int
      - ``"collapse_rate"``  : float (n_collapsed / n_queries)
      - ``"n_queries"``      : int
    """
    common_qids = sorted(
        set(cand_overlap_per_query) & set(clean_lex_pq) & set(pert_lex_pq)
    )

    # Derive τ from the CandOverlap distribution if not given
    if tau is None:
        cand_jaccards = [cand_overlap_per_query[q]["jaccard"] for q in common_qids]
        tau = float(np.percentile(cand_jaccards, tau_percentile)) if cand_jaccards else 0.0

    collapsed: Dict[str, bool] = {}
    low_stability: Dict[str, bool] = {}
    delta_simulonly: Dict[str, float] = {}
    for qid in common_qids:
        cand_below = cand_overlap_per_query[qid]["jaccard"] < tau
        tok_below = False
        if tok_jaccard_per_query and qid in tok_jaccard_per_query:
            tok_below = tok_jaccard_per_query[qid]["jaccard"] < tau

        overlap_trigger = cand_below or tok_below
        low_stability[qid] = overlap_trigger

        clean_m = clean_lex_pq.get(qid, {}).get(metric_name, 0.0)
        pert_m = pert_lex_pq.get(qid, {}).get(metric_name, 0.0)
        delta_m = pert_m - clean_m  # ΔM_SimulOnly = M(q̃) - M(q)
        delta_simulonly[qid] = delta_m

        collapsed[qid] = overlap_trigger and (delta_m <= -delta)

    n_collapsed = sum(collapsed.values())
    return {
        "collapsed": collapsed,
        "low_stability": low_stability,
        "delta_simulonly": delta_simulonly,
        "tau": tau,
        "delta": delta,
        "n_collapsed": n_collapsed,
        "collapse_rate": n_collapsed / len(common_qids) if common_qids else 0.0,
        "n_queries": len(common_qids),
    }


def plan_collapse_sensitivity(
    cand_overlap_per_query: Dict[str, Dict[str, float]],
    tok_jaccard_per_query: Optional[Dict[str, Dict[str, float]]],
    clean_lex_pq: Dict[str, Dict[str, float]],
    pert_lex_pq: Dict[str, Dict[str, float]],
    metric_name: str = "NDCG@10",
    tau_percentiles: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    Sensitivity ablation: sweep over (τ_percentile, δ) and report collapse rates.

    Returns a list of dicts, one per (τ_percentile, δ) pair, each containing:
      ``tau_percentile``, ``tau``, ``delta``, ``n_collapsed``, ``collapse_rate``,
      ``n_queries``.
    """
    if tau_percentiles is None:
        tau_percentiles = [5.0, 10.0, 15.0, 20.0, 25.0]
    if deltas is None:
        deltas = [0.01, 0.03, 0.05, 0.10]

    common_qids = sorted(
        set(cand_overlap_per_query) & set(clean_lex_pq) & set(pert_lex_pq)
    )
    cand_jaccards = [cand_overlap_per_query[q]["jaccard"] for q in common_qids]

    rows = []
    for tp in tau_percentiles:
        tau = float(np.percentile(cand_jaccards, tp)) if cand_jaccards else 0.0
        for d in deltas:
            info = classify_plan_collapse(
                cand_overlap_per_query=cand_overlap_per_query,
                tok_jaccard_per_query=tok_jaccard_per_query,
                clean_lex_pq=clean_lex_pq,
                pert_lex_pq=pert_lex_pq,
                metric_name=metric_name,
                tau=tau,
                delta=d,
            )
            rows.append({
                "tau_percentile": tp,
                "tau": info["tau"],
                "delta": d,
                "n_collapsed": info["n_collapsed"],
                "collapse_rate": info["collapse_rate"],
                "n_queries": info["n_queries"],
            })
    return rows


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


def _distribution_stats(values: List[float], prefix: str) -> Dict[str, float]:
    """Return mean, median, and tail quantiles (p10, p25, p75, p90) for *values*.

    Keys are named ``{prefix}_mean``, ``{prefix}_median``, ``{prefix}_p10``, etc.
    """
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
        }
    arr = np.asarray(valid, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }

"""
Planner diagnostic metrics for RQ3 cross-lingual evaluation.

Metrics computed:
1. Planner candidate recall@n: Is the relevant doc in D_n(q)?
2. Planner score quality (SimulOnly): Effectiveness of planner-only ranking
3. Prefix prior coverage: How often does gold doc's prefix receive meaningful bonus?
4. Cross-lingual token overlap: Token-plan similarity between English and target

These metrics directly measure how the lexical planner degrades under
cross-lingual mismatch, enabling the causal analysis:
    planner recall drops -> PAG gain shrinks
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Planner Candidate Recall
# ---------------------------------------------------------------------------

def planner_candidate_recall(
    lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    topk_values: List[int] = [10, 50, 100, 500, 1000],
) -> Dict[str, Dict[str, float]]:
    """
    Compute planner candidate recall at various cutoffs.

    For each query, check if the relevant document(s) are in the planner's
    top-K candidate set. This is the key diagnostic for RQ3: if the planner
    fails to retrieve relevant docs, PAG cannot succeed.

    Args:
        lex_run: Lexical planner output {qid: {docid: score}}
        qrels: Relevance judgments {qid: {docid: relevance}}
        topk_values: List of cutoff values to compute recall at

    Returns:
        Dict with per-query and aggregate recall at each cutoff
        {
            "per_query": {qid: {"recall@10": float, "recall@100": float, ...}},
            "aggregate": {"recall@10": float, "recall@100": float, ...},
        }
    """
    per_query = {}
    aggregate = {f"recall@{k}": [] for k in topk_values}

    for qid, docs in lex_run.items():
        if qid not in qrels:
            continue

        # Get relevant docs for this query
        relevant = set(
            did for did, rel in qrels[qid].items() if rel > 0
        )
        if not relevant:
            continue

        # Sort docs by score
        sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)

        per_query[qid] = {}
        for k in topk_values:
            topk_set = set(d for d, _ in sorted_docs[:k])
            retrieved_relevant = len(relevant & topk_set)
            recall = retrieved_relevant / len(relevant)
            per_query[qid][f"recall@{k}"] = recall
            aggregate[f"recall@{k}"].append(recall)

    # Compute aggregate means
    agg_means = {
        metric: np.mean(values) if values else 0.0
        for metric, values in aggregate.items()
    }

    return {
        "per_query": per_query,
        "aggregate": agg_means,
        "n_queries": len(per_query),
    }


def compare_planner_recall(
    english_lex_run: Dict[str, Dict[str, float]],
    target_lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    topk_values: List[int] = [10, 50, 100, 500, 1000],
) -> Dict:
    """
    Compare planner candidate recall between English and target language.

    This is the core diagnostic showing whether planner recall drops.
    """
    english_recall = planner_candidate_recall(english_lex_run, qrels, topk_values)
    target_recall = planner_candidate_recall(target_lex_run, qrels, topk_values)

    # Compute per-query deltas
    per_query_delta = {}
    common_qids = (
        set(english_recall["per_query"].keys()) &
        set(target_recall["per_query"].keys())
    )

    for qid in common_qids:
        per_query_delta[qid] = {}
        for k in topk_values:
            metric = f"recall@{k}"
            en_val = english_recall["per_query"][qid].get(metric, 0)
            tgt_val = target_recall["per_query"][qid].get(metric, 0)
            per_query_delta[qid][f"delta_{metric}"] = tgt_val - en_val

    # Compute aggregate deltas
    agg_delta = {}
    for k in topk_values:
        metric = f"recall@{k}"
        agg_delta[f"delta_{metric}"] = (
            target_recall["aggregate"].get(metric, 0) -
            english_recall["aggregate"].get(metric, 0)
        )

    return {
        "english": english_recall,
        "target": target_recall,
        "per_query_delta": per_query_delta,
        "aggregate_delta": agg_delta,
        "n_common_queries": len(common_qids),
    }


# ---------------------------------------------------------------------------
# SimulOnly Quality (Planner-only ranking effectiveness)
# ---------------------------------------------------------------------------

def simul_only_metrics(
    lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute retrieval metrics using only the lexical planner output.

    This measures the intrinsic quality of the planner's ranking,
    independent of the sequential decoder.
    """
    import pytrec_eval

    # Filter to queries in qrels
    filtered_run = {qid: docs for qid, docs in lex_run.items() if qid in qrels}

    str_run = {
        str(qid): {str(did): float(s) for did, s in docs.items()}
        for qid, docs in filtered_run.items()
    }
    str_qrels = {
        str(qid): {str(did): int(s) for did, s in docs.items()}
        for qid, docs in qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        str_qrels,
        {f"ndcg_cut_{k}", "recip_rank", f"recall_{100}"}
    )
    results = evaluator.evaluate(str_run)

    n = len(results)
    return {
        f"SimulOnly_NDCG@{k}": np.mean([v[f"ndcg_cut_{k}"] for v in results.values()]) if n else 0,
        f"SimulOnly_MRR@{k}": np.mean([v["recip_rank"] for v in results.values()]) if n else 0,
        "SimulOnly_Recall@100": np.mean([v[f"recall_{100}"] for v in results.values()]) if n else 0,
        "n_evaluated": n,
    }


def simul_only_per_query(
    lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute per-query SimulOnly metrics."""
    import pytrec_eval

    filtered_run = {qid: docs for qid, docs in lex_run.items() if qid in qrels}

    str_run = {
        str(qid): {str(did): float(s) for did, s in docs.items()}
        for qid, docs in filtered_run.items()
    }
    str_qrels = {
        str(qid): {str(did): int(s) for did, s in docs.items()}
        for qid, docs in qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        str_qrels,
        {f"ndcg_cut_{k}", "recip_rank"}
    )
    results = evaluator.evaluate(str_run)

    per_query = {}
    for qid, vals in results.items():
        per_query[qid] = {
            f"SimulOnly_NDCG@{k}": vals[f"ndcg_cut_{k}"],
            f"SimulOnly_MRR@{k}": vals["recip_rank"],
        }
    return per_query


def compare_simul_only(
    english_lex_run: Dict[str, Dict[str, float]],
    target_lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict:
    """
    Compare SimulOnly metrics between English and target language.
    """
    english_metrics = simul_only_metrics(english_lex_run, qrels, k)
    target_metrics = simul_only_metrics(target_lex_run, qrels, k)

    # Per-query comparison
    english_pq = simul_only_per_query(english_lex_run, qrels, k)
    target_pq = simul_only_per_query(target_lex_run, qrels, k)

    per_query_delta = {}
    common_qids = set(english_pq.keys()) & set(target_pq.keys())

    for qid in common_qids:
        per_query_delta[qid] = {}
        for metric in [f"SimulOnly_NDCG@{k}", f"SimulOnly_MRR@{k}"]:
            en_val = english_pq[qid].get(metric, 0)
            tgt_val = target_pq[qid].get(metric, 0)
            per_query_delta[qid][f"delta_{metric}"] = tgt_val - en_val

    # Aggregate deltas
    agg_delta = {}
    for metric in [f"SimulOnly_NDCG@{k}", f"SimulOnly_MRR@{k}", "SimulOnly_Recall@100"]:
        agg_delta[f"delta_{metric}"] = (
            target_metrics.get(metric, 0) - english_metrics.get(metric, 0)
        )

    return {
        "english": english_metrics,
        "target": target_metrics,
        "per_query_delta": per_query_delta,
        "aggregate_delta": agg_delta,
        "n_common_queries": len(common_qids),
    }


# ---------------------------------------------------------------------------
# Token-Plan Overlap (Cross-lingual)
# ---------------------------------------------------------------------------

def crosslingual_token_overlap(
    english_tokens: Dict[str, List[int]],
    target_tokens: Dict[str, List[int]],
    topk: int = 100,
) -> Dict:
    """
    Compute token-plan overlap between English and target language queries.

    This measures how similar the planner's vocabulary token selections are
    between the English query and its translation.

    Args:
        english_tokens: {qid: [token_ids]} from English queries
        target_tokens: {qid: [token_ids]} from target language queries
        topk: Number of top tokens to compare

    Returns:
        Per-query and aggregate overlap statistics
    """
    per_query = {}
    jaccards = []
    overlaps = []

    common_qids = set(english_tokens.keys()) & set(target_tokens.keys())

    for qid in common_qids:
        en_toks = set(english_tokens[qid][:topk])
        tgt_toks = set(target_tokens[qid][:topk])

        inter = en_toks & tgt_toks
        union = en_toks | tgt_toks

        jaccard = len(inter) / len(union) if union else 1.0
        overlap_at_k = len(inter) / topk if topk > 0 else 0.0

        per_query[qid] = {
            "jaccard": jaccard,
            "overlap_at_k": overlap_at_k,
            "intersection_size": len(inter),
            "english_size": len(en_toks),
            "target_size": len(tgt_toks),
        }
        jaccards.append(jaccard)
        overlaps.append(overlap_at_k)

    return {
        "per_query": per_query,
        "aggregate": {
            "mean_jaccard": np.mean(jaccards) if jaccards else 0.0,
            "median_jaccard": np.median(jaccards) if jaccards else 0.0,
            "mean_overlap_at_k": np.mean(overlaps) if overlaps else 0.0,
            "p10_jaccard": np.percentile(jaccards, 10) if jaccards else 0.0,
            "p25_jaccard": np.percentile(jaccards, 25) if jaccards else 0.0,
            "p75_jaccard": np.percentile(jaccards, 75) if jaccards else 0.0,
            "p90_jaccard": np.percentile(jaccards, 90) if jaccards else 0.0,
        },
        "n_queries": len(common_qids),
    }


# ---------------------------------------------------------------------------
# Prefix Prior Coverage (Optional)
# ---------------------------------------------------------------------------

def prefix_prior_coverage(
    lex_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    docid_to_tokenids: Dict[str, List[int]],
    threshold: float = 0.0,
) -> Dict:
    """
    Compute how often the gold document's prefix receives a meaningful bonus.

    For each relevant document, check if its sequential docid prefix
    appears in the planner's candidate set with score > threshold.

    Args:
        lex_run: Lexical planner output {qid: {docid: score}}
        qrels: Relevance judgments
        docid_to_tokenids: Mapping from docid to sequential token IDs
        threshold: Minimum score for "meaningful" bonus

    Returns:
        Coverage statistics
    """
    # Build prefix -> docids mapping (first token of sequential ID)
    prefix_to_docs = defaultdict(set)
    for docid, tokenids in docid_to_tokenids.items():
        if tokenids:
            prefix_to_docs[tokenids[0]].add(docid)

    per_query = {}
    coverage_rates = []

    for qid, docs in lex_run.items():
        if qid not in qrels:
            continue

        relevant = [did for did, rel in qrels[qid].items() if rel > 0]
        if not relevant:
            continue

        # Check coverage for each relevant doc
        covered = 0
        for rel_docid in relevant:
            if rel_docid not in docid_to_tokenids:
                continue

            tokenids = docid_to_tokenids[rel_docid]
            if not tokenids:
                continue

            prefix = tokenids[0]
            # Check if any doc with this prefix is in planner output
            prefix_docs = prefix_to_docs[prefix]
            for pdoc in prefix_docs:
                if pdoc in docs and docs[pdoc] > threshold:
                    covered += 1
                    break

        coverage = covered / len(relevant)
        per_query[qid] = {
            "coverage": coverage,
            "n_relevant": len(relevant),
            "n_covered": covered,
        }
        coverage_rates.append(coverage)

    return {
        "per_query": per_query,
        "aggregate": {
            "mean_coverage": np.mean(coverage_rates) if coverage_rates else 0.0,
            "full_coverage_rate": np.mean([1 if c == 1.0 else 0 for c in coverage_rates]) if coverage_rates else 0.0,
        },
        "n_queries": len(per_query),
    }


# ---------------------------------------------------------------------------
# PAG Gain Analysis
# ---------------------------------------------------------------------------

def pag_gain(
    lex_metrics: Dict[str, float],
    smt_metrics: Dict[str, float],
    metric_names: List[str] = ["NDCG@10", "MRR@10"],
) -> Dict[str, float]:
    """
    Compute PAG gain: improvement of Stage 2 over Stage 1.

    PAG_Gain = SMT_metric - LEX_metric

    Positive gain means the sequential decoder improves over planner-only.
    """
    gains = {}
    for metric in metric_names:
        lex_val = lex_metrics.get(metric, 0)
        smt_val = smt_metrics.get(metric, 0)
        gains[f"PAG_Gain_{metric}"] = smt_val - lex_val
    return gains


def compare_pag_gain(
    english_lex: Dict[str, float],
    english_smt: Dict[str, float],
    target_lex: Dict[str, float],
    target_smt: Dict[str, float],
) -> Dict:
    """
    Compare PAG gain between English and target language.

    Tests whether PAG gain shrinks when planner recall drops.
    """
    english_gain = pag_gain(english_lex, english_smt)
    target_gain = pag_gain(target_lex, target_smt)

    delta = {}
    for metric in english_gain:
        delta[f"delta_{metric}"] = target_gain.get(metric, 0) - english_gain.get(metric, 0)

    return {
        "english_gain": english_gain,
        "target_gain": target_gain,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Plan Swap Impact
# ---------------------------------------------------------------------------

def plan_swap_impact(
    naive_smt_metrics: Dict[str, float],
    swap_smt_metrics: Dict[str, float],
    metric_names: List[str] = ["NDCG@10", "MRR@10"],
) -> Dict[str, float]:
    """
    Compute plan swap impact: how much using English plan helps.

    PlanSwap_Gain = swap_metric - naive_metric

    Positive gain means using English plan improves over cross-lingual plan.
    This is direct evidence that planner mismatch is the bottleneck.
    """
    impact = {}
    for metric in metric_names:
        naive_val = naive_smt_metrics.get(metric, 0)
        swap_val = swap_smt_metrics.get(metric, 0)
        impact[f"PlanSwap_Gain_{metric}"] = swap_val - naive_val
    return impact


# ---------------------------------------------------------------------------
# Aggregate all diagnostics
# ---------------------------------------------------------------------------

def aggregate_planner_diagnostics(
    english_lex_run: Dict[str, Dict[str, float]],
    target_lex_run: Dict[str, Dict[str, float]],
    english_smt_run: Dict[str, Dict[str, float]],
    target_smt_run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    english_tokens: Optional[Dict[str, List[int]]] = None,
    target_tokens: Optional[Dict[str, List[int]]] = None,
    swap_smt_run: Optional[Dict[str, Dict[str, float]]] = None,
    k: int = 10,
) -> Dict:
    """
    Compute all planner diagnostic metrics for cross-lingual analysis.

    Returns a comprehensive dict with:
    - Planner candidate recall comparison
    - SimulOnly quality comparison
    - Token overlap (if tokens provided)
    - PAG gain comparison
    - Plan swap impact (if swap run provided)
    """
    results = {}

    # 1. Planner candidate recall
    results["candidate_recall"] = compare_planner_recall(
        english_lex_run, target_lex_run, qrels
    )

    # 2. SimulOnly quality
    results["simul_only"] = compare_simul_only(
        english_lex_run, target_lex_run, qrels, k
    )

    # 3. Token overlap
    if english_tokens and target_tokens:
        results["token_overlap"] = crosslingual_token_overlap(
            english_tokens, target_tokens
        )

    # 4. PAG gain
    from robustness.metrics.plan_collapse import compute_retrieval_metrics

    english_lex_metrics = compute_retrieval_metrics(english_lex_run, qrels, k)
    english_smt_metrics = compute_retrieval_metrics(english_smt_run, qrels, k)
    target_lex_metrics = compute_retrieval_metrics(target_lex_run, qrels, k)
    target_smt_metrics = compute_retrieval_metrics(target_smt_run, qrels, k)

    results["pag_gain"] = compare_pag_gain(
        english_lex_metrics, english_smt_metrics,
        target_lex_metrics, target_smt_metrics,
    )

    # 5. Plan swap impact
    if swap_smt_run:
        swap_metrics = compute_retrieval_metrics(swap_smt_run, qrels, k)
        results["plan_swap_impact"] = plan_swap_impact(
            target_smt_metrics, swap_metrics
        )

    # 6. Store raw metrics for reference
    results["english_lex_metrics"] = english_lex_metrics
    results["english_smt_metrics"] = english_smt_metrics
    results["target_lex_metrics"] = target_lex_metrics
    results["target_smt_metrics"] = target_smt_metrics
    if swap_smt_run:
        results["swap_smt_metrics"] = swap_metrics

    return results

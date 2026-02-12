#!/usr/bin/env python3
"""
Evaluate the adapted planner checkpoint alongside inference-only baselines.

Evaluation protocol MATCHES the inference-only baselines exactly:
  - Same dev queries (mMARCO dev ∩ MS MARCO qrels)
  - Same qrels
  - Same metrics: MRR@10, nDCG@10, Recall@10
  - Same TREC output format
  - Same latency measurement

Systems compared:
  1. naive           - English PAG on non-English queries (original checkpoint)
  2. sequential-only - unconstrained beam search (original checkpoint)
  3. translate       - translate query to English then run PAG (original checkpoint)
  4. adapted-planner - adapted checkpoint on non-English queries (NO translation)

Additionally reports planner alignment diagnostics:
  - TokJaccard@100 between q_en and q_lang planner tokens (same qid)
  - CandOverlap@100 (optional)

Usage:
  python -m cross_lingual.trained_extension.evaluate_adapted \\
      --language fr \\
      --adapted_checkpoint cross_lingual/trained_extension/checkpoints/fr/best

  python -m cross_lingual.trained_extension.evaluate_adapted \\
      --language all --significance
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robustness.utils.pag_inference import (
    PRETRAINED_PATH,
    load_planner_tokens,
    load_planner_tokens_with_scores,
)
from cross_lingual.data_loader import (
    RQ3_LANGUAGES,
    prepare_query_files,
    save_filtered_qrels,
    get_evaluation_qids,
)
from cross_lingual.retrieval_engine import (
    run_stage1,
    run_stage2,
    run_json_to_trec,
    count_queries,
    load_lexical_run,
    load_sequential_run,
    _get_dataset_name,
)
from cross_lingual.evaluate import (
    compute_metrics,
    compute_per_query_metrics,
    save_metrics,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# Planner alignment diagnostics
# ---------------------------------------------------------------------------

def compute_tok_jaccard(
    tokens_en: Dict[str, Dict],
    tokens_lang: Dict[str, Dict],
    topk: int = 100,
) -> Dict[str, float]:
    """Per-query TokJaccard@K between English and non-English planner tokens.

    Uses the same definition as RQ2 plan_intersect (robustness/metrics/plan_collapse.py).

    Args:
        tokens_en: {qid: {"token_ids": [...], "scores": [...]}} from teacher on q_en
        tokens_lang: same structure from student (or teacher) on q_lang

    Returns:
        {qid: jaccard_score}
    """
    common = set(tokens_en.keys()) & set(tokens_lang.keys())
    results = {}
    for qid in common:
        en_entry = tokens_en[qid]
        lang_entry = tokens_lang[qid]

        # Extract token IDs (handle both dict and list formats)
        if isinstance(en_entry, dict) and "token_ids" in en_entry:
            en_ids = en_entry["token_ids"][:topk]
        elif isinstance(en_entry, list):
            en_ids = en_entry[:topk]
        else:
            continue

        if isinstance(lang_entry, dict) and "token_ids" in lang_entry:
            lang_ids = lang_entry["token_ids"][:topk]
        elif isinstance(lang_entry, list):
            lang_ids = lang_entry[:topk]
        else:
            continue

        en_set = set(en_ids)
        lang_set = set(lang_ids)
        inter = en_set & lang_set
        union = en_set | lang_set
        results[qid] = len(inter) / len(union) if union else 1.0

    return results


def compute_cand_overlap(
    run_en: Dict[str, Dict[str, float]],
    run_lang: Dict[str, Dict[str, float]],
    topk: int = 100,
) -> Dict[str, float]:
    """Per-query CandOverlap@K between English and non-English Stage 1 results.

    Jaccard of top-K candidate document sets.

    Args:
        run_en: {qid: {docid: score}} Stage 1 run for English queries
        run_lang: same for non-English queries

    Returns:
        {qid: overlap_score}
    """
    common = set(run_en.keys()) & set(run_lang.keys())
    results = {}
    for qid in common:
        en_docs = run_en[qid]
        lang_docs = run_lang[qid]

        en_topk = set(sorted(en_docs, key=en_docs.get, reverse=True)[:topk])
        lang_topk = set(sorted(lang_docs, key=lang_docs.get, reverse=True)[:topk])

        inter = en_topk & lang_topk
        union = en_topk | lang_topk
        results[qid] = len(inter) / len(union) if union else 1.0

    return results


def summary_stats(values: Dict[str, float]) -> Dict[str, float]:
    """Compute summary statistics: mean, median, p10, p25, p75, p90."""
    if not values:
        return {}
    arr = np.array(list(values.values()))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# Run adapted-planner evaluation
# ---------------------------------------------------------------------------

def evaluate_adapted_planner(
    language: str,
    adapted_checkpoint: str,
    output_dir: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
    skip_existing: bool = True,
    compute_diagnostics: bool = True,
) -> Dict:
    """Run evaluation for the adapted-planner system.

    Protocol matches inference-only baselines exactly.

    Returns dict with metrics, timing, and diagnostics.
    """
    out = Path(output_dir) / language / "adapted"
    out.mkdir(parents=True, exist_ok=True)

    # Prepare data (same query set as baselines)
    queries_dir = Path(output_dir) / "queries"
    en_dir, tgt_dir, n_queries, coverage = prepare_query_files(
        language, queries_dir, download=True,
    )

    eval_qids = set(coverage["intersection_qids"])
    qrel_path = out / "qrels.json"
    save_filtered_qrels(eval_qids, qrel_path)

    dataset_name = _get_dataset_name(str(tgt_dir))
    result = {
        "baseline": "adapted",
        "language": language,
        "n_queries": n_queries,
        "adapted_checkpoint": adapted_checkpoint,
    }

    lex_out_dir = str(out / "lex_ret")
    smt_out_dir = str(out / "smt_ret")

    # --- Stage 1 (planner / SimulOnly) using ADAPTED checkpoint ---
    # Override pretrained path for adapted model
    lex_run_path = os.path.join(lex_out_dir, dataset_name, "run.json")
    if os.path.exists(lex_run_path) and skip_existing:
        print(f"[adapted] Stage 1 exists, skipping")
        result["stage1_time_s"] = 0
    else:
        rc1, t1 = _run_stage1_with_checkpoint(
            str(tgt_dir), lex_out_dir, adapted_checkpoint,
            lex_topk, batch_size,
        )
        result["stage1_time_s"] = t1
        if rc1 != 0:
            result["error"] = f"stage1_failed_rc={rc1}"
            _save_result(result, out)
            return result

    # --- Stage 2 (full PAG) using ADAPTED checkpoint ---
    smt_run_path = os.path.join(smt_out_dir, dataset_name, "run.json")
    if os.path.exists(smt_run_path) and skip_existing:
        print(f"[adapted] Stage 2 exists, skipping")
        result["stage2_time_s"] = 0
    else:
        rc2, t2 = _run_stage2_with_checkpoint(
            str(tgt_dir), smt_out_dir, lex_out_dir,
            adapted_checkpoint,
            smt_topk, batch_size * 2, n_gpu,
        )
        result["stage2_time_s"] = t2
        if rc2 != 0:
            result["error"] = f"stage2_failed_rc={rc2}"
            _save_result(result, out)
            return result

    # --- Merge & evaluate ---
    from robustness.utils.pag_inference import merge_and_evaluate
    merge_and_evaluate(str(tgt_dir), smt_out_dir, str(qrel_path))

    # Load runs and save TREC files
    lex_run = load_lexical_run(lex_out_dir, dataset_name)
    smt_run = load_sequential_run(smt_out_dir, dataset_name)

    if lex_run:
        run_json_to_trec(lex_run, str(out / "stage1_run.trec"),
                         f"adapted_{language}_s1")

    if smt_run:
        run_json_to_trec(smt_run, str(out / "stage2_run.trec"),
                         f"adapted_{language}_s2")

    # Compute retrieval metrics
    with open(qrel_path) as f:
        qrels = json.load(f)

    total_time = result.get("stage1_time_s", 0) + result.get("stage2_time_s", 0)

    if lex_run:
        s1_metrics = compute_metrics(lex_run, qrels, n_queries, total_time)
        result["stage1_metrics"] = s1_metrics
        save_metrics(s1_metrics, out / "stage1_metrics.json")

    if smt_run:
        s2_metrics = compute_metrics(smt_run, qrels, n_queries, total_time)
        result["stage2_metrics"] = s2_metrics
        save_metrics(s2_metrics, out / "stage2_metrics.json")

    # --- Planner alignment diagnostics ---
    if compute_diagnostics:
        diag = _compute_planner_diagnostics(
            language, en_dir, tgt_dir,
            adapted_checkpoint, lex_out_dir,
            out, batch_size,
        )
        result["diagnostics"] = diag
        save_metrics(diag, out / "planner_diagnostics.json")

    _save_result(result, out)
    return result


def _save_result(result: Dict, out_dir: Path):
    """Save full result dict."""
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Run stages with custom checkpoint
# ---------------------------------------------------------------------------

def _run_stage1_with_checkpoint(
    query_dir: str,
    lex_out_dir: str,
    pretrained_path: str,
    topk: int = 1000,
    batch_size: int = 8,
) -> Tuple[int, float]:
    """Run Stage 1 with a specific checkpoint (adapted model).

    Matches robustness.utils.pag_inference.run_lexical_retrieval exactly.
    """
    from robustness.utils.pag_inference import _run_cmd, LEX_DOCID_PATH

    os.makedirs(lex_out_dir, exist_ok=True)
    q_collection_paths = json.dumps([query_dir])

    eval_qrel_json = json.dumps([None])
    cmd = [
        sys.executable, "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={pretrained_path}",
        f"--out_dir={lex_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--topk={topk}",
        f"--lex_docid_to_smtid_path={LEX_DOCID_PATH}",
        "--max_length=128",
        f"--eval_qrel_path={eval_qrel_json}",
    ]

    t0 = time.time()
    rc = _run_cmd(cmd)
    elapsed = time.time() - t0
    return rc, elapsed


def _run_stage2_with_checkpoint(
    query_dir: str,
    smt_out_dir: str,
    lex_out_dir: str,
    pretrained_path: str,
    topk: int = 100,
    batch_size: int = 16,
    n_gpu: int = 1,
) -> Tuple[int, float]:
    """Run Stage 2 with a specific checkpoint (adapted model).

    Matches robustness.utils.pag_inference.run_sequential_decoding exactly.
    """
    from robustness.utils.pag_inference import (
        _run_cmd, LEX_DOCID_PATH, SMT_DOCID_PATH,
    )

    os.makedirs(smt_out_dir, exist_ok=True)
    q_collection_paths = json.dumps([query_dir])
    eval_qrel_json = json.dumps([None])
    master_port = os.environ.get("MASTER_PORT", "29500")

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={n_gpu}",
        f"--master_port={master_port}",
        "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={pretrained_path}",
        f"--out_dir={smt_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank_2",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--topk={topk}",
        f"--lex_docid_to_smtid_path={LEX_DOCID_PATH}",
        f"--smt_docid_to_smtid_path={SMT_DOCID_PATH}",
        "--max_length=128",
        "--max_new_token_for_docid=8",
        "--lex_constrained=lexical_tmp_rescore",
        f"--eval_qrel_path={eval_qrel_json}",
    ]

    t0 = time.time()
    rc = _run_cmd(cmd)
    elapsed = time.time() - t0
    return rc, elapsed


# ---------------------------------------------------------------------------
# Planner diagnostics
# ---------------------------------------------------------------------------

def _compute_planner_diagnostics(
    language: str,
    en_dir: Path,
    tgt_dir: Path,
    adapted_checkpoint: str,
    adapted_lex_out_dir: str,
    out_dir: Path,
    batch_size: int = 8,
) -> Dict:
    """Compute TokJaccard@100 and CandOverlap@100 diagnostics.

    Compares planner tokens from:
      - Teacher (original checkpoint) on English queries
      - Adapted student on non-English queries
    """
    from cross_lingual.trained_extension.extract_planner_tokens import (
        extract_planner_tokens_file,
    )

    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Extract teacher tokens on English queries
    en_tokens_path = str(diag_dir / "teacher_en_tokens.json")
    if not os.path.exists(en_tokens_path):
        print(f"[diag] Extracting teacher planner tokens for English queries...")
        extract_planner_tokens_file(
            query_dir=str(en_dir),
            out_path=en_tokens_path,
            pretrained_path=str(PRETRAINED_PATH),
            topk=100,
            batch_size=batch_size,
        )

    # Extract adapted student tokens on non-English queries
    lang_tokens_path = str(diag_dir / f"adapted_{language}_tokens.json")
    if not os.path.exists(lang_tokens_path):
        print(f"[diag] Extracting adapted planner tokens for {language} queries...")
        extract_planner_tokens_file(
            query_dir=str(tgt_dir),
            out_path=lang_tokens_path,
            pretrained_path=adapted_checkpoint,
            topk=100,
            batch_size=batch_size,
        )

    # Also extract original (non-adapted) tokens on non-English queries for comparison
    orig_tokens_path = str(diag_dir / f"original_{language}_tokens.json")
    if not os.path.exists(orig_tokens_path):
        print(f"[diag] Extracting original planner tokens for {language} queries...")
        extract_planner_tokens_file(
            query_dir=str(tgt_dir),
            out_path=orig_tokens_path,
            pretrained_path=str(PRETRAINED_PATH),
            topk=100,
            batch_size=batch_size,
        )

    # Load tokens
    en_tokens = load_planner_tokens_with_scores(en_tokens_path)
    adapted_tokens = load_planner_tokens_with_scores(lang_tokens_path)
    original_tokens = load_planner_tokens_with_scores(orig_tokens_path)

    # TokJaccard: adapted vs teacher-English
    tj_adapted = compute_tok_jaccard(en_tokens, adapted_tokens, topk=100)
    tj_adapted_stats = summary_stats(tj_adapted)

    # TokJaccard: original vs teacher-English (baseline comparison)
    tj_original = compute_tok_jaccard(en_tokens, original_tokens, topk=100)
    tj_original_stats = summary_stats(tj_original)

    diagnostics = {
        "tok_jaccard_adapted_vs_en": tj_adapted_stats,
        "tok_jaccard_original_vs_en": tj_original_stats,
    }

    # CandOverlap@100 from Stage 1 runs (if available)
    dataset_name = _get_dataset_name(str(tgt_dir))
    adapted_lex_run = load_lexical_run(adapted_lex_out_dir, dataset_name)

    # Load naive baseline lex run for comparison (if exists)
    naive_lex_dir = str(Path(out_dir).parent / "naive" / "lex_ret")
    naive_lex_run = load_lexical_run(naive_lex_dir, dataset_name)

    # English lex run (need to run Stage 1 on English queries)
    en_lex_dir = str(diag_dir / "en_lex_ret")
    en_lex_run_path = os.path.join(en_lex_dir, dataset_name, "run.json")
    if os.path.exists(en_lex_run_path):
        en_lex_run = load_lexical_run(en_lex_dir, _get_dataset_name(str(en_dir)))
    else:
        en_lex_run = None

    if adapted_lex_run and en_lex_run:
        co_adapted = compute_cand_overlap(en_lex_run, adapted_lex_run, topk=100)
        diagnostics["cand_overlap_adapted_vs_en"] = summary_stats(co_adapted)

    if naive_lex_run and en_lex_run:
        co_naive = compute_cand_overlap(en_lex_run, naive_lex_run, topk=100)
        diagnostics["cand_overlap_original_vs_en"] = summary_stats(co_naive)

    return diagnostics


# ---------------------------------------------------------------------------
# Aggregate all systems and produce summary CSV
# ---------------------------------------------------------------------------

def aggregate_all_systems(
    results_dir: str,
    languages: Optional[List[str]] = None,
    run_significance: bool = False,
) -> Dict:
    """Aggregate results from all systems across languages.

    Returns combined results dict.
    """
    if languages is None:
        languages = RQ3_LANGUAGES

    baselines = ["naive", "sequential", "translate", "adapted"]
    all_results = {}

    for lang in languages:
        lang_results = {}
        for bl in baselines:
            result_path = Path(results_dir) / lang / bl / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    lang_results[bl] = json.load(f)
            else:
                lang_results[bl] = {"baseline": bl, "language": lang, "missing": True}
        all_results[lang] = lang_results

    return all_results


def print_comparison_table(all_results: Dict):
    """Print a formatted comparison table including adapted-planner."""
    header = (
        f"{'Lang':>4s} {'System':>15s} | "
        f"{'MRR@10':>8s} {'Recall@10':>10s} {'nDCG@10':>8s} | "
        f"{'TJ@100':>8s}"
    )
    print(f"\n{'=' * len(header)}")
    print("RQ3 Cross-lingual Results (with Adapted Planner)")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for lang, lang_results in sorted(all_results.items()):
        for bl_name in ["naive", "sequential", "translate", "adapted"]:
            entry = lang_results.get(bl_name, {})
            if entry.get("missing"):
                print(f"{lang:>4s} {bl_name:>15s} | {'(missing)':>30s} |")
                continue

            s2 = entry.get("stage2_metrics", entry.get("stage2", {}))
            diag = entry.get("diagnostics", {})
            tj = diag.get("tok_jaccard_adapted_vs_en", diag.get("tok_jaccard_original_vs_en", {}))
            tj_mean = tj.get("mean", 0.0) if tj else 0.0

            print(
                f"{lang:>4s} {bl_name:>15s} | "
                f"{s2.get('MRR@10', 0)*100:>7.2f}% "
                f"{s2.get('Recall@10', 0)*100:>9.2f}% "
                f"{s2.get('nDCG@10', 0)*100:>7.2f}% | "
                f"{tj_mean*100:>7.2f}%"
            )
        print(f"{'-' * len(header)}")


def save_summary_csv(all_results: Dict, output_path: str):
    """Save summary_rq3_trained.csv."""
    rows = []
    for lang, lang_results in sorted(all_results.items()):
        for bl_name in ["naive", "sequential", "translate", "adapted"]:
            entry = lang_results.get(bl_name, {})
            if entry.get("missing"):
                continue

            s2 = entry.get("stage2_metrics", entry.get("stage2", {}))
            s1 = entry.get("stage1_metrics", entry.get("stage1", {}))
            diag = entry.get("diagnostics", {})

            # Pick the right TokJaccard key
            if bl_name == "adapted":
                tj_stats = diag.get("tok_jaccard_adapted_vs_en", {})
            else:
                tj_stats = diag.get("tok_jaccard_original_vs_en", {})

            rows.append({
                "language": lang,
                "system": bl_name,
                "MRR@10": f"{s2.get('MRR@10', 0):.4f}",
                "Recall@10": f"{s2.get('Recall@10', 0):.4f}",
                "nDCG@10": f"{s2.get('nDCG@10', 0):.4f}",
                "S1_MRR@10": f"{s1.get('MRR@10', 0):.4f}" if s1 else "",
                "TokJaccard@100_mean": f"{tj_stats.get('mean', 0):.4f}" if tj_stats else "",
                "TokJaccard@100_median": f"{tj_stats.get('median', 0):.4f}" if tj_stats else "",
                "avg_latency_ms": f"{s2.get('avg_latency_ms_per_query', 0):.1f}",
                "error_rate": f"{s2.get('error_rate', 0):.4f}",
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "language", "system", "MRR@10", "Recall@10", "nDCG@10",
        "S1_MRR@10", "TokJaccard@100_mean", "TokJaccard@100_median",
        "avg_latency_ms", "error_rate",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Summary CSV -> {output_path}")


# ---------------------------------------------------------------------------
# Significance tests: adapted vs naive
# ---------------------------------------------------------------------------

def run_significance_vs_naive(
    results_dir: str,
    languages: Optional[List[str]] = None,
) -> List[Dict]:
    """Run paired bootstrap tests: adapted vs naive."""
    if languages is None:
        languages = RQ3_LANGUAGES

    sig_results = []

    for lang in languages:
        naive_result = Path(results_dir) / lang / "naive" / "result.json"
        adapted_result = Path(results_dir) / lang / "adapted" / "result.json"

        if not naive_result.exists() or not adapted_result.exists():
            print(f"[sig] Skipping {lang}: missing results")
            continue

        # Load qrels
        qrel_path = Path(results_dir) / lang / "adapted" / "qrels.json"
        if not qrel_path.exists():
            continue
        with open(qrel_path) as f:
            qrels = json.load(f)

        # Load runs
        for stage, stage_label in [("smt_ret", "stage2"), ("lex_ret", "stage1")]:
            naive_dir = Path(results_dir) / lang / "naive" / stage
            adapted_dir = Path(results_dir) / lang / "adapted" / stage

            naive_run = _find_and_load_run(naive_dir)
            adapted_run = _find_and_load_run(adapted_dir)

            if not naive_run or not adapted_run:
                continue

            naive_pq = compute_per_query_metrics(naive_run, qrels)
            adapted_pq = compute_per_query_metrics(adapted_run, qrels)

            for metric in ["MRR@10", "nDCG@10", "Recall@10"]:
                vals_naive = {q: v[metric] for q, v in naive_pq.items()}
                vals_adapted = {q: v[metric] for q, v in adapted_pq.items()}

                test = paired_bootstrap_test(vals_adapted, vals_naive, metric)
                test["language"] = lang
                test["stage"] = stage_label
                test["system_a"] = "adapted"
                test["system_b"] = "naive"
                sig_results.append(test)

                sig = "*" if test.get("significant_at_05") else ""
                print(
                    f"  {lang} {stage_label} adapted vs naive ({metric}): "
                    f"delta={test.get('delta', 0):.4f}, "
                    f"p={test.get('p_value', 1):.4f}{sig}"
                )

    # Save
    sig_path = Path(results_dir) / "significance_adapted_vs_naive.json"
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2)
    print(f"[sig] Results -> {sig_path}")

    return sig_results


def _find_and_load_run(run_dir: Path) -> Optional[Dict]:
    """Find and load run.json from a stage output directory."""
    if not run_dir.exists():
        return None
    for ds_name in ["MSMARCO", "TREC_DL_2019", "TREC_DL_2020", "other_dataset"]:
        run_path = run_dir / ds_name / "run.json"
        if run_path.exists():
            with open(run_path) as f:
                return json.load(f)
    # Try any subdirectory
    for sub in run_dir.iterdir():
        if sub.is_dir():
            run_path = sub / "run.json"
            if run_path.exists():
                with open(run_path) as f:
                    return json.load(f)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate adapted planner alongside baselines"
    )
    parser.add_argument("--language", type=str, required=True,
                        help=f"Language: {RQ3_LANGUAGES} or 'all'")
    parser.add_argument("--adapted_checkpoint", type=str, default=None,
                        help="Path to adapted checkpoint (default: auto-detect from results)")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "cross_lingual" / "results"))
    parser.add_argument("--lex_topk", type=int, default=1000)
    parser.add_argument("--smt_topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--significance", action="store_true",
                        help="Run paired bootstrap significance tests")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Output path for summary CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = RQ3_LANGUAGES if args.language == "all" else [args.language]

    ckpt_base = Path(REPO_ROOT / "cross_lingual" / "trained_extension" / "checkpoints")

    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating adapted planner for {lang}")
        print(f"{'=' * 60}")

        # Auto-detect checkpoint if not specified
        ckpt = args.adapted_checkpoint
        if ckpt is None:
            best_ckpt = ckpt_base / lang / "best"
            final_ckpt = ckpt_base / lang / "final"
            if best_ckpt.exists():
                ckpt = str(best_ckpt)
            elif final_ckpt.exists():
                ckpt = str(final_ckpt)
            else:
                print(f"[eval] No checkpoint found for {lang}, skipping")
                continue

        print(f"[eval] Using checkpoint: {ckpt}")

        result = evaluate_adapted_planner(
            language=lang,
            adapted_checkpoint=ckpt,
            output_dir=args.output_dir,
            lex_topk=args.lex_topk,
            smt_topk=args.smt_topk,
            batch_size=args.batch_size,
            n_gpu=args.n_gpu,
            skip_existing=not args.force,
        )

        s2 = result.get("stage2_metrics", {})
        print(f"[eval] {lang} adapted: "
              f"MRR@10={s2.get('MRR@10', 0)*100:.2f}% "
              f"R@10={s2.get('Recall@10', 0)*100:.2f}%")

    # Aggregate all systems
    all_results = aggregate_all_systems(args.output_dir, languages)
    print_comparison_table(all_results)

    # Summary CSV
    csv_path = args.summary_csv or str(
        Path(args.output_dir) / "summary_rq3_trained.csv"
    )
    save_summary_csv(all_results, csv_path)

    # Significance tests
    if args.significance:
        print("\n[eval] Running significance tests...")
        run_significance_vs_naive(args.output_dir, languages)


if __name__ == "__main__":
    main()

"""
PAG inference wrapper for cross-lingual evaluation (RQ3).

Extends the robustness.utils.pag_inference module with additional modes:
- Naive cross-lingual PAG (System A): non-English query, standard PAG
- Sequential-only (System B): non-English query, planner disabled
- Translate-at-inference (System C): translated query, standard PAG

Also provides plan-swapped decoding for cross-lingual causal probes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import from existing robustness module
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robustness.utils.pag_inference import (
    PRETRAINED_PATH,
    LEX_DOCID_PATH,
    SMT_DOCID_PATH,
    run_lexical_retrieval,
    run_sequential_decoding,
    merge_and_evaluate,
    extract_planner_tokens,
    load_lexical_run,
    load_sequential_run,
    load_planner_tokens,
    load_planner_tokens_with_scores,
    _run_cmd,
    _get_dataset_name,
)


# ---------------------------------------------------------------------------
# System A: Naive Cross-lingual PAG (standard pipeline)
# ---------------------------------------------------------------------------

def run_naive_crosslingual_pag(
    query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
) -> Dict:
    """
    System A: Run standard PAG on non-English queries.

    This is the "naive" baseline that tests how PAG performs when
    queries are non-English but documents/docids are English.
    The planner is expected to underperform due to lexical mismatch.
    """
    lex_out_dir = os.path.join(output_dir, "naive_pag", "lex_ret")
    smt_out_dir = os.path.join(output_dir, "naive_pag", "smt_ret")
    os.makedirs(lex_out_dir, exist_ok=True)
    os.makedirs(smt_out_dir, exist_ok=True)

    # Stage 1: Lexical retrieval (planner)
    rc = run_lexical_retrieval(
        query_dir=query_dir,
        lex_out_dir=lex_out_dir,
        topk=lex_topk,
        batch_size=batch_size,
    )
    if rc != 0:
        return {"_error": "stage1_failed", "_return_code": rc}

    # Stage 2: Sequential decoding with lexical constraint
    rc = run_sequential_decoding(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        lex_out_dir=lex_out_dir,
        topk=smt_topk,
        batch_size=batch_size * 2,
        n_gpu=n_gpu,
    )
    if rc != 0:
        return {"_error": "stage2_failed", "_return_code": rc}

    # Stage 3: Merge and evaluate
    results = merge_and_evaluate(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        eval_qrel_path=eval_qrel_path,
    )
    return results


# ---------------------------------------------------------------------------
# System B: Sequential-only (planner disabled)
# ---------------------------------------------------------------------------

def run_sequential_only(
    query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    topk: int = 100,
    batch_size: int = 16,
    n_gpu: int = 1,
) -> Dict:
    """
    System B: Run sequential decoder without lexical planner constraint.

    This baseline disables the planner entirely by:
    1. Running lexical retrieval but ignoring its scores
    2. Running sequential decoding with lex_constrained="none"

    This isolates whether the planner is helping or hurting in cross-lingual.
    """
    smt_out_dir = os.path.join(output_dir, "seq_only", "smt_ret")
    os.makedirs(smt_out_dir, exist_ok=True)

    # We still need a lex_out_dir for the pipeline, but we use "none" constraint
    # Create a dummy lexical output that returns all documents
    lex_dummy_dir = os.path.join(output_dir, "seq_only", "lex_dummy")
    os.makedirs(lex_dummy_dir, exist_ok=True)

    # Run sequential decoding without lexical constraint
    rc = _run_sequential_unconstrained(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        topk=topk,
        batch_size=batch_size,
        n_gpu=n_gpu,
    )
    if rc != 0:
        return {"_error": "sequential_failed", "_return_code": rc}

    results = merge_and_evaluate(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        eval_qrel_path=eval_qrel_path,
    )
    return results


def _run_sequential_unconstrained(
    query_dir: str,
    smt_out_dir: str,
    topk: int = 100,
    max_new_token: int = 8,
    batch_size: int = 16,
    max_length: int = 128,
    n_gpu: int = 1,
) -> int:
    """
    Run sequential decoding without lexical constraint.

    Uses task=constrained_beam_search_for_qid_rankdata which does
    standard beam search without planner guidance.
    """
    q_collection_paths = json.dumps([query_dir])

    master_port = os.environ.get("MASTER_PORT", "29500")
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={n_gpu}",
        f"--master_port={master_port}",
        "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={PRETRAINED_PATH}",
        f"--out_dir={smt_out_dir}",
        "--task=constrained_beam_search_for_qid_rankdata",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--topk={topk}",
        f"--smt_docid_to_smtid_path={SMT_DOCID_PATH}",
        f"--max_length={max_length}",
        f"--max_new_token_for_docid={max_new_token}",
    ]
    return _run_cmd(cmd)


# ---------------------------------------------------------------------------
# System C: Translate-at-inference PAG
# ---------------------------------------------------------------------------

def run_translate_at_inference_pag(
    original_query_dir: str,
    translated_query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
) -> Dict:
    """
    System C: Run PAG on machine-translated English queries.

    This baseline:
    1. Takes non-English queries translated back to English
    2. Runs the standard PAG pipeline on translated queries
    3. Evaluates using original qrels

    This tests whether "restoring overlap" via translation restores planner usefulness.
    """
    lex_out_dir = os.path.join(output_dir, "translate_pag", "lex_ret")
    smt_out_dir = os.path.join(output_dir, "translate_pag", "smt_ret")
    os.makedirs(lex_out_dir, exist_ok=True)
    os.makedirs(smt_out_dir, exist_ok=True)

    # Stage 1: Lexical retrieval on translated queries
    rc = run_lexical_retrieval(
        query_dir=translated_query_dir,
        lex_out_dir=lex_out_dir,
        topk=lex_topk,
        batch_size=batch_size,
    )
    if rc != 0:
        return {"_error": "stage1_failed", "_return_code": rc}

    # Stage 2: Sequential decoding with lexical constraint
    rc = run_sequential_decoding(
        query_dir=translated_query_dir,
        smt_out_dir=smt_out_dir,
        lex_out_dir=lex_out_dir,
        topk=smt_topk,
        batch_size=batch_size * 2,
        n_gpu=n_gpu,
    )
    if rc != 0:
        return {"_error": "stage2_failed", "_return_code": rc}

    # Stage 3: Merge and evaluate
    results = merge_and_evaluate(
        query_dir=translated_query_dir,
        smt_out_dir=smt_out_dir,
        eval_qrel_path=eval_qrel_path,
    )
    return results


# ---------------------------------------------------------------------------
# Plan Swap: Cross-lingual causal probe
# ---------------------------------------------------------------------------

def run_plan_swap_crosslingual(
    target_query_dir: str,
    source_lex_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    topk: int = 100,
    batch_size: int = 16,
    n_gpu: int = 1,
) -> Dict:
    """
    Plan swap experiment: decode non-English queries using English planner output.

    This is a causal probe that tests whether the planner mismatch is the
    bottleneck. If using the English plan helps, it's direct evidence that
    the cross-lingual planner failure is the major driver of degradation.

    Args:
        target_query_dir: Directory with non-English queries
        source_lex_dir: Directory with lexical output from English queries
        output_dir: Where to write outputs
        eval_qrel_path: Path to qrels
    """
    smt_out_dir = os.path.join(output_dir, "plan_swap", "smt_ret")
    os.makedirs(smt_out_dir, exist_ok=True)

    # Stage 2: Sequential decoding using source (English) lexical plan
    rc = run_sequential_decoding(
        query_dir=target_query_dir,
        smt_out_dir=smt_out_dir,
        lex_out_dir=source_lex_dir,  # Use English planner output
        topk=topk,
        batch_size=batch_size,
        n_gpu=n_gpu,
    )
    if rc != 0:
        return {"_error": "plan_swap_failed", "_return_code": rc}

    results = merge_and_evaluate(
        query_dir=target_query_dir,
        smt_out_dir=smt_out_dir,
        eval_qrel_path=eval_qrel_path,
    )
    return results


# ---------------------------------------------------------------------------
# Full RQ3 pipeline for a single language/split
# ---------------------------------------------------------------------------

def run_rq3_full_pipeline(
    language: str,
    split: str,
    english_query_dir: str,
    target_query_dir: str,
    translated_query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
    skip_existing: bool = True,
) -> Dict:
    """
    Run the complete RQ3 evaluation for one language and split.

    Runs all three systems plus plan swap experiment:
    - System A: Naive cross-lingual PAG
    - System B: Sequential-only (planner disabled)
    - System C: Translate-at-inference PAG
    - Plan swap: non-English query with English planner

    Returns dict with results from all systems.
    """
    results = {
        "language": language,
        "split": split,
    }

    # Create output directories
    base_out = os.path.join(output_dir, language, split)

    # === System A: Naive cross-lingual PAG ===
    print(f"\n[RQ3] === System A: Naive cross-lingual PAG ({language}/{split}) ===")
    naive_out = os.path.join(base_out, "naive_pag")
    results["naive_pag"] = run_naive_crosslingual_pag(
        query_dir=target_query_dir,
        output_dir=base_out,
        eval_qrel_path=eval_qrel_path,
        lex_topk=lex_topk,
        smt_topk=smt_topk,
        batch_size=batch_size,
        n_gpu=n_gpu,
    )

    # === System B: Sequential-only ===
    print(f"\n[RQ3] === System B: Sequential-only ({language}/{split}) ===")
    results["seq_only"] = run_sequential_only(
        query_dir=target_query_dir,
        output_dir=base_out,
        eval_qrel_path=eval_qrel_path,
        topk=smt_topk,
        batch_size=batch_size * 2,
        n_gpu=n_gpu,
    )

    # === System C: Translate-at-inference ===
    print(f"\n[RQ3] === System C: Translate-at-inference ({language}/{split}) ===")
    results["translate_pag"] = run_translate_at_inference_pag(
        original_query_dir=target_query_dir,
        translated_query_dir=translated_query_dir,
        output_dir=base_out,
        eval_qrel_path=eval_qrel_path,
        lex_topk=lex_topk,
        smt_topk=smt_topk,
        batch_size=batch_size,
        n_gpu=n_gpu,
    )

    # === English baseline (for reference) ===
    print(f"\n[RQ3] === English baseline ({split}) ===")
    english_out = os.path.join(output_dir, "english", split)
    english_lex_dir = os.path.join(english_out, "pag", "lex_ret")
    english_smt_dir = os.path.join(english_out, "pag", "smt_ret")

    # Check if English baseline already exists
    dataset_name = _get_dataset_name(english_query_dir)
    english_run_path = os.path.join(english_smt_dir, dataset_name, "run.json")

    if os.path.exists(english_run_path) and skip_existing:
        print(f"[RQ3] English baseline already exists, loading from cache")
        results["english_pag"] = {"_cached": True}
    else:
        os.makedirs(english_lex_dir, exist_ok=True)
        os.makedirs(english_smt_dir, exist_ok=True)

        # Run lexical
        rc = run_lexical_retrieval(
            query_dir=english_query_dir,
            lex_out_dir=english_lex_dir,
            topk=lex_topk,
            batch_size=batch_size,
        )
        if rc == 0:
            rc = run_sequential_decoding(
                query_dir=english_query_dir,
                smt_out_dir=english_smt_dir,
                lex_out_dir=english_lex_dir,
                topk=smt_topk,
                batch_size=batch_size * 2,
                n_gpu=n_gpu,
            )
        if rc == 0:
            results["english_pag"] = merge_and_evaluate(
                query_dir=english_query_dir,
                smt_out_dir=english_smt_dir,
                eval_qrel_path=eval_qrel_path,
            )
        else:
            results["english_pag"] = {"_error": "failed"}

    # === Plan swap: non-English query with English plan ===
    print(f"\n[RQ3] === Plan swap: {language} query + English plan ({split}) ===")
    if os.path.exists(english_lex_dir):
        results["plan_swap"] = run_plan_swap_crosslingual(
            target_query_dir=target_query_dir,
            source_lex_dir=english_lex_dir,
            output_dir=base_out,
            eval_qrel_path=eval_qrel_path,
            topk=smt_topk,
            batch_size=batch_size * 2,
            n_gpu=n_gpu,
        )
    else:
        results["plan_swap"] = {"_error": "english_lex_missing"}

    return results


# ---------------------------------------------------------------------------
# Planner token extraction for cross-lingual analysis
# ---------------------------------------------------------------------------

def extract_crosslingual_planner_tokens(
    english_query_dir: str,
    target_query_dir: str,
    output_dir: str,
    topk: int = 100,
    batch_size: int = 8,
) -> Tuple[Dict, Dict]:
    """
    Extract planner tokens for both English and target language queries.

    This enables analysis of token-plan overlap between languages.
    """
    english_tokens_path = os.path.join(output_dir, "english_planner_tokens.json")
    target_tokens_path = os.path.join(output_dir, "target_planner_tokens.json")

    # Extract English tokens
    if not os.path.exists(english_tokens_path):
        extract_planner_tokens(
            query_dir=english_query_dir,
            out_path=english_tokens_path,
            topk=topk,
            batch_size=batch_size,
        )

    # Extract target language tokens
    if not os.path.exists(target_tokens_path):
        extract_planner_tokens(
            query_dir=target_query_dir,
            out_path=target_tokens_path,
            topk=topk,
            batch_size=batch_size,
        )

    english_tokens = load_planner_tokens(english_tokens_path)
    target_tokens = load_planner_tokens(target_tokens_path)

    return english_tokens, target_tokens

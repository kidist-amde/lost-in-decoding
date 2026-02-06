"""
PAG retrieval engine for RQ3 cross-lingual evaluation.

Wraps the existing t5_pretrainer inference pipeline with:
- Latency tracking (per-query average)
- TREC-format run file output
- Error counting
- Stage 1 (SimulOnly) and Stage 2 (full PAG) modes

Reuses: robustness.utils.pag_inference for all subprocess calls.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
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
# TREC run file conversion
# ---------------------------------------------------------------------------

def run_json_to_trec(
    run: Dict[str, Dict[str, float]],
    output_path: str,
    run_id: str = "pag_run",
) -> str:
    """Convert a run dict to TREC format file.

    Format: qid Q0 pid rank score run_id

    Args:
        run: {qid: {pid: score, ...}}
        output_path: where to write the TREC file
        run_id: identifier for this run

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for qid in sorted(run.keys(), key=lambda x: int(x) if x.isdigit() else x):
            docs = run[qid]
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_docs, 1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {run_id}\n")
    return output_path


# ---------------------------------------------------------------------------
# Stage 1: Lexical retrieval (planner / SimulOnly)
# ---------------------------------------------------------------------------

def run_stage1(
    query_dir: str,
    lex_out_dir: str,
    topk: int = 1000,
    batch_size: int = 8,
) -> Tuple[int, float]:
    """Run Stage 1 (lexical planner).

    Returns:
        (return_code, elapsed_seconds)
    """
    os.makedirs(lex_out_dir, exist_ok=True)
    t0 = time.time()
    rc = run_lexical_retrieval(
        query_dir=query_dir,
        lex_out_dir=lex_out_dir,
        topk=topk,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0
    return rc, elapsed


# ---------------------------------------------------------------------------
# Stage 2: Sequential decoding with lexical constraint
# ---------------------------------------------------------------------------

def run_stage2(
    query_dir: str,
    smt_out_dir: str,
    lex_out_dir: str,
    topk: int = 100,
    batch_size: int = 16,
    n_gpu: int = 1,
    lex_constrained: str = "lexical_tmp_rescore",
) -> Tuple[int, float]:
    """Run Stage 2 (sequential constrained decoding).

    Returns:
        (return_code, elapsed_seconds)
    """
    os.makedirs(smt_out_dir, exist_ok=True)
    t0 = time.time()
    rc = run_sequential_decoding(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        lex_out_dir=lex_out_dir,
        topk=topk,
        batch_size=batch_size,
        n_gpu=n_gpu,
        lex_constrained=lex_constrained,
    )
    elapsed = time.time() - t0
    return rc, elapsed


# ---------------------------------------------------------------------------
# Sequential-only (unconstrained) decoding
# ---------------------------------------------------------------------------

def run_sequential_unconstrained(
    query_dir: str,
    smt_out_dir: str,
    topk: int = 100,
    max_new_token: int = 8,
    batch_size: int = 16,
    n_gpu: int = 1,
) -> Tuple[int, float]:
    """Run sequential decoder without lexical planner constraint.

    Returns:
        (return_code, elapsed_seconds)
    """
    os.makedirs(smt_out_dir, exist_ok=True)
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
        f"--max_length=128",
        f"--max_new_token_for_docid={max_new_token}",
    ]
    t0 = time.time()
    rc = _run_cmd(cmd)
    elapsed = time.time() - t0
    return rc, elapsed


# ---------------------------------------------------------------------------
# Full pipeline helpers
# ---------------------------------------------------------------------------

def run_full_pag(
    query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    label: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
) -> Dict:
    """Run Stages 1 + 2 + merge-evaluate.

    Returns result dict with metrics and timing.
    """
    lex_out_dir = os.path.join(output_dir, label, "lex_ret")
    smt_out_dir = os.path.join(output_dir, label, "smt_ret")

    result = {"label": label, "errors": []}

    # Stage 1
    rc1, t1 = run_stage1(query_dir, lex_out_dir, lex_topk, batch_size)
    result["stage1_time_s"] = t1
    if rc1 != 0:
        result["errors"].append(f"stage1_rc={rc1}")
        return result

    # Stage 2
    rc2, t2 = run_stage2(
        query_dir, smt_out_dir, lex_out_dir,
        smt_topk, batch_size * 2, n_gpu,
    )
    result["stage2_time_s"] = t2
    if rc2 != 0:
        result["errors"].append(f"stage2_rc={rc2}")
        return result

    # Merge & evaluate
    eval_result = merge_and_evaluate(query_dir, smt_out_dir, eval_qrel_path)
    result["eval"] = eval_result

    return result


def count_queries(query_dir: str) -> int:
    """Count queries in a raw.tsv file."""
    tsv = os.path.join(query_dir, "raw.tsv")
    if not os.path.exists(tsv):
        return 0
    with open(tsv) as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Load and convert run files
# ---------------------------------------------------------------------------

def load_and_save_trec_run(
    out_dir: str,
    dataset_name: str,
    trec_path: str,
    run_id: str,
) -> Optional[Dict]:
    """Load a run.json and also save it as TREC format.

    Returns the run dict, or None if not found.
    """
    run = load_sequential_run(out_dir, dataset_name)
    if not run:
        run = load_lexical_run(out_dir, dataset_name)
    if run:
        run_json_to_trec(run, trec_path, run_id)
    return run

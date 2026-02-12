"""
Wrapper around the RIPOR inference pipeline.

Provides functions to run the single-stage RIPOR retrieval
(constrained beam search over semantic IDs, no lexical planner)
on arbitrary query sets and evaluate the results against qrels.

This module shells out to the RIPOR ``t5_pretrainer.evaluate`` entry
point (at ``RIPOR/t5_pretrainer/``) so that all model-loading,
tokenisation and decoding logic is reused without duplication.

IMPORTANT: RIPOR has its own ``t5_pretrainer`` package under ``RIPOR/``
with different model classes (``T5SeqAQEncoder``).  The subprocess is
launched with ``PYTHONPATH=RIPOR/`` and ``cwd=RIPOR/`` to ensure the
correct package is imported.
"""

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Default paths (relative to pag-repro repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RIPOR_ROOT = os.path.join(REPO_ROOT, "RIPOR")

PRETRAINED_PATH = os.path.join(
    REPO_ROOT,
    "RIPOR", "RIPOR_data",
    "t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2",
    "checkpoint",
)

DOCID_TO_SMTID_PATH = os.path.join(
    REPO_ROOT,
    "RIPOR", "RIPOR_data",
    "experiments-full-t5seq-aq",
    "t5_docid_gen_encoder_1",
    "aq_smtid", "docid_to_smtid.json",
)

# Reuse merge and dataset-name logic from PAG wrapper
from robustness.utils.pag_inference import (
    _get_dataset_name,
    _merge_run_shards,
)


def _run_cmd(cmd: List[str], cwd: Optional[str] = None,
             env: Optional[dict] = None) -> int:
    """Run a subprocess and stream output. Returns the exit code."""
    cwd = cwd or RIPOR_ROOT
    print(f"[ripor_inference] Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc.returncode


def _ripor_env(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
) -> dict:
    """Build an environment dict that makes RIPOR's t5_pretrainer importable."""
    env = os.environ.copy()
    # Prepend RIPOR/ so ``import t5_pretrainer`` resolves to RIPOR/t5_pretrainer
    env["PYTHONPATH"] = RIPOR_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if rank is not None:
        env["RANK"] = str(rank)
    if world_size is not None:
        env["WORLD_SIZE"] = str(world_size)
    if local_rank is not None:
        env["LOCAL_RANK"] = str(local_rank)
    if master_addr is not None:
        env["MASTER_ADDR"] = str(master_addr)
    if master_port is not None:
        env["MASTER_PORT"] = str(master_port)
    return env


# ---------------------------------------------------------------------------
# RIPOR retrieval (single-stage constrained beam search)
# ---------------------------------------------------------------------------

def run_ripor_retrieval(
    query_dir: str,
    out_dir: str,
    pretrained_path: str = PRETRAINED_PATH,
    docid_to_smtid_path: str = DOCID_TO_SMTID_PATH,
    topk: int = 1000,
    max_new_token: int = 32,
    batch_size: int = 1,
    n_gpu: int = 1,
) -> int:
    """
    Run RIPOR constrained beam search retrieval.

    Calls RIPOR's ``t5_pretrainer.evaluate`` with
    ``task=t5seq_aq_retrieve_docids``.
    """
    query_dir = os.path.abspath(query_dir)
    out_dir = os.path.abspath(out_dir)
    q_collection_paths = json.dumps([query_dir])

    base_args = [
        "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={pretrained_path}",
        f"--out_dir={out_dir}",
        "--task=t5seq_aq_retrieve_docids",
        f"--docid_to_smtid_path={docid_to_smtid_path}",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--max_new_token_for_docid={max_new_token}",
        f"--topk={topk}",
    ]

    master_port = os.environ.get("MASTER_PORT", "29500")

    if n_gpu <= 1:
        # RIPOR evaluate always calls ddp_setup(); provide env:// variables
        # explicitly for a single-process world to avoid launcher wrappers.
        cmd = [sys.executable] + base_args
        env = _ripor_env(
            rank=0,
            world_size=1,
            local_rank=0,
            master_addr="127.0.0.1",
            master_port=master_port,
        )
    else:
        # Multi-GPU path: use torch.distributed.run and rely on env injection.
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpu}",
            f"--master_port={master_port}",
        ] + base_args
        env = _ripor_env()

    return _run_cmd(cmd, cwd=RIPOR_ROOT, env=env)


# ---------------------------------------------------------------------------
# Merge run shards & evaluate
# ---------------------------------------------------------------------------

def merge_and_evaluate(
    query_dir: str,
    out_dir: str,
    eval_qrel_path: str,
) -> Dict:
    """
    Merge per-GPU run files into ``run.json`` and evaluate with pytrec_eval.

    Uses the same logic as pag_inference.merge_and_evaluate.
    """
    dataset_name = _get_dataset_name(query_dir)
    ds_out_dir = os.path.join(out_dir, dataset_name)

    # Merge shards if needed
    merged_path = os.path.join(ds_out_dir, "run.json")
    shard_paths = glob.glob(os.path.join(ds_out_dir, "run_*.json"))
    if shard_paths:
        _merge_run_shards(ds_out_dir)
    elif not os.path.exists(merged_path):
        print(f"[merge] No run files found in {ds_out_dir}")
        return {"_error": "no_run_files"}

    # Evaluate using pytrec_eval directly
    try:
        import pytrec_eval

        with open(merged_path) as f:
            run = json.load(f)
        with open(eval_qrel_path) as f:
            qrels = json.load(f)

        # Convert to string keys (pytrec_eval requirement)
        str_run = {str(q): {str(d): float(s) for d, s in docs.items()}
                   for q, docs in run.items() if str(q) in qrels}
        str_qrels = {str(q): {str(d): int(s) for d, s in docs.items()}
                     for q, docs in qrels.items()}

        evaluator = pytrec_eval.RelevanceEvaluator(
            str_qrels, {"ndcg_cut_10", "recip_rank", "recall_100"}
        )
        per_query = evaluator.evaluate(str_run)

        n = len(per_query)
        results = {
            dataset_name: {
                "NDCG@10": sum(v["ndcg_cut_10"] for v in per_query.values()) / max(n, 1),
                "MRR@10": sum(v["recip_rank"] for v in per_query.values()) / max(n, 1),
                "Recall@100": sum(v["recall_100"] for v in per_query.values()) / max(n, 1),
                "n_evaluated": n,
            }
        }
        # Save per-dataset results
        perf_path = os.path.join(ds_out_dir, "perf.json")
        with open(perf_path, "w") as f:
            json.dump(results[dataset_name], f, indent=2)

        print(f"[eval] {dataset_name}: NDCG@10={results[dataset_name]['NDCG@10']:.5f}, "
              f"MRR@10={results[dataset_name]['MRR@10']:.5f}")
        return results

    except Exception as e:
        print(f"[eval] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Full RIPOR pipeline (retrieve + merge + eval)
# ---------------------------------------------------------------------------

def run_ripor_pipeline(
    query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    label: str = "ripor",
    pretrained_path: str = PRETRAINED_PATH,
    docid_to_smtid_path: str = DOCID_TO_SMTID_PATH,
    topk: int = 1000,
    max_new_token: int = 32,
    batch_size: int = 1,
    n_gpu: int = 1,
) -> Dict:
    """
    Run the full RIPOR pipeline and return evaluation results.

    Outputs are written to ``output_dir/<label>/``.
    """
    out_dir = os.path.abspath(os.path.join(output_dir, label))
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1 (only stage): constrained beam search
    rc = run_ripor_retrieval(
        query_dir=query_dir,
        out_dir=out_dir,
        pretrained_path=pretrained_path,
        docid_to_smtid_path=docid_to_smtid_path,
        topk=topk,
        max_new_token=max_new_token,
        batch_size=batch_size,
        n_gpu=n_gpu,
    )
    if rc != 0:
        print(f"[ripor_inference] Retrieval failed with exit code {rc}")
        return {"_error": "retrieval_failed", "_return_code": rc}

    # Merge shards & evaluate
    results = merge_and_evaluate(
        query_dir=os.path.abspath(query_dir),
        out_dir=out_dir,
        eval_qrel_path=os.path.abspath(eval_qrel_path),
    )
    return results


# ---------------------------------------------------------------------------
# Load run files
# ---------------------------------------------------------------------------

def load_ripor_run(out_dir: str, dataset_name: str) -> Dict:
    """Load the RIPOR output run.json. Returns {qid: {docid: score, ...}}."""
    run_path = os.path.join(out_dir, dataset_name, "run.json")
    if not os.path.exists(run_path):
        return {}
    with open(run_path) as f:
        return json.load(f)

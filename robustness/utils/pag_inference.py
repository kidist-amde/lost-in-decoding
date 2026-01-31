"""
Wrapper around the PAG inference pipeline.

Provides functions to run the two-stage PAG retrieval
(lexical planner -> constrained sequential decoder) on arbitrary query sets
and evaluate the results against qrels.

This module shells out to the existing ``t5_pretrainer.evaluate`` entry
points so that all model-loading, tokenisation and decoding logic is reused
without duplication.
"""

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR = os.path.join(
    REPO_ROOT,
    "data", "experiments-full-lexical-ripor",
    "lexical_ripor_direct_lng_knp_seq2seq_1",
)
PRETRAINED_PATH = os.path.join(MODEL_DIR, "checkpoint")

LEX_DOCID_PATH = os.path.join(
    REPO_ROOT,
    "data", "experiments-splade", "t5-splade-0-12l",
    "top_bow", "docid_to_tokenids.json",
)

SMT_DOCID_PATH = os.path.join(
    REPO_ROOT,
    "data", "experiments-full-lexical-ripor",
    "t5-full-dense-1-5e-4-12l",
    "aq_smtid", "docid_to_tokenids.json",
)


def _run_cmd(cmd: List[str], cwd: Optional[str] = None) -> int:
    """Run a subprocess and stream output. Returns the exit code."""
    cwd = cwd or REPO_ROOT
    print(f"[pag_inference] Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc.returncode


# ---------------------------------------------------------------------------
# Stage 1: Lexical retrieval (planner)
# ---------------------------------------------------------------------------

def run_lexical_retrieval(
    query_dir: str,
    lex_out_dir: str,
    pretrained_path: str = PRETRAINED_PATH,
    lex_docid_path: str = LEX_DOCID_PATH,
    smt_docid_path: str = SMT_DOCID_PATH,
    topk: int = 1000,
    batch_size: int = 8,
    max_length: int = 128,
    eval_qrel_path: Optional[str] = None,
) -> int:
    """
    Stage 1 of PAG: lexical retrieval (planner).

    Calls ``t5_pretrainer.evaluate`` with
    ``task=lexical_constrained_retrieve_and_rerank``.

    Parameters
    ----------
    query_dir : str
        Directory containing ``raw.tsv`` with queries.
    lex_out_dir : str
        Directory to write lexical run files (``<dataset>/run.json``).
    """
    q_collection_paths = json.dumps([query_dir])

    # Build a minimal eval_qrel_path list (stage 1 eval is optional)
    if eval_qrel_path is None:
        eval_qrel_json = json.dumps([None])
    else:
        eval_qrel_json = json.dumps([eval_qrel_path])

    cmd = [
        sys.executable, "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={pretrained_path}",
        f"--out_dir={lex_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--topk={topk}",
        f"--lex_docid_to_smtid_path={lex_docid_path}",
        f"--smt_docid_to_smtid_path={smt_docid_path}",
        f"--max_length={max_length}",
        f"--eval_qrel_path={eval_qrel_json}",
    ]
    return _run_cmd(cmd)


# ---------------------------------------------------------------------------
# Stage 2: Sequential decoding with lexical constraint
# ---------------------------------------------------------------------------

def run_sequential_decoding(
    query_dir: str,
    smt_out_dir: str,
    lex_out_dir: str,
    pretrained_path: str = PRETRAINED_PATH,
    lex_docid_path: str = LEX_DOCID_PATH,
    smt_docid_path: str = SMT_DOCID_PATH,
    topk: int = 100,
    max_new_token: int = 8,
    batch_size: int = 16,
    max_length: int = 128,
    lex_constrained: str = "lexical_tmp_rescore",
    eval_qrel_path: Optional[str] = None,
    n_gpu: int = 1,
) -> int:
    """
    Stage 2 of PAG: sequential constrained decoding.

    Uses ``torch.distributed.launch`` to run
    ``task=lexical_constrained_retrieve_and_rerank_2``.
    """
    q_collection_paths = json.dumps([query_dir])
    if eval_qrel_path is None:
        eval_qrel_json = json.dumps([None])
    else:
        eval_qrel_json = json.dumps([eval_qrel_path])

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={n_gpu}",
        "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={pretrained_path}",
        f"--out_dir={smt_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank_2",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={batch_size}",
        f"--topk={topk}",
        f"--lex_docid_to_smtid_path={lex_docid_path}",
        f"--smt_docid_to_smtid_path={smt_docid_path}",
        f"--max_length={max_length}",
        f"--max_new_token_for_docid={max_new_token}",
        f"--eval_qrel_path={eval_qrel_json}",
        f"--lex_constrained={lex_constrained}",
    ]
    return _run_cmd(cmd)


# ---------------------------------------------------------------------------
# Stage 3: Merge run shards & evaluate
# ---------------------------------------------------------------------------

def _get_dataset_name(path: str) -> str:
    """
    Replicate t5_pretrainer.utils.utils.get_dataset_name() so we can compute
    the dataset subdirectory name without importing (and initialising
    torch.distributed).
    """
    if "TREC_DL_2019" in path:
        return "TREC_DL_2019"
    elif "trec2020" in path or "TREC_DL_2020" in path:
        return "TREC_DL_2020"
    elif "msmarco" in path:
        if "train_queries" in path:
            return "MSMARCO_TRAIN"
        return "MSMARCO"
    return "other_dataset"


def _merge_run_shards(out_dir: str) -> Dict:
    """
    Merge per-GPU ``run_*.json`` shard files into a single ``run.json``.

    This is a standalone reimplementation of the merge logic in
    ``lexical_constrained_retrieve_and_rerank_3`` that avoids the
    ``assert len(sub_paths) == torch.cuda.device_count()`` guard (which
    can fail when the number of visible GPUs differs from ``nproc_per_node``).
    """
    # Find all shard files
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
    if not shard_paths:
        print(f"[merge] No run shards found in {out_dir}")
        return {}

    qid_to_rankdata: Dict = {}
    for shard in shard_paths:
        with open(shard) as f:
            sub = json.load(f)
        for qid, rankdata in sub.items():
            if qid not in qid_to_rankdata:
                qid_to_rankdata[qid] = rankdata
            else:
                qid_to_rankdata[qid].update(rankdata)

    merged_path = os.path.join(out_dir, "run.json")
    with open(merged_path, "w") as f:
        json.dump(qid_to_rankdata, f)

    # Clean up shards
    for shard in shard_paths:
        os.remove(shard)

    print(f"[merge] Merged {len(shard_paths)} shards -> {merged_path} "
          f"({len(qid_to_rankdata)} queries)")
    return qid_to_rankdata


def merge_and_evaluate(
    query_dir: str,
    smt_out_dir: str,
    eval_qrel_path: str,
) -> Dict:
    """
    Stage 3: merge per-GPU run files into ``run.json`` and evaluate.

    Performs the merge locally (avoiding the upstream assertion that
    ``#shards == torch.cuda.device_count()``) and evaluates using
    pytrec_eval directly rather than shelling out to the ``_3`` task.
    """
    dataset_name = _get_dataset_name(query_dir)
    ds_out_dir = os.path.join(smt_out_dir, dataset_name)

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

        # Save combined results
        all_path = os.path.join(smt_out_dir, "perf_all_datasets.json")
        with open(all_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[eval] {dataset_name}: NDCG@10={results[dataset_name]['NDCG@10']:.5f}, "
              f"MRR@10={results[dataset_name]['MRR@10']:.5f}")
        return results

    except Exception as e:
        print(f"[eval] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Full PAG pipeline (stages 1+2+3) on a single query set
# ---------------------------------------------------------------------------

def run_pag_pipeline(
    query_dir: str,
    output_dir: str,
    eval_qrel_path: str,
    label: str = "run",
    pretrained_path: str = PRETRAINED_PATH,
    lex_docid_path: str = LEX_DOCID_PATH,
    smt_docid_path: str = SMT_DOCID_PATH,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    max_new_token: int = 8,
    batch_size: int = 8,
    n_gpu: int = 1,
) -> Dict:
    """
    Run the full PAG pipeline and return evaluation results.

    Outputs are written to ``output_dir/<label>/``.
    """
    lex_out_dir = os.path.join(output_dir, label, "lex_ret")
    smt_out_dir = os.path.join(output_dir, label, "smt_ret")
    os.makedirs(lex_out_dir, exist_ok=True)
    os.makedirs(smt_out_dir, exist_ok=True)

    # Stage 1
    rc = run_lexical_retrieval(
        query_dir=query_dir,
        lex_out_dir=lex_out_dir,
        pretrained_path=pretrained_path,
        lex_docid_path=lex_docid_path,
        smt_docid_path=smt_docid_path,
        topk=lex_topk,
        batch_size=batch_size,
    )
    if rc != 0:
        print(f"[pag_inference] Stage 1 failed with exit code {rc}")
        return {"_error": "stage1_failed", "_return_code": rc}

    # Stage 2
    rc = run_sequential_decoding(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        lex_out_dir=lex_out_dir,
        pretrained_path=pretrained_path,
        lex_docid_path=lex_docid_path,
        smt_docid_path=smt_docid_path,
        topk=smt_topk,
        max_new_token=max_new_token,
        batch_size=batch_size * 2,
        n_gpu=n_gpu,
    )
    if rc != 0:
        print(f"[pag_inference] Stage 2 failed with exit code {rc}")
        return {"_error": "stage2_failed", "_return_code": rc}

    # Stage 3
    results = merge_and_evaluate(
        query_dir=query_dir,
        smt_out_dir=smt_out_dir,
        eval_qrel_path=eval_qrel_path,
    )
    return results


# ---------------------------------------------------------------------------
# Collect lexical planner outputs for plan-collapse analysis
# ---------------------------------------------------------------------------

def load_lexical_run(lex_out_dir: str, dataset_name: str) -> Dict:
    """
    Load the lexical planner output (Stage 1 run.json) for plan-collapse
    analysis.  Returns {qid: {docid: score, ...}}.
    """
    run_path = os.path.join(lex_out_dir, dataset_name, "run.json")
    if not os.path.exists(run_path):
        return {}
    with open(run_path) as f:
        return json.load(f)


def load_sequential_run(smt_out_dir: str, dataset_name: str) -> Dict:
    """Load the sequential decoder output (Stage 2 run.json)."""
    run_path = os.path.join(smt_out_dir, dataset_name, "run.json")
    if not os.path.exists(run_path):
        return {}
    with open(run_path) as f:
        return json.load(f)

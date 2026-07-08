#!/usr/bin/env python3
"""
Re-decode RQ2 query variations (clean + 5 attacks x 5 seeds) through the
exact Table 3 headline pipeline (t5_pretrainer.evaluate, tasks
lexical_constrained_retrieve_and_rerank / _2 / _3), into a fresh, isolated
output root that cannot collide with any existing artifact.

Does NOT change any decoding config from the verified Table 3 command.
Only the query set (and matching qrel) is swapped per run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from robustness.query_variations.loader import (
    ATTACK_METHODS, SEEDS, SPLIT_QUERY_PATHS, SPLIT_QREL_PATHS,
    load_queries_tsv, load_variation_json, write_perturbed_queries_tsv,
)

MODEL_DIR = os.path.join(REPO_ROOT, "data", "experiments-full-lexical-ripor",
                          "lexical_ripor_direct_lng_knp_seq2seq_1")
PRETRAINED_PATH = os.path.join(MODEL_DIR, "checkpoint") + "/"
LEX_DOCID_PATH = os.path.join(REPO_ROOT, "data", "experiments-splade",
                               "t5-splade-0-12l", "top_bow", "docid_to_tokenids.json")
SMT_DOCID_PATH = os.path.join(REPO_ROOT, "data", "experiments-full-lexical-ripor",
                               "t5-full-dense-1-5e-4-12l", "aq_smtid", "docid_to_tokenids.json")

OUT_ROOT = os.path.join(MODEL_DIR, "RQ2_REDECODE_2026-07-07")

SPLIT_TO_LABEL = {"dl19": "dl19", "dl20": "dl20", "dev": "dev"}
SPLIT_TO_QREL_GRADED = {
    "dl19": SPLIT_QREL_PATHS["dl19"],
    "dl20": SPLIT_QREL_PATHS["dl20"],
    "dev": SPLIT_QREL_PATHS["dev"],
}
SPLIT_TO_QREL_BINARY = {
    "dl19": SPLIT_QREL_PATHS["dl19_binary"],
    "dl20": SPLIT_QREL_PATHS["dl20_binary"],
    "dev": SPLIT_QREL_PATHS["dev"],
}

LEX_TOPK = 1000
SMT_TOPK = 100
MAX_NEW_TOKEN = 8
MAX_LENGTH = 128
BATCH_SIZE_S1 = 8
BATCH_SIZE_S2 = 16
LEX_CONSTRAINED = "lexical_tmp_rescore"


def safe_mkdir(path):
    """mkdir with no -p: fails loudly if path already exists or parent missing."""
    os.mkdir(path)


def build_query_dir(split, condition_dir, attack_method=None, seed=None):
    """
    Write a raw.tsv for this split/condition into condition_dir/<dataset_label>/raw.tsv
    dataset_label must match get_dataset_name(): TREC_DL_2019, TREC_DL_2020, msmarco
    """
    clean_queries = load_queries_tsv(SPLIT_QUERY_PATHS[split])
    split_qids = set(clean_queries.keys())

    label = {"dl19": "TREC_DL_2019", "dl20": "TREC_DL_2020", "dev": "msmarco_dev"}[split]
    q_dir = os.path.join(condition_dir, label)

    if attack_method is None:
        queries = clean_queries
    else:
        all_variations = load_variation_json(attack_method, seed)
        queries = {qid: text for qid, text in all_variations.items() if qid in split_qids}

    tsv_path = os.path.join(q_dir, "raw.tsv")
    write_perturbed_queries_tsv(queries, tsv_path)
    return q_dir, len(queries)


def run_cmd(cmd, log_path):
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def run_pipeline_for_condition(split, attack_method, seed, dry_run_print_only=False):
    condition_label = "clean" if attack_method is None else f"{attack_method}"
    if attack_method is None:
        run_dir = os.path.join(OUT_ROOT, split, "clean")
    else:
        run_dir = os.path.join(OUT_ROOT, split, attack_method, f"seed_{seed}")

    # mkdir with no -p: fails if it already exists or parent missing -> abort this run
    try:
        os.makedirs(os.path.dirname(run_dir), exist_ok=True)  # ok to pre-create dataset/attack level dirs
        safe_mkdir(run_dir)
    except FileExistsError:
        print(f"[SKIP-EXISTS] {run_dir} already exists — aborting this run, not overwriting.")
        return {"status": "skipped_exists", "run_dir": run_dir}

    logs_dir = os.path.join(run_dir, "logs")
    os.mkdir(logs_dir)

    q_dir, n_queries = build_query_dir(split, run_dir, attack_method, seed)
    dataset_name = {"dl19": "TREC_DL_2019", "dl20": "TREC_DL_2020", "dev": "MSMARCO"}[split]

    lex_out_dir = os.path.join(run_dir, "lex_ret")
    smt_out_dir = os.path.join(run_dir, "smt_ret")

    graded_qrel = SPLIT_TO_QREL_GRADED[split]
    binary_qrel = SPLIT_TO_QREL_BINARY[split]
    # Same positional convention as t5_pretrainer.arguments default eval_metric:
    # slot0 -> [mrr_10, recall] (needs binary qrel), slot1 -> [ndcg_cut] (needs graded qrel)
    if split == "dev":
        eval_qrel_path = json.dumps([graded_qrel])  # dev only has one (binary) qrel
    else:
        eval_qrel_path = json.dumps([binary_qrel, graded_qrel])

    q_collection_paths = json.dumps([q_dir])

    cmd1 = [
        sys.executable, "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={PRETRAINED_PATH}",
        f"--out_dir={lex_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={BATCH_SIZE_S1}",
        f"--topk={LEX_TOPK}",
        f"--lex_docid_to_smtid_path={LEX_DOCID_PATH}",
        f"--smt_docid_to_smtid_path={SMT_DOCID_PATH}",
        f"--max_length={MAX_LENGTH}",
        f"--eval_qrel_path={eval_qrel_path}",
    ]

    master_port = str(29500 + (hash((split, attack_method, seed)) % 500))
    cmd2 = [
        sys.executable, "-m", "torch.distributed.launch",
        "--nproc_per_node=1", f"--master_port={master_port}",
        "-m", "t5_pretrainer.evaluate",
        f"--pretrained_path={PRETRAINED_PATH}",
        f"--out_dir={smt_out_dir}",
        f"--lex_out_dir={lex_out_dir}",
        "--task=lexical_constrained_retrieve_and_rerank_2",
        f"--q_collection_paths={q_collection_paths}",
        f"--batch_size={BATCH_SIZE_S2}",
        f"--topk={SMT_TOPK}",
        f"--lex_docid_to_smtid_path={LEX_DOCID_PATH}",
        f"--smt_docid_to_smtid_path={SMT_DOCID_PATH}",
        f"--max_length={MAX_LENGTH}",
        f"--max_new_token_for_docid={MAX_NEW_TOKEN}",
        f"--eval_qrel_path={eval_qrel_path}",
        f"--lex_constrained={LEX_CONSTRAINED}",
    ]

    cmd3 = [
        sys.executable, "-m", "t5_pretrainer.evaluate",
        "--task=lexical_constrained_retrieve_and_rerank_3",
        f"--out_dir={smt_out_dir}",
        f"--q_collection_paths={q_collection_paths}",
        f"--eval_qrel_path={eval_qrel_path}",
    ]

    provenance = {
        "run_dir": run_dir,
        "split": split,
        "dataset_name": dataset_name,
        "condition": condition_label,
        "seed": seed,
        "n_queries": n_queries,
        "query_dir": q_dir,
        "eval_qrel_path": json.loads(eval_qrel_path),
        "pretrained_path": PRETRAINED_PATH,
        "lex_docid_to_smtid_path": LEX_DOCID_PATH,
        "smt_docid_to_smtid_path": SMT_DOCID_PATH,
        "lex_topk": LEX_TOPK,
        "smt_topk": SMT_TOPK,
        "max_new_token_for_docid": MAX_NEW_TOKEN,
        "max_length": MAX_LENGTH,
        "lex_constrained": LEX_CONSTRAINED,
        "cmd_stage1": cmd1,
        "cmd_stage2": cmd2,
        "cmd_stage3": cmd3,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }

    if dry_run_print_only:
        print(f"[DRY-RUN] run_dir={run_dir}")
        print(f"[DRY-RUN] out_dir (stage2)     = {smt_out_dir}")
        print(f"[DRY-RUN] lex_out_dir (stage1) = {lex_out_dir}")
        print(f"[DRY-RUN] resolved under OUT_ROOT: {os.path.commonpath([smt_out_dir, OUT_ROOT]) == OUT_ROOT}")
        with open(os.path.join(run_dir, "PROVENANCE.txt"), "w") as f:
            f.write(json.dumps(provenance, indent=2))
        return {"status": "dry_run", "run_dir": run_dir, "provenance": provenance}

    rc1 = run_cmd(cmd1, os.path.join(logs_dir, "stage1.log"))
    if rc1 != 0:
        provenance["status"] = "failed_stage1"
        with open(os.path.join(run_dir, "PROVENANCE.txt"), "w") as f:
            f.write(json.dumps(provenance, indent=2))
        return {"status": "failed_stage1", "run_dir": run_dir}

    rc2 = run_cmd(cmd2, os.path.join(logs_dir, "stage2.log"))
    if rc2 != 0:
        provenance["status"] = "failed_stage2"
        with open(os.path.join(run_dir, "PROVENANCE.txt"), "w") as f:
            f.write(json.dumps(provenance, indent=2))
        return {"status": "failed_stage2", "run_dir": run_dir}

    rc3 = run_cmd(cmd3, os.path.join(logs_dir, "stage3.log"))
    provenance["status"] = "completed" if rc3 == 0 else "failed_stage3"
    provenance["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(run_dir, "PROVENANCE.txt"), "w") as f:
        f.write(json.dumps(provenance, indent=2))

    return {"status": provenance["status"], "run_dir": run_dir,
            "run_json": os.path.join(smt_out_dir, dataset_name, "run.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["dl19", "dl20"],
                         choices=["dl19", "dl20", "dev"])
    parser.add_argument("--dry_run_first", action="store_true",
                         help="Print resolved paths for the first run only, then stop.")
    args = parser.parse_args()

    if os.path.exists(OUT_ROOT):
        print(f"NOTE: {OUT_ROOT} already exists (continuing; per-run mkdir guards still apply).")
    else:
        os.makedirs(OUT_ROOT)

    conditions = [(None, None)] + [(m, s) for m in ATTACK_METHODS for s in SEEDS]

    results = []
    first = True
    for split in args.splits:
        for attack_method, seed in conditions:
            label = "clean" if attack_method is None else f"{attack_method}/seed_{seed}"
            print(f"\n=== {split} / {label} ===", flush=True)
            res = run_pipeline_for_condition(
                split, attack_method, seed,
                dry_run_print_only=(args.dry_run_first and first),
            )
            print(f"  -> {res['status']}  ({res['run_dir']})", flush=True)
            results.append({"split": split, "condition": label, **res})
            if args.dry_run_first and first:
                first = False
                print("\n[DRY-RUN] Stopping after first run as requested.")
                return

    summary_path = os.path.join(OUT_ROOT, f"driver_summary_{int(time.time())}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDriver summary written to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RQ2 Robustness Evaluation for RIPOR (single-stage generative retriever).

This script evaluates RIPOR's robustness to query perturbations by:
1. Loading pre-generated query variations (JSON).
2. Running the RIPOR pipeline (constrained beam search, no planner) on both
   clean and perturbed queries.
3. Computing retrieval quality metrics (NDCG@10, MRR@10).
4. Computing deltas (clean - perturbed).
5. Saving results to experiments/RQ2_robustness_ripor/.

Usage
-----
    # Single attack, single split
    python -m robustness.evaluation.rq2_ripor \\
        --split dl19 \\
        --attack_method mispelling \\
        --seed 1999

    # All attacks, all splits
    python -m robustness.evaluation.rq2_ripor \\
        --split all \\
        --attack_method all \\
        --seed all

    # Skip inference (evaluate only from existing runs)
    python -m robustness.evaluation.rq2_ripor \\
        --split dl19 --attack_method mispelling --seed 1999 \\
        --eval_only
"""

import argparse
import csv
import fcntl
import json
import os
import sys
from typing import Dict, List, Optional

# Repo root (two levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from robustness.query_variations.loader import (
    ATTACK_METHODS,
    SEEDS,
    SPLIT_QREL_PATHS,
    prepare_split_queries,
)
from robustness.metrics.plan_collapse import (
    compute_retrieval_metrics,
)
from robustness.utils.ripor_inference import (
    load_ripor_run,
    merge_and_evaluate,
    run_ripor_pipeline,
)

# ---------------------------------------------------------------------------
# Dataset name mapping (must match get_dataset_name() in t5_pretrainer)
# ---------------------------------------------------------------------------

SPLIT_TO_DATASET = {
    "dl19": "TREC_DL_2019",
    "dl20": "TREC_DL_2020",
    "dev": "MSMARCO",
}

# Primary qrel for evaluation per split
SPLIT_PRIMARY_QREL = {
    "dl19": SPLIT_QREL_PATHS["dl19"],        # graded for NDCG
    "dl20": SPLIT_QREL_PATHS["dl20"],
    "dev": SPLIT_QREL_PATHS["dev"],
}

# ---------------------------------------------------------------------------
# Core RQ2 evaluation function
# ---------------------------------------------------------------------------

def evaluate_single(
    split: str,
    attack_method: str,
    seed: int,
    output_dir: str,
    variation_dir: Optional[str] = None,
    max_variants_per_qid: int = 1,
    n_gpu: int = 1,
    batch_size: int = 1,
    eval_only: bool = False,
    topk: int = 1000,
) -> Dict:
    """
    Run the full RQ2 evaluation for one (split, attack_method, seed) triple
    using RIPOR (single-stage, no planner).

    Produces retrieval quality metrics:
      - RIPOR NDCG@10, MRR@10  (clean + perturbed)
      - Deltas (clean - perturbed)
    """
    dataset_name = SPLIT_TO_DATASET[split]

    # -- 1. Prepare query TSVs ---
    queries_base = os.path.join(output_dir, "queries")
    clean_dir, perturbed_dir, n_queries = prepare_split_queries(
        attack_method=attack_method,
        seed=seed,
        split=split,
        output_base=queries_base,
        variation_dir=variation_dir,
        max_variants_per_qid=max_variants_per_qid,
    )
    print(f"[RQ2-RIPOR] Split={split}, attack={attack_method}, seed={seed}, "
          f"n_queries={n_queries}")

    qrel_path = SPLIT_PRIMARY_QREL[split]

    result = {
        "split": split,
        "dataset": dataset_name,
        "attack_method": attack_method,
        "seed": seed,
        "n_queries": n_queries,
    }

    # Output directories
    clean_output = os.path.join(output_dir, dataset_name, "clean")
    pert_label = f"{attack_method}_seed_{seed}"
    pert_output = os.path.join(output_dir, dataset_name, "perturbed", pert_label)

    clean_run_dir = os.path.join(clean_output, "ripor")
    pert_run_dir = os.path.join(pert_output, "ripor")

    if not eval_only:
        # -- 2. Run RIPOR on clean queries ---
        clean_run_path = os.path.join(
            clean_run_dir, dataset_name, "run.json"
        )
        if os.path.exists(clean_run_path):
            print(f"\n[RQ2-RIPOR] Clean-stage outputs already exist for {split}, "
                  f"skipping inference.")
        else:
            print(f"\n[RQ2-RIPOR] === Running RIPOR on CLEAN queries ({split}) ===")
            clean_results = run_ripor_pipeline(
                query_dir=clean_dir,
                output_dir=clean_output,
                eval_qrel_path=qrel_path,
                label="ripor",
                topk=topk,
                batch_size=batch_size,
                n_gpu=n_gpu,
            )
            result["clean_eval"] = clean_results

        # -- 3. Run RIPOR on perturbed queries ---
        print(f"\n[RQ2-RIPOR] === Running RIPOR on PERTURBED queries "
              f"({split}, {attack_method}, seed={seed}) ===")
        pert_results = run_ripor_pipeline(
            query_dir=perturbed_dir,
            output_dir=pert_output,
            eval_qrel_path=qrel_path,
            label="ripor",
            topk=topk,
            batch_size=batch_size,
            n_gpu=n_gpu,
        )
        result["perturbed_eval"] = pert_results

    # -- 4. Compute metrics from run files ---
    try:
        with open(qrel_path) as f:
            qrels = json.load(f)

        # Load runs
        clean_run = load_ripor_run(clean_run_dir, dataset_name)
        pert_run = load_ripor_run(pert_run_dir, dataset_name)

        if clean_run:
            result["clean_metrics"] = compute_retrieval_metrics(
                clean_run, qrels, k=10
            )
        if pert_run:
            result["perturbed_metrics"] = compute_retrieval_metrics(
                pert_run, qrels, k=10
            )

    except Exception as e:
        print(f"[RQ2-RIPOR] Warning: metric computation failed: {e}")
        import traceback
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(results: List[Dict], output_dir: str, split_tag: Optional[str] = None):
    """Write summary.json and summary.csv (optionally split-tagged)."""
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{split_tag}" if split_tag and split_tag != "all" else ""
    summary_json = os.path.join(output_dir, f"summary{suffix}.json")
    summary_csv = os.path.join(output_dir, f"summary{suffix}.csv")
    lock_path = os.path.join(output_dir, f"summary{suffix}.lock")

    # Merge with existing summaries under a lock so concurrent SLURM jobs
    # append/update rows instead of clobbering each other.
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

        merged: Dict[tuple, Dict] = {}

        if os.path.exists(summary_json):
            try:
                with open(summary_json) as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    for r in existing:
                        key = (r.get("split"), r.get("attack_method"), r.get("seed"))
                        merged[key] = r
            except Exception:
                # If an existing file is malformed, continue with new rows.
                pass

        for r in results:
            key = (r.get("split"), r.get("attack_method"), r.get("seed"))
            merged[key] = r

        merged_results = list(merged.values())
        merged_results.sort(key=lambda r: (str(r.get("split")), str(r.get("attack_method")), str(r.get("seed"))))

        with open(summary_json, "w") as f:
            json.dump(merged_results, f, indent=2)
        print(f"[RQ2-RIPOR] Wrote {summary_json}")

        # CSV: flatten key metrics into rows
        csv_rows = []
        for r in merged_results:
            row = {
                "split": r.get("split"),
                "attack_method": r.get("attack_method"),
                "seed": r.get("seed"),
                "n_queries": r.get("n_queries"),
            }

            # Retrieval metrics
            for prefix, key in [
                ("clean", "clean_metrics"),
                ("pert", "perturbed_metrics"),
            ]:
                metrics = r.get(key, {})
                for m in ["NDCG@10", "MRR@10"]:
                    row[f"{prefix}_{m}"] = metrics.get(m)

            # Deltas
            for m in ["NDCG@10", "MRR@10"]:
                c = row.get(f"clean_{m}")
                p = row.get(f"pert_{m}")
                if c is not None and p is not None:
                    row[f"delta_{m}"] = c - p

            csv_rows.append(row)

        if csv_rows:
            fieldnames = []
            seen = set()
            for row in csv_rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        fieldnames.append(k)

            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[RQ2-RIPOR] Wrote {summary_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2 Robustness Evaluation for RIPOR"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dl19",
        help="Evaluation split: dl19, dl20, dev, or 'all'.",
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="mispelling",
        help=(
            "Attack method: mispelling, ordering, synonym, paraphrase, "
            "naturality, or 'all'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="1999",
        help="Random seed used for variation generation: 1999, 5, 27, 2016, 2026, or 'all'.",
    )
    parser.add_argument(
        "--variation_dir",
        type=str,
        default=None,
        help="Override directory for query variation JSONs.",
    )
    parser.add_argument(
        "--max_variants_per_qid",
        type=int,
        default=1,
        help="Max number of variants per query (currently 1 per JSON).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "RQ2_robustness_ripor"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs for retrieval.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip inference; only compute metrics from existing run files.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1000,
        help="Top-K for constrained beam search.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve 'all' values
    splits = SPLIT_TO_DATASET.keys() if args.split == "all" else [args.split]
    attacks = ATTACK_METHODS if args.attack_method == "all" else [args.attack_method]
    seeds = SEEDS if args.seed == "all" else [int(args.seed)]

    all_results = []
    total = len(list(splits)) * len(attacks) * len(seeds)
    i = 0

    for split in splits:
        for attack in attacks:
            for seed in seeds:
                i += 1
                print(f"\n{'='*70}")
                print(f"[RQ2-RIPOR] Experiment {i}/{total}: "
                      f"split={split}, attack={attack}, seed={seed}")
                print(f"{'='*70}\n")

                result = evaluate_single(
                    split=split,
                    attack_method=attack,
                    seed=seed,
                    output_dir=args.output_dir,
                    variation_dir=args.variation_dir,
                    max_variants_per_qid=args.max_variants_per_qid,
                    n_gpu=args.n_gpu,
                    batch_size=args.batch_size,
                    eval_only=args.eval_only,
                    topk=args.topk,
                )
                all_results.append(result)

                # Write intermediate summary
                write_summary(all_results, args.output_dir, args.split)

    # Final summary
    write_summary(all_results, args.output_dir, args.split)
    print(f"\n[RQ2-RIPOR] All experiments complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()

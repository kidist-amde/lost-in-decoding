#!/usr/bin/env python3
"""
RQ2 Robustness Evaluation for PAG (Planning Ahead in Generative Retrieval).

This script evaluates PAG's robustness to query perturbations by:
1. Loading pre-generated query variations (JSON).
2. Running the PAG pipeline (lexical planner + sequential decoder) on both
   clean and perturbed queries.
3. Computing retrieval quality metrics (NDCG@10, MRR@10, Recall).
4. Computing plan-collapse metrics (candidate overlap, size ratio, rank
   correlation) and sequential recovery deltas.
5. Saving results to experiments/RQ2_robustness/.

Usage
-----
    # Single attack, single split
    python -m robustness.evaluation.rq2 \\
        --split dl19 \\
        --attack_method mispelling \\
        --seed 1999

    # All attacks, all splits
    python -m robustness.evaluation.rq2 \\
        --split all \\
        --attack_method all \\
        --seed all

    # Skip inference (evaluate only from existing runs)
    python -m robustness.evaluation.rq2 \\
        --split dl19 --attack_method mispelling --seed 1999 \\
        --eval_only
"""

import argparse
import csv
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
from robustness.utils.pag_inference import (
    LEX_DOCID_PATH,
    MODEL_DIR,
    PRETRAINED_PATH,
    SMT_DOCID_PATH,
    load_lexical_run,
    load_sequential_run,
    run_pag_pipeline,
)
from robustness.metrics.plan_collapse import (
    aggregate_plan_collapse,
    candidate_overlap,
    compute_retrieval_metrics,
    recovery_delta,
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
    batch_size: int = 8,
    eval_only: bool = False,
    lex_topk: int = 1000,
    smt_topk: int = 100,
) -> Dict:
    """
    Run the full RQ2 evaluation for one (split, attack_method, seed) triple.
    """
    dataset_name = SPLIT_TO_DATASET[split]

    # ── 1. Prepare query TSVs ──────────────────────────────────────────
    queries_base = os.path.join(output_dir, "queries")
    clean_dir, perturbed_dir, n_queries = prepare_split_queries(
        attack_method=attack_method,
        seed=seed,
        split=split,
        output_base=queries_base,
        variation_dir=variation_dir,
        max_variants_per_qid=max_variants_per_qid,
    )
    print(f"[RQ2] Split={split}, attack={attack_method}, seed={seed}, "
          f"n_queries={n_queries}")

    qrel_path = SPLIT_PRIMARY_QREL[split]

    result = {
        "split": split,
        "dataset": dataset_name,
        "attack_method": attack_method,
        "seed": seed,
        "n_queries": n_queries,
    }

    if not eval_only:
        # ── 2. Run PAG on clean queries ──────────────────────────────────
        print(f"\n[RQ2] === Running PAG on CLEAN queries ({split}) ===")
        clean_output = os.path.join(output_dir, dataset_name, "clean")
        clean_results = run_pag_pipeline(
            query_dir=clean_dir,
            output_dir=clean_output,
            eval_qrel_path=qrel_path,
            label="pag",
            lex_topk=lex_topk,
            smt_topk=smt_topk,
            batch_size=batch_size,
            n_gpu=n_gpu,
        )
        result["clean_pag_eval"] = clean_results

        # ── 3. Run PAG on perturbed queries ────────────────────────────
        print(f"\n[RQ2] === Running PAG on PERTURBED queries "
              f"({split}, {attack_method}, seed={seed}) ===")
        pert_label = f"{attack_method}_seed_{seed}"
        pert_output = os.path.join(
            output_dir, dataset_name, "perturbed", pert_label
        )
        pert_results = run_pag_pipeline(
            query_dir=perturbed_dir,
            output_dir=pert_output,
            eval_qrel_path=qrel_path,
            label="pag",
            lex_topk=lex_topk,
            smt_topk=smt_topk,
            batch_size=batch_size,
            n_gpu=n_gpu,
        )
        result["perturbed_pag_eval"] = pert_results

    # ── 4. Plan-collapse analysis ────────────────────────────────────
    try:
        pert_label = f"{attack_method}_seed_{seed}"
        clean_lex_dir = os.path.join(
            output_dir, dataset_name, "clean", "pag", "lex_ret"
        )
        pert_lex_dir = os.path.join(
            output_dir, dataset_name, "perturbed", pert_label, "pag", "lex_ret"
        )
        clean_smt_dir = os.path.join(
            output_dir, dataset_name, "clean", "pag", "smt_ret"
        )
        pert_smt_dir = os.path.join(
            output_dir, dataset_name, "perturbed", pert_label, "pag", "smt_ret"
        )

        clean_lex_run = load_lexical_run(clean_lex_dir, dataset_name)
        pert_lex_run = load_lexical_run(pert_lex_dir, dataset_name)

        if clean_lex_run and pert_lex_run:
            plan_stats = aggregate_plan_collapse(
                clean_lex_run, pert_lex_run, topk=100
            )
            result["plan_collapse"] = plan_stats

            # Per-query plan collapse for detailed analysis
            per_query = candidate_overlap(clean_lex_run, pert_lex_run, topk=100)
            pq_path = os.path.join(
                output_dir, dataset_name, "perturbed", pert_label,
                "plan_collapse_per_query.json"
            )
            os.makedirs(os.path.dirname(pq_path), exist_ok=True)
            with open(pq_path, "w") as f:
                json.dump(per_query, f, indent=2)

        # ── 5. Evaluate lexical planner runs with qrels ────────────────
        with open(qrel_path) as f:
            qrels = json.load(f)

        if clean_lex_run:
            result["clean_lex_metrics"] = compute_retrieval_metrics(
                clean_lex_run, qrels, k=10
            )
        if pert_lex_run:
            result["perturbed_lex_metrics"] = compute_retrieval_metrics(
                pert_lex_run, qrels, k=10
            )

        # Load sequential runs for recovery analysis
        clean_smt_run = load_sequential_run(clean_smt_dir, dataset_name)
        pert_smt_run = load_sequential_run(pert_smt_dir, dataset_name)

        if clean_smt_run:
            result["clean_smt_metrics"] = compute_retrieval_metrics(
                clean_smt_run, qrels, k=10
            )
        if pert_smt_run:
            result["perturbed_smt_metrics"] = compute_retrieval_metrics(
                pert_smt_run, qrels, k=10
            )

        # ── 6. Sequential recovery delta ──────────────────────────────
        if all(k in result for k in [
            "clean_lex_metrics", "perturbed_lex_metrics",
            "clean_smt_metrics", "perturbed_smt_metrics"
        ]):
            for metric_name in ["NDCG@10", "MRR@10"]:
                delta = recovery_delta(
                    result["clean_lex_metrics"].get(metric_name, 0),
                    result["perturbed_lex_metrics"].get(metric_name, 0),
                    result["clean_smt_metrics"].get(metric_name, 0),
                    result["perturbed_smt_metrics"].get(metric_name, 0),
                )
                result[f"recovery_delta_{metric_name}"] = delta

    except Exception as e:
        print(f"[RQ2] Warning: plan-collapse analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(results: List[Dict], output_dir: str):
    """Write summary.json and summary.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    summary_json = os.path.join(output_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[RQ2] Wrote {summary_json}")

    # CSV: flatten key metrics into rows
    csv_rows = []
    for r in results:
        row = {
            "split": r.get("split"),
            "attack_method": r.get("attack_method"),
            "seed": r.get("seed"),
            "n_queries": r.get("n_queries"),
        }

        # Lexical planner metrics
        for prefix, key in [
            ("clean_lex", "clean_lex_metrics"),
            ("pert_lex", "perturbed_lex_metrics"),
            ("clean_smt", "clean_smt_metrics"),
            ("pert_smt", "perturbed_smt_metrics"),
        ]:
            metrics = r.get(key, {})
            for m in ["NDCG@10", "MRR@10"]:
                row[f"{prefix}_{m}"] = metrics.get(m)

        # Deltas
        for m in ["NDCG@10", "MRR@10"]:
            # Absolute drops
            cl = row.get(f"clean_smt_{m}")
            pl = row.get(f"pert_smt_{m}")
            if cl is not None and pl is not None:
                row[f"delta_smt_{m}"] = cl - pl
            row[f"recovery_delta_{m}"] = r.get(f"recovery_delta_{m}")

        # Plan collapse
        pc = r.get("plan_collapse", {})
        row["avg_jaccard@100"] = pc.get("avg_jaccard@100")
        row["avg_size_ratio"] = pc.get("avg_size_ratio")
        row["avg_rank_correlation@100"] = pc.get("avg_rank_correlation@100")

        csv_rows.append(row)

    if csv_rows:
        summary_csv = os.path.join(output_dir, "summary.csv")
        fieldnames = list(csv_rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[RQ2] Wrote {summary_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2 Robustness Evaluation for PAG"
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
        default=os.path.join(REPO_ROOT, "experiments", "RQ2_robustness"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs for sequential decoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip inference; only compute metrics from existing run files.",
    )
    parser.add_argument(
        "--lex_topk",
        type=int,
        default=1000,
        help="Top-K for lexical planner (Stage 1).",
    )
    parser.add_argument(
        "--smt_topk",
        type=int,
        default=100,
        help="Top-K for sequential decoder (Stage 2).",
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
                print(f"[RQ2] Experiment {i}/{total}: "
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
                    lex_topk=args.lex_topk,
                    smt_topk=args.smt_topk,
                )
                all_results.append(result)

                # Write intermediate summary
                write_summary(all_results, args.output_dir)

    # Final summary
    write_summary(all_results, args.output_dir)
    print(f"\n[RQ2] All experiments complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()

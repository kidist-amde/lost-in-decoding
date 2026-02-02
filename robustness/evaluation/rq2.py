#!/usr/bin/env python3
"""
RQ2 Robustness Evaluation for PAG (Planning Ahead in Generative Retrieval).

This script evaluates PAG's robustness to query perturbations by:
1. Loading pre-generated query variations (JSON).
2. Running the PAG pipeline (lexical planner + sequential decoder) on both
   clean and perturbed queries.
3. Computing retrieval quality metrics (NDCG@10, MRR@10, Recall).
4. Computing plan-collapse metrics (CandOverlap@100, PlanIntersect@100,
   rank correlation) and sequential recovery deltas.
5. Running plan-swapped decoding to measure PlanSwapDrop.
6. Saving results to experiments/RQ2_robustness/.

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
    extract_planner_tokens,
    load_lexical_run,
    load_planner_tokens,
    load_sequential_run,
    merge_and_evaluate,
    run_plan_swapped_decoding,
    run_pag_pipeline,
)
from robustness.metrics.plan_collapse import (
    aggregate_plan_collapse,
    candidate_overlap,
    compute_retrieval_metrics,
    plan_intersect,
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

    Produces all metrics needed for Table 1 (retrieval quality) and
    Table 2 (plan collapse / sensitivity):
      - Stage 1 (SimulOnly) NDCG@10, MRR@10  (clean + perturbed)
      - Stage 2 (PAG)       NDCG@10, MRR@10  (clean + perturbed)
      - CandOverlap@100     (candidate-set Jaccard)
      - PlanIntersect@100   (token-plan Jaccard)
      - SeqGain             (Stage2 - Stage1 on perturbed)
      - PlanSwapDrop        (perturbed-query with clean-plan vs normal PAG)
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

    # Output directories
    clean_output = os.path.join(output_dir, dataset_name, "clean")
    pert_label = f"{attack_method}_seed_{seed}"
    pert_output = os.path.join(output_dir, dataset_name, "perturbed", pert_label)

    clean_lex_dir = os.path.join(clean_output, "pag", "lex_ret")
    clean_smt_dir = os.path.join(clean_output, "pag", "smt_ret")
    pert_lex_dir = os.path.join(pert_output, "pag", "lex_ret")
    pert_smt_dir = os.path.join(pert_output, "pag", "smt_ret")

    # Planner token paths
    clean_tokens_path = os.path.join(clean_output, "planner_tokens.json")
    pert_tokens_path = os.path.join(pert_output, "planner_tokens.json")

    # Plan-swapped decoding output
    swap_smt_dir = os.path.join(pert_output, "pag_planswap", "smt_ret")

    if not eval_only:
        # ── 2. Run PAG on clean queries ──────────────────────────────────
        print(f"\n[RQ2] === Running PAG on CLEAN queries ({split}) ===")
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

        # ── 4. Extract planner tokens for PlanIntersect ────────────────
        print(f"\n[RQ2] === Extracting planner tokens ===")
        extract_planner_tokens(
            query_dir=clean_dir,
            out_path=clean_tokens_path,
            topk=100,
            batch_size=batch_size,
        )
        extract_planner_tokens(
            query_dir=perturbed_dir,
            out_path=pert_tokens_path,
            topk=100,
            batch_size=batch_size,
        )

        # ── 5. Plan-swapped decoding (PlanSwapDrop) ────────────────────
        # Decode perturbed queries using the lexical plan from clean queries
        print(f"\n[RQ2] === Plan-swapped decoding ===")
        os.makedirs(swap_smt_dir, exist_ok=True)
        rc = run_plan_swapped_decoding(
            query_dir=perturbed_dir,
            smt_out_dir=swap_smt_dir,
            swap_lex_out_dir=clean_lex_dir,
            topk=smt_topk,
            batch_size=batch_size * 2,
            n_gpu=n_gpu,
        )
        if rc == 0:
            swap_results = merge_and_evaluate(
                query_dir=perturbed_dir,
                smt_out_dir=swap_smt_dir,
                eval_qrel_path=qrel_path,
            )
            result["planswap_eval"] = swap_results
        else:
            print(f"[RQ2] Plan-swapped decoding failed (rc={rc})")

    # ── 6. Compute all metrics from run files ──────────────────────────
    try:
        with open(qrel_path) as f:
            qrels = json.load(f)

        # Load lexical planner runs
        clean_lex_run = load_lexical_run(clean_lex_dir, dataset_name)
        pert_lex_run = load_lexical_run(pert_lex_dir, dataset_name)

        # Load sequential decoder runs
        clean_smt_run = load_sequential_run(clean_smt_dir, dataset_name)
        pert_smt_run = load_sequential_run(pert_smt_dir, dataset_name)

        # Load planner tokens
        clean_tokens = load_planner_tokens(clean_tokens_path)
        pert_tokens = load_planner_tokens(pert_tokens_path)

        # --- Retrieval metrics (Table 1) ---

        if clean_lex_run:
            result["clean_lex_metrics"] = compute_retrieval_metrics(
                clean_lex_run, qrels, k=10
            )
        if pert_lex_run:
            result["perturbed_lex_metrics"] = compute_retrieval_metrics(
                pert_lex_run, qrels, k=10
            )
        if clean_smt_run:
            result["clean_smt_metrics"] = compute_retrieval_metrics(
                clean_smt_run, qrels, k=10
            )
        if pert_smt_run:
            result["perturbed_smt_metrics"] = compute_retrieval_metrics(
                pert_smt_run, qrels, k=10
            )

        # --- Plan-collapse metrics (Table 2) ---

        if clean_lex_run and pert_lex_run:
            plan_stats = aggregate_plan_collapse(
                clean_lex_run, pert_lex_run, topk=100,
                clean_planner_tokens=clean_tokens or None,
                perturbed_planner_tokens=pert_tokens or None,
            )
            result["plan_collapse"] = plan_stats

            # Per-query plan collapse for detailed analysis
            per_query = candidate_overlap(clean_lex_run, pert_lex_run, topk=100)

            # Add per-query PlanIntersect if tokens available
            if clean_tokens and pert_tokens:
                pi_per_query = plan_intersect(clean_tokens, pert_tokens, topk=100)
                for qid in per_query:
                    if qid in pi_per_query:
                        per_query[qid]["plan_intersect"] = pi_per_query[qid]["jaccard"]

            pq_path = os.path.join(pert_output, "plan_collapse_per_query.json")
            os.makedirs(os.path.dirname(pq_path), exist_ok=True)
            with open(pq_path, "w") as f:
                json.dump(per_query, f, indent=2)

        # --- SeqGain: marginal gain of Stage 2 over Stage 1 on perturbed ---

        if "perturbed_lex_metrics" in result and "perturbed_smt_metrics" in result:
            seq_gain = {}
            for m in ["NDCG@10", "MRR@10"]:
                pert_smt_val = result["perturbed_smt_metrics"].get(m, 0)
                pert_lex_val = result["perturbed_lex_metrics"].get(m, 0)
                seq_gain[m] = pert_smt_val - pert_lex_val
            result["seq_gain"] = seq_gain

        # --- Sequential recovery delta ---

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

        # --- PlanSwapDrop ---
        # Drop = normal_PAG_on_perturbed - planswap_PAG_on_perturbed
        # Positive means using the wrong plan hurts.

        swap_smt_run = load_sequential_run(swap_smt_dir, dataset_name)
        if swap_smt_run:
            result["planswap_smt_metrics"] = compute_retrieval_metrics(
                swap_smt_run, qrels, k=10
            )

        if "perturbed_smt_metrics" in result and "planswap_smt_metrics" in result:
            planswap_drop = {}
            for m in ["NDCG@10", "MRR@10"]:
                normal_val = result["perturbed_smt_metrics"].get(m, 0)
                swap_val = result["planswap_smt_metrics"].get(m, 0)
                planswap_drop[m] = normal_val - swap_val
            result["planswap_drop"] = planswap_drop

    except Exception as e:
        print(f"[RQ2] Warning: metric computation failed: {e}")
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

        # Retrieval metrics (Table 1)
        for prefix, key in [
            ("clean_lex", "clean_lex_metrics"),
            ("pert_lex", "perturbed_lex_metrics"),
            ("clean_smt", "clean_smt_metrics"),
            ("pert_smt", "perturbed_smt_metrics"),
            ("swap_smt", "planswap_smt_metrics"),
        ]:
            metrics = r.get(key, {})
            for m in ["NDCG@10", "MRR@10"]:
                row[f"{prefix}_{m}"] = metrics.get(m)

        # Deltas (Table 1)
        for m in ["NDCG@10", "MRR@10"]:
            # Absolute drops for lexical planner
            cl = row.get(f"clean_lex_{m}")
            pl = row.get(f"pert_lex_{m}")
            if cl is not None and pl is not None:
                row[f"delta_lex_{m}"] = cl - pl

            # Absolute drops for end-to-end PAG
            cs = row.get(f"clean_smt_{m}")
            ps = row.get(f"pert_smt_{m}")
            if cs is not None and ps is not None:
                row[f"delta_smt_{m}"] = cs - ps

            row[f"recovery_delta_{m}"] = r.get(f"recovery_delta_{m}")

        # Plan collapse (Table 2)
        pc = r.get("plan_collapse", {})
        row["CandOverlap@100"] = pc.get("avg_jaccard@100")
        row["PlanIntersect@100"] = pc.get("avg_plan_intersect@100")
        row["avg_rank_correlation@100"] = pc.get("avg_rank_correlation@100")
        row["avg_size_ratio"] = pc.get("avg_size_ratio")

        # SeqGain (Table 2)
        sg = r.get("seq_gain", {})
        row["SeqGain_MRR@10"] = sg.get("MRR@10")
        row["SeqGain_NDCG@10"] = sg.get("NDCG@10")

        # PlanSwapDrop (Table 2)
        psd = r.get("planswap_drop", {})
        row["PlanSwapDrop_MRR@10"] = psd.get("MRR@10")
        row["PlanSwapDrop_NDCG@10"] = psd.get("NDCG@10")

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

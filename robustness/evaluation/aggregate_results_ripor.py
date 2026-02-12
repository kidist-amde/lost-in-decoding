#!/usr/bin/env python3
"""
Aggregate RQ2 RIPOR results across seeds and produce LaTeX-ready table values.

Reads ``experiments/RQ2_robustness_ripor/summary.csv`` (produced by rq2_ripor.py),
averages over seeds, and prints formatted tables matching the paper layout.

Usage
-----
    python -m robustness.evaluation.aggregate_results_ripor
    python -m robustness.evaluation.aggregate_results_ripor --results_dir experiments/RQ2_robustness_ripor
    python -m robustness.evaluation.aggregate_results_ripor --latex
    python -m robustness.evaluation.aggregate_results_ripor \
               --splits dl19 dl20 dev \
               --attacks mispelling ordering synonym paraphrase naturality
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Reuse aggregation utilities from the PAG aggregate script
from robustness.evaluation.aggregate_results import (
    aggregate_over_seeds,
    fmt,
    fmt_delta_pm,
    fmt_pm,
    load_summary_csv,
    _get,
)


def print_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 1: RIPOR retrieval performance under query perturbations (mean +/- std)."""
    W = 15  # column width to fit mean +/- std
    print("\n" + "=" * 100)
    print("TABLE 1 (RIPOR): Retrieval performance under query perturbations  (mean \u00b1 std over seeds)")
    print("=" * 100)
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'NDCG@10':>{W}} {'Delta':>{W}} {'MRR@10':>{W}} {'Delta':>{W}}")
    print("-" * 100)

    for split in splits:
        # Clean baseline
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        cl_n, cl_n_s = _get(clean, "clean_NDCG@10")
        cl_m, cl_m_s = _get(clean, "clean_MRR@10")

        print(f"{split:<8} {'Clean':<14} "
              f"{fmt_pm(cl_n, cl_n_s):>{W}} {'--':>{W}} "
              f"{fmt_pm(cl_m, cl_m_s):>{W}} {'--':>{W}}")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            pn, pn_s = _get(r, "pert_NDCG@10")
            pm, pm_s = _get(r, "pert_MRR@10")
            dn, dn_s = _get(r, "delta_NDCG@10")
            dm, dm_s = _get(r, "delta_MRR@10")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"{'':8} {attack_label:<14} "
                  f"{fmt_pm(pn, pn_s):>{W}} {fmt_delta_pm(dn, dn_s):>{W}} "
                  f"{fmt_pm(pm, pm_s):>{W}} {fmt_delta_pm(dm, dm_s):>{W}}")

        print("-" * 100)


def _latex_pm(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    """Format as 'mean$\\pm$std' for LaTeX."""
    if mean is None:
        return "--"
    m = f"{mean:.{digits}f}"
    if std is None:
        return m
    return f"{m}$\\pm${std:.{digits}f}"


def _latex_delta_pm(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    """Format as '+mean$\\pm$std' (signed mean) for LaTeX."""
    if mean is None:
        return "--"
    m = f"{mean:+.{digits}f}" if mean != 0 else f"{mean:.{digits}f}"
    if std is None:
        return m
    return f"{m}$\\pm${std:.{digits}f}"


def print_latex_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print LaTeX-formatted Table 1 values (mean $\\pm$ std)."""
    print("\n% LaTeX Table 1 (RIPOR) values (mean $\\pm$ std over seeds)")
    print("% Format: NDCG@10 & Delta & MRR@10 & Delta")

    for split in splits:
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        cl_n, cl_n_s = _get(clean, "clean_NDCG@10")
        cl_m, cl_m_s = _get(clean, "clean_MRR@10")

        split_label = {"dl19": "DL19", "dl20": "DL20", "dev": "Dev"}[split]
        print(f"% {split_label} Clean")
        print(f"  & Clean & {_latex_pm(cl_n, cl_n_s)} & -- "
              f"& {_latex_pm(cl_m, cl_m_s)} & -- \\\\")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]
            pn, pn_s = _get(r, "pert_NDCG@10")
            pm, pm_s = _get(r, "pert_MRR@10")
            dn, dn_s = _get(r, "delta_NDCG@10")
            dm, dm_s = _get(r, "delta_MRR@10")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"% {split_label} {attack_label}")
            print(f"  & {attack_label} "
                  f"& {_latex_pm(pn, pn_s)} & {_latex_delta_pm(dn, dn_s)} "
                  f"& {_latex_pm(pm, pm_s)} & {_latex_delta_pm(dm, dm_s)} \\\\")


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ2 RIPOR results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "RQ2_robustness_ripor"),
    )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["mispelling", "paraphrase"],
        help="Attack methods to include in tables.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["dl19", "dl20", "dev"],
    )
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX values.")
    args = parser.parse_args()

    # Load per-split summary CSVs produced by rq2_ripor.py
    rows = []
    for split in args.splits:
        split_csv = os.path.join(args.results_dir, f"summary_{split}.csv")
        if os.path.exists(split_csv):
            split_rows = load_summary_csv(split_csv)
            rows.extend(split_rows)
            print(f"Loaded {len(split_rows)} rows from {split_csv}")
    if not rows:
        csv_path = os.path.join(args.results_dir, "summary.csv")
        if not os.path.exists(csv_path):
            print(f"Error: No summary CSV found in {args.results_dir}. Run rq2_ripor.py first.")
            sys.exit(1)
        rows = load_summary_csv(csv_path)
        print(f"Loaded {len(rows)} result rows from {csv_path}")

    # Filter to requested attacks
    rows = [r for r in rows if r["attack_method"] in args.attacks]

    agg = aggregate_over_seeds(rows)
    print(f"Aggregated into {len(agg)} (split, attack) groups")

    print_table1(agg, args.splits, args.attacks)

    if args.latex:
        print_latex_table1(agg, args.splits, args.attacks)


if __name__ == "__main__":
    main()

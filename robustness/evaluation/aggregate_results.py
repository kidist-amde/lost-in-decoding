#!/usr/bin/env python3
"""
Aggregate RQ2 results across seeds and produce LaTeX-ready table values.

Reads ``experiments/RQ2_robustness/summary.csv`` (produced by rq2.py),
averages over seeds, and prints formatted tables matching the paper layout.

Usage
-----
    python -m robustness.evaluation.aggregate_results
    python -m robustness.evaluation.aggregate_results --results_dir experiments/RQ2_robustness
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_summary_csv(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k, v in row.items():
                if v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "--"
    return f"{val:.{digits}f}"


def fmt_delta(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "--"
    return f"{val:+.{digits}f}" if val != 0 else f"{val:.{digits}f}"


def aggregate_over_seeds(rows: List[Dict]) -> Dict:
    """
    Group rows by (split, attack_method) and average numeric columns
    over seeds.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["split"], r["attack_method"])
        groups[key].append(r)

    agg = {}
    for key, group in groups.items():
        agg_row = {"split": key[0], "attack_method": key[1], "n_seeds": len(group)}
        # Average all numeric columns
        numeric_keys = [k for k in group[0] if k not in ("split", "attack_method", "seed")]
        for nk in numeric_keys:
            vals = [r[nk] for r in group if r[nk] is not None]
            agg_row[nk] = safe_mean(vals)
        agg[key] = agg_row
    return agg


def print_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 1: Retrieval performance under query perturbations."""
    print("\n" + "=" * 90)
    print("TABLE 1: Retrieval performance under query perturbations")
    print("=" * 90)
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'S1-NDCG':>8} {'Delta':>8} {'S1-MRR':>8} {'Delta':>8} "
          f"{'S2-NDCG':>8} {'Delta':>8} {'S2-MRR':>8} {'Delta':>8}")
    print("-" * 90)

    for split in splits:
        # Clean baseline
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        clean_lex_ndcg = clean.get("clean_lex_NDCG@10")
        clean_lex_mrr = clean.get("clean_lex_MRR@10")
        clean_smt_ndcg = clean.get("clean_smt_NDCG@10")
        clean_smt_mrr = clean.get("clean_smt_MRR@10")

        print(f"{split:<8} {'Clean':<14} "
              f"{fmt(clean_lex_ndcg):>8} {'--':>8} "
              f"{fmt(clean_lex_mrr):>8} {'--':>8} "
              f"{fmt(clean_smt_ndcg):>8} {'--':>8} "
              f"{fmt(clean_smt_mrr):>8} {'--':>8}")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            # Compute deltas from clean
            pert_lex_ndcg = r.get("pert_lex_NDCG@10")
            pert_lex_mrr = r.get("pert_lex_MRR@10")
            pert_smt_ndcg = r.get("pert_smt_NDCG@10")
            pert_smt_mrr = r.get("pert_smt_MRR@10")

            d_lex_ndcg = (clean_lex_ndcg - pert_lex_ndcg) if (clean_lex_ndcg is not None and pert_lex_ndcg is not None) else None
            d_lex_mrr = (clean_lex_mrr - pert_lex_mrr) if (clean_lex_mrr is not None and pert_lex_mrr is not None) else None
            d_smt_ndcg = (clean_smt_ndcg - pert_smt_ndcg) if (clean_smt_ndcg is not None and pert_smt_ndcg is not None) else None
            d_smt_mrr = (clean_smt_mrr - pert_smt_mrr) if (clean_smt_mrr is not None and pert_smt_mrr is not None) else None

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"{'':8} {attack_label:<14} "
                  f"{fmt(pert_lex_ndcg):>8} {fmt_delta(-d_lex_ndcg if d_lex_ndcg is not None else None):>8} "
                  f"{fmt(pert_lex_mrr):>8} {fmt_delta(-d_lex_mrr if d_lex_mrr is not None else None):>8} "
                  f"{fmt(pert_smt_ndcg):>8} {fmt_delta(-d_smt_ndcg if d_smt_ndcg is not None else None):>8} "
                  f"{fmt(pert_smt_mrr):>8} {fmt_delta(-d_smt_mrr if d_smt_mrr is not None else None):>8}")

        print("-" * 90)


def print_table2(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 2: Planner stability and sensitivity to plan corruption."""
    print("\n" + "=" * 100)
    print("TABLE 2: Planner stability and sensitivity to plan corruption")
    print("=" * 100)
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'CandOvlp':>9} {'PlanInt':>9} "
          f"{'SeqGain-MRR':>12} {'SeqGain-NDCG':>13} "
          f"{'SwapDrp-MRR':>12} {'SwapDrp-NDCG':>13}")
    print("-" * 100)

    for split in splits:
        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            cand_overlap = r.get("CandOverlap@100")
            plan_intersect = r.get("PlanIntersect@100")
            sg_mrr = r.get("SeqGain_MRR@10")
            sg_ndcg = r.get("SeqGain_NDCG@10")
            psd_mrr = r.get("PlanSwapDrop_MRR@10")
            psd_ndcg = r.get("PlanSwapDrop_NDCG@10")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"{split:<8} {attack_label:<14} "
                  f"{fmt(cand_overlap):>9} {fmt(plan_intersect):>9} "
                  f"{fmt_delta(sg_mrr):>12} {fmt_delta(sg_ndcg):>13} "
                  f"{fmt_delta(psd_mrr):>12} {fmt_delta(psd_ndcg):>13}")

        print("-" * 100)


def print_latex_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print LaTeX-formatted Table 1 values."""
    print("\n% LaTeX Table 1 values (copy into your table)")
    print("% Format: NDCG@10 & Delta & MRR@10 & Delta & NDCG@10 & Delta & MRR@10 & Delta")

    for split in splits:
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        cn = clean.get("clean_lex_NDCG@10")
        cm = clean.get("clean_lex_MRR@10")
        csn = clean.get("clean_smt_NDCG@10")
        csm = clean.get("clean_smt_MRR@10")

        split_label = {"dl19": "DL19", "dl20": "DL20", "dev": "Dev"}[split]
        print(f"% {split_label} Clean")
        print(f"  & Clean & {fmt(cn)} & -- & {fmt(cm)} & -- & {fmt(csn)} & -- & {fmt(csm)} & -- \\\\")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]
            pn = r.get("pert_lex_NDCG@10")
            pm = r.get("pert_lex_MRR@10")
            psn = r.get("pert_smt_NDCG@10")
            psm = r.get("pert_smt_MRR@10")

            dn = (cn - pn) if (cn is not None and pn is not None) else None
            dm = (cm - pm) if (cm is not None and pm is not None) else None
            dsn = (csn - psn) if (csn is not None and psn is not None) else None
            dsm = (csm - psm) if (csm is not None and psm is not None) else None

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"% {split_label} {attack_label}")
            print(f"  & {attack_label} & {fmt(pn)} & {fmt(dn)} & {fmt(pm)} & {fmt(dm)} "
                  f"& {fmt(psn)} & {fmt(dsn)} & {fmt(psm)} & {fmt(dsm)} \\\\")


def print_latex_table2(agg: Dict, splits: List[str], attacks: List[str]):
    """Print LaTeX-formatted Table 2 values."""
    print("\n% LaTeX Table 2 values (copy into your table)")
    print("% Format: CandOverlap@100 & PlanIntersect@100 & SeqGain(MRR/NDCG) & PlanSwapDrop(MRR/NDCG)")

    for split in splits:
        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            co = r.get("CandOverlap@100")
            pi = r.get("PlanIntersect@100")
            sg_m = r.get("SeqGain_MRR@10")
            sg_n = r.get("SeqGain_NDCG@10")
            psd_m = r.get("PlanSwapDrop_MRR@10")
            psd_n = r.get("PlanSwapDrop_NDCG@10")

            split_label = {"dl19": "DL19", "dl20": "DL20", "dev": "Dev"}[split]
            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()

            sg_str = f"{fmt_delta(sg_m)}/{fmt_delta(sg_n)}"
            psd_str = f"{fmt_delta(psd_m)}/{fmt_delta(psd_n)}"

            print(f"% {split_label} {attack_label}")
            print(f"  & {attack_label} & {fmt(co)} & {fmt(pi)} & {sg_str} & {psd_str} \\\\")


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ2 results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "RQ2_robustness"),
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

    csv_path = os.path.join(args.results_dir, "summary.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run rq2.py first.")
        sys.exit(1)

    rows = load_summary_csv(csv_path)
    print(f"Loaded {len(rows)} result rows from {csv_path}")

    # Filter to requested attacks
    rows = [r for r in rows if r["attack_method"] in args.attacks]

    agg = aggregate_over_seeds(rows)
    print(f"Aggregated into {len(agg)} (split, attack) groups")

    print_table1(agg, args.splits, args.attacks)
    print_table2(agg, args.splits, args.attacks)

    if args.latex:
        print_latex_table1(agg, args.splits, args.attacks)
        print_latex_table2(agg, args.splits, args.attacks)


if __name__ == "__main__":
    main()

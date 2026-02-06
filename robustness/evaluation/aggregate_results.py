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
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def safe_std(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    mu = sum(valid) / len(valid)
    return math.sqrt(sum((x - mu) ** 2 for x in valid) / (len(valid) - 1))


def fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "--"
    return f"{val:.{digits}f}"


def fmt_pm(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    """Format as 'mean +/- std'."""
    if mean is None:
        return "--"
    m = f"{mean:.{digits}f}"
    if std is None:
        return m
    return f"{m}\u00b1{std:.{digits}f}"


def fmt_delta(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "--"
    return f"{val:+.{digits}f}" if val != 0 else f"{val:.{digits}f}"


def fmt_delta_pm(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    """Format as '+mean +/- std' (signed mean)."""
    if mean is None:
        return "--"
    m = f"{mean:+.{digits}f}" if mean != 0 else f"{mean:.{digits}f}"
    if std is None:
        return m
    return f"{m}\u00b1{std:.{digits}f}"


def fmt_pct_pm(mean: Optional[float], std: Optional[float]) -> str:
    """Format a rate as 'mean% +/- std%'."""
    if mean is None:
        return "--"
    m = f"{mean:.1%}"
    if std is None:
        return m
    return f"{m}\u00b1{std:.1%}"


def aggregate_over_seeds(rows: List[Dict]) -> Dict:
    """
    Group rows by (split, attack_method) and compute mean and std of
    numeric columns over seeds.

    For each numeric key ``k``, the aggregated row stores:
      - ``k``       : mean over seeds  (backward-compatible)
      - ``k__std``  : sample std over seeds
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["split"], r["attack_method"])
        groups[key].append(r)

    agg = {}
    for key, group in groups.items():
        agg_row = {"split": key[0], "attack_method": key[1], "n_seeds": len(group)}
        numeric_keys = [k for k in group[0] if k not in ("split", "attack_method", "seed")]
        for nk in numeric_keys:
            vals = [r[nk] for r in group if r[nk] is not None]
            agg_row[nk] = safe_mean(vals)
            agg_row[f"{nk}__std"] = safe_std(vals)
        agg[key] = agg_row
    return agg


def _get(r: Dict, key: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, std) for a key from an aggregated row."""
    return r.get(key), r.get(f"{key}__std")


def print_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 1: Retrieval performance under query perturbations (mean +/- std)."""
    W = 15  # column width to fit mean±std
    print("\n" + "=" * 150)
    print("TABLE 1: Retrieval performance under query perturbations  (mean \u00b1 std over seeds)")
    print("=" * 150)
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'S1-NDCG':>{W}} {'Delta':>{W}} {'S1-MRR':>{W}} {'Delta':>{W}} "
          f"{'S2-NDCG':>{W}} {'Delta':>{W}} {'S2-MRR':>{W}} {'Delta':>{W}}")
    print("-" * 150)

    for split in splits:
        # Clean baseline
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        cl_n, cl_n_s = _get(clean, "clean_lex_NDCG@10")
        cl_m, cl_m_s = _get(clean, "clean_lex_MRR@10")
        cs_n, cs_n_s = _get(clean, "clean_smt_NDCG@10")
        cs_m, cs_m_s = _get(clean, "clean_smt_MRR@10")

        print(f"{split:<8} {'Clean':<14} "
              f"{fmt_pm(cl_n, cl_n_s):>{W}} {'--':>{W}} "
              f"{fmt_pm(cl_m, cl_m_s):>{W}} {'--':>{W}} "
              f"{fmt_pm(cs_n, cs_n_s):>{W}} {'--':>{W}} "
              f"{fmt_pm(cs_m, cs_m_s):>{W}} {'--':>{W}}")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            pln, pln_s = _get(r, "pert_lex_NDCG@10")
            plm, plm_s = _get(r, "pert_lex_MRR@10")
            psn, psn_s = _get(r, "pert_smt_NDCG@10")
            psm, psm_s = _get(r, "pert_smt_MRR@10")

            # delta_lex and delta_smt are already in the CSV from rq2.py
            dln, dln_s = _get(r, "delta_lex_NDCG@10")
            dlm, dlm_s = _get(r, "delta_lex_MRR@10")
            dsn, dsn_s = _get(r, "delta_smt_NDCG@10")
            dsm, dsm_s = _get(r, "delta_smt_MRR@10")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"{'':8} {attack_label:<14} "
                  f"{fmt_pm(pln, pln_s):>{W}} {fmt_delta_pm(dln, dln_s):>{W}} "
                  f"{fmt_pm(plm, plm_s):>{W}} {fmt_delta_pm(dlm, dlm_s):>{W}} "
                  f"{fmt_pm(psn, psn_s):>{W}} {fmt_delta_pm(dsn, dsn_s):>{W}} "
                  f"{fmt_pm(psm, psm_s):>{W}} {fmt_delta_pm(dsm, dsm_s):>{W}}")

        print("-" * 150)


def print_table2(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 2: Planner stability and sensitivity (mean +/- std)."""
    W = 15
    print("\n" + "=" * 170)
    print("TABLE 2: Planner stability and sensitivity to plan corruption  (mean \u00b1 std over seeds)")
    print("=" * 170)
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'CandOvlp':>{W}} {'PlanInt':>{W}} "
          f"{'TokOvlp/l':>{W}} "
          f"{'SeqGain-MRR':>{W}} {'SeqGain-NDCG':>{W}} "
          f"{'SwapDrp-MRR':>{W}} {'SwapDrp-NDCG':>{W}} "
          f"{'Collps%':>{W}}")
    print("-" * 170)

    for split in splits:
        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            co, co_s = _get(r, "CandOverlap@100")
            pi, pi_s = _get(r, "PlanIntersect@100")
            to, to_s = _get(r, "TokOverlapAtEll@100_mean")
            sg_m, sg_m_s = _get(r, "SeqGain_MRR@10")
            sg_n, sg_n_s = _get(r, "SeqGain_NDCG@10")
            psd_m, psd_m_s = _get(r, "PlanSwapDrop_MRR@10")
            psd_n, psd_n_s = _get(r, "PlanSwapDrop_NDCG@10")
            cr, cr_s = _get(r, "collapse_rate")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"{split:<8} {attack_label:<14} "
                  f"{fmt_pm(co, co_s):>{W}} {fmt_pm(pi, pi_s):>{W}} "
                  f"{fmt_pm(to, to_s):>{W}} "
                  f"{fmt_delta_pm(sg_m, sg_m_s):>{W}} {fmt_delta_pm(sg_n, sg_n_s):>{W}} "
                  f"{fmt_delta_pm(psd_m, psd_m_s):>{W}} {fmt_delta_pm(psd_n, psd_n_s):>{W}} "
                  f"{fmt_pct_pm(cr, cr_s):>{W}}")

        print("-" * 170)


def print_table3(agg: Dict, splits: List[str], attacks: List[str]):
    """Print Table 3: Stability distributional statistics (mean +/- std over seeds)."""
    W = 15
    stats = ["mean", "median", "p10", "p25", "p75", "p90"]
    header_w = W * len(stats) + len(stats) - 1
    print("\n" + "=" * (22 + 2 * header_w + 2))
    print("TABLE 3: Stability distributional statistics  (mean \u00b1 std over seeds)")
    print("=" * (22 + 2 * header_w + 2))
    print(f"{'Split':<8} {'Perturbation':<14} "
          f"{'--- CandOverlap@100 ---':^{header_w}s}  "
          f"{'--- TokJaccard@100 ---':^{header_w}s}")
    print(f"{'':8} {'':14} "
          + " ".join(f"{s:>{W}}" for s in stats)
          + "  "
          + " ".join(f"{s:>{W}}" for s in stats))
    print("-" * (22 + 2 * header_w + 2))

    for split in splits:
        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()

            co_str = " ".join(
                f"{fmt_pm(*_get(r, f'CandOverlap@100_{s}')):>{W}}" for s in stats
            )
            tj_str = " ".join(
                f"{fmt_pm(*_get(r, f'TokJaccard@100_{s}')):>{W}}" for s in stats
            )

            print(f"{split:<8} {attack_label:<14} {co_str}  {tj_str}")

        print("-" * (22 + 2 * header_w + 2))


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


def _latex_pct_pm(mean: Optional[float], std: Optional[float]) -> str:
    """Format a rate as 'mean%$\\pm$std%' for LaTeX."""
    if mean is None:
        return "--"
    m = f"{mean:.1%}"
    if std is None:
        return m
    return f"{m}$\\pm${std:.1%}"


def print_latex_table1(agg: Dict, splits: List[str], attacks: List[str]):
    """Print LaTeX-formatted Table 1 values (mean $\\pm$ std)."""
    print("\n% LaTeX Table 1 values (mean $\\pm$ std over seeds)")
    print("% Format: NDCG@10 & Delta & MRR@10 & Delta & NDCG@10 & Delta & MRR@10 & Delta")

    for split in splits:
        clean_key = (split, attacks[0])
        if clean_key not in agg:
            continue
        clean = agg[clean_key]
        cl_n, cl_n_s = _get(clean, "clean_lex_NDCG@10")
        cl_m, cl_m_s = _get(clean, "clean_lex_MRR@10")
        cs_n, cs_n_s = _get(clean, "clean_smt_NDCG@10")
        cs_m, cs_m_s = _get(clean, "clean_smt_MRR@10")

        split_label = {"dl19": "DL19", "dl20": "DL20", "dev": "Dev"}[split]
        print(f"% {split_label} Clean")
        print(f"  & Clean & {_latex_pm(cl_n, cl_n_s)} & -- "
              f"& {_latex_pm(cl_m, cl_m_s)} & -- "
              f"& {_latex_pm(cs_n, cs_n_s)} & -- "
              f"& {_latex_pm(cs_m, cs_m_s)} & -- \\\\")

        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]
            pln, pln_s = _get(r, "pert_lex_NDCG@10")
            plm, plm_s = _get(r, "pert_lex_MRR@10")
            psn, psn_s = _get(r, "pert_smt_NDCG@10")
            psm, psm_s = _get(r, "pert_smt_MRR@10")

            dln, dln_s = _get(r, "delta_lex_NDCG@10")
            dlm, dlm_s = _get(r, "delta_lex_MRR@10")
            dsn, dsn_s = _get(r, "delta_smt_NDCG@10")
            dsm, dsm_s = _get(r, "delta_smt_MRR@10")

            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()
            print(f"% {split_label} {attack_label}")
            print(f"  & {attack_label} "
                  f"& {_latex_pm(pln, pln_s)} & {_latex_delta_pm(dln, dln_s)} "
                  f"& {_latex_pm(plm, plm_s)} & {_latex_delta_pm(dlm, dlm_s)} "
                  f"& {_latex_pm(psn, psn_s)} & {_latex_delta_pm(dsn, dsn_s)} "
                  f"& {_latex_pm(psm, psm_s)} & {_latex_delta_pm(dsm, dsm_s)} \\\\")


def print_latex_table2(agg: Dict, splits: List[str], attacks: List[str]):
    """Print LaTeX-formatted Table 2 values (mean $\\pm$ std)."""
    print("\n% LaTeX Table 2 values (mean $\\pm$ std over seeds)")
    print("% Format: CandOverlap@100 & TokJaccard@100 & TokOverlap/ell"
          " & SeqGain(MRR) & SeqGain(NDCG) & PlanSwapDrop(MRR) & PlanSwapDrop(NDCG) & CollapseRate")

    for split in splits:
        for attack in attacks:
            key = (split, attack)
            if key not in agg:
                continue
            r = agg[key]

            co, co_s = _get(r, "CandOverlap@100")
            pi, pi_s = _get(r, "PlanIntersect@100")
            to, to_s = _get(r, "TokOverlapAtEll@100_mean")
            sg_m, sg_m_s = _get(r, "SeqGain_MRR@10")
            sg_n, sg_n_s = _get(r, "SeqGain_NDCG@10")
            psd_m, psd_m_s = _get(r, "PlanSwapDrop_MRR@10")
            psd_n, psd_n_s = _get(r, "PlanSwapDrop_NDCG@10")
            cr, cr_s = _get(r, "collapse_rate")

            split_label = {"dl19": "DL19", "dl20": "DL20", "dev": "Dev"}[split]
            attack_label = "Misspelling" if attack == "mispelling" else attack.capitalize()

            print(f"% {split_label} {attack_label}")
            print(f"  & {attack_label} "
                  f"& {_latex_pm(co, co_s)} & {_latex_pm(pi, pi_s)} & {_latex_pm(to, to_s)} "
                  f"& {_latex_delta_pm(sg_m, sg_m_s)} & {_latex_delta_pm(sg_n, sg_n_s)} "
                  f"& {_latex_delta_pm(psd_m, psd_m_s)} & {_latex_delta_pm(psd_n, psd_n_s)} "
                  f"& {_latex_pct_pm(cr, cr_s)} \\\\")


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

    # Load per-split summary CSVs produced by rq2.py (summary_dl19.csv, etc.)
    # Fall back to a single summary.csv if per-split files don't exist.
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
            print(f"Error: No summary CSV found in {args.results_dir}. Run rq2.py first.")
            sys.exit(1)
        rows = load_summary_csv(csv_path)
        print(f"Loaded {len(rows)} result rows from {csv_path}")

    # Filter to requested attacks
    rows = [r for r in rows if r["attack_method"] in args.attacks]

    agg = aggregate_over_seeds(rows)
    print(f"Aggregated into {len(agg)} (split, attack) groups")

    print_table1(agg, args.splits, args.attacks)
    print_table2(agg, args.splits, args.attacks)
    print_table3(agg, args.splits, args.attacks)

    if args.latex:
        print_latex_table1(agg, args.splits, args.attacks)
        print_latex_table2(agg, args.splits, args.attacks)


if __name__ == "__main__":
    main()

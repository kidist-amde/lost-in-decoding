#!/usr/bin/env python3
"""
Aggregate and analyze RQ3 cross-lingual evaluation results.

This script:
1. Loads results from all language/split experiments
2. Computes summary statistics across languages
3. Generates LaTeX tables for the paper
4. Creates analysis figures (optional)

Usage
-----
    python -m cross_lingual.evaluation.aggregate_results \\
        --results_dir experiments/RQ3_crosslingual

    # Generate LaTeX tables only
    python -m cross_lingual.evaluation.aggregate_results \\
        --results_dir experiments/RQ3_crosslingual \\
        --latex_only
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGES = ["fr", "de", "zh", "nl"]
SPLITS = ["dl19", "dl20", "dev"]

LANGUAGE_NAMES = {
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "nl": "Dutch",
}

SPLIT_NAMES = {
    "dl19": "DL19",
    "dl20": "DL20",
    "dev": "Dev",
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> List[Dict]:
    """Load all summary.json files from results directory."""
    results = []
    results_path = Path(results_dir)

    # Try loading main summary
    summary_path = results_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
        print(f"[aggregate] Loaded {len(results)} results from {summary_path}")
        return results

    # Try loading per-language summaries
    for lang in LANGUAGES:
        for split in SPLITS:
            summary_path = results_path / f"summary_{lang}_{split}.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    lang_results = json.load(f)
                results.extend(lang_results)

    print(f"[aggregate] Loaded {len(results)} results total")
    return results


def load_per_split_csvs(results_dir: str) -> Dict[str, List[Dict]]:
    """Load per-split CSV summaries."""
    results = {}
    results_path = Path(results_dir)

    for split in SPLITS:
        csv_path = results_path / f"summary_{split}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                results[split] = list(reader)

    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregates(results: List[Dict]) -> Dict:
    """Compute aggregate statistics across all results."""
    aggregates = {}

    # Group by language
    by_language = defaultdict(list)
    for r in results:
        by_language[r.get("language")].append(r)

    # Group by split
    by_split = defaultdict(list)
    for r in results:
        by_split[r.get("split")].append(r)

    # Per-language averages
    for lang, lang_results in by_language.items():
        lang_agg = {}

        for metric in [
            "naive_smt_NDCG@10", "naive_smt_MRR@10",
            "seq_only_NDCG@10", "seq_only_MRR@10",
            "translate_smt_NDCG@10", "translate_smt_MRR@10",
            "swap_smt_NDCG@10", "swap_smt_MRR@10",
            "delta_simul_NDCG@10", "delta_simul_MRR@10",
            "delta_pag_NDCG@10", "delta_pag_MRR@10",
            "trans_gain_NDCG@10", "trans_gain_MRR@10",
            "swap_gain_NDCG@10", "swap_gain_MRR@10",
        ]:
            values = []
            for r in lang_results:
                # Extract from nested dicts
                val = _extract_metric(r, metric)
                if val is not None:
                    values.append(val)

            if values:
                lang_agg[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        aggregates[f"by_language_{lang}"] = lang_agg

    # Per-split averages (across languages)
    for split, split_results in by_split.items():
        split_agg = {}

        for metric in [
            "naive_smt_NDCG@10", "naive_smt_MRR@10",
            "delta_pag_NDCG@10", "delta_pag_MRR@10",
        ]:
            values = []
            for r in split_results:
                val = _extract_metric(r, metric)
                if val is not None:
                    values.append(val)

            if values:
                split_agg[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        aggregates[f"by_split_{split}"] = split_agg

    # Overall averages
    overall = {}
    for metric in [
        "delta_pag_NDCG@10", "delta_pag_MRR@10",
        "trans_gain_NDCG@10", "trans_gain_MRR@10",
        "swap_gain_NDCG@10", "swap_gain_MRR@10",
    ]:
        values = []
        for r in results:
            val = _extract_metric(r, metric)
            if val is not None:
                values.append(val)

        if values:
            overall[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    aggregates["overall"] = overall

    return aggregates


def _extract_metric(result: Dict, metric_name: str) -> Optional[float]:
    """Extract a metric value from a result dict, handling nested structures."""
    # Direct key
    if metric_name in result:
        val = result[metric_name]
        if val is not None:
            return float(val)
        return None

    # Try extracting from nested metric dicts
    parts = metric_name.split("_")
    if len(parts) >= 2:
        prefix = "_".join(parts[:-1])
        suffix = parts[-1]

        # e.g., naive_smt -> naive_smt_metrics, NDCG@10
        key = f"{prefix}_metrics"
        if key in result:
            return result[key].get(suffix)

    return None


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_main_results_table(results: List[Dict]) -> str:
    """
    Generate LaTeX table for main RQ3 results (Table 3 in paper).

    Columns: Language | Split | En-PAG | Naive-PAG | Δ | Seq-Only | Trans-PAG | Swap-PAG
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{RQ3 Cross-lingual evaluation. En-PAG: English queries (upper bound). "
        r"Naive-PAG: non-English queries. Seq-Only: planner disabled. "
        r"Trans-PAG: translated queries. Swap-PAG: non-English query + English plan.}",
        r"\label{tab:rq3-main}",
        r"\begin{tabular}{ll|cc|c|cc|c}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{En-PAG} & Naive & Seq & Trans & Swap \\",
        r"Lang & Split & N@10 & M@10 & Δ N@10 & N@10 & Δ N@10 & Δ N@10 \\",
        r"\midrule",
    ]

    # Sort results by language, then split
    sorted_results = sorted(
        results,
        key=lambda r: (
            LANGUAGES.index(r.get("language", "")) if r.get("language") in LANGUAGES else 99,
            SPLITS.index(r.get("split", "")) if r.get("split") in SPLITS else 99,
        )
    )

    current_lang = None
    for r in sorted_results:
        lang = r.get("language", "")
        split = r.get("split", "")

        # Language name (only on first row for language)
        if lang != current_lang:
            lang_display = LANGUAGE_NAMES.get(lang, lang)
            current_lang = lang
        else:
            lang_display = ""

        split_display = SPLIT_NAMES.get(split, split)

        # Extract metrics
        en_ndcg = r.get("english_smt_metrics", {}).get("NDCG@10", 0) * 100
        en_mrr = r.get("english_smt_metrics", {}).get("MRR@10", 0) * 100
        naive_ndcg = r.get("naive_smt_metrics", {}).get("NDCG@10", 0) * 100
        delta_ndcg = naive_ndcg - en_ndcg
        seq_ndcg = r.get("seq_only_metrics", {}).get("NDCG@10", 0) * 100
        trans_ndcg = r.get("translate_smt_metrics", {}).get("NDCG@10", 0) * 100
        trans_delta = trans_ndcg - naive_ndcg
        swap_ndcg = r.get("swap_smt_metrics", {}).get("NDCG@10", 0) * 100
        swap_delta = swap_ndcg - naive_ndcg

        # Format with sign for deltas
        delta_str = f"{delta_ndcg:+.1f}" if delta_ndcg != 0 else "0.0"
        trans_delta_str = f"{trans_delta:+.1f}" if trans_delta != 0 else "0.0"
        swap_delta_str = f"{swap_delta:+.1f}" if swap_delta != 0 else "0.0"

        lines.append(
            f"{lang_display} & {split_display} & "
            f"{en_ndcg:.1f} & {en_mrr:.1f} & "
            f"{delta_str} & "
            f"{seq_ndcg:.1f} & "
            f"{trans_delta_str} & "
            f"{swap_delta_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_planner_diagnostics_table(results: List[Dict]) -> str:
    """
    Generate LaTeX table for planner diagnostics (Table 4 in paper).

    Shows: Language | Split | Cand-Recall@100 (En/Tgt/Δ) | SimulOnly-N@10 (En/Tgt/Δ) | Tok-Jaccard
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{RQ3 Planner diagnostics. Cand-Recall: fraction of relevant docs in planner's top-100. "
        r"SimulOnly: planner-only NDCG@10. Tok-Jaccard: token-plan overlap between English and target.}",
        r"\label{tab:rq3-planner}",
        r"\begin{tabular}{ll|ccc|ccc|c}",
        r"\toprule",
        r"& & \multicolumn{3}{c|}{Cand-Recall@100} & \multicolumn{3}{c|}{SimulOnly-N@10} & Tok \\",
        r"Lang & Split & En & Tgt & Δ & En & Tgt & Δ & Jacc \\",
        r"\midrule",
    ]

    sorted_results = sorted(
        results,
        key=lambda r: (
            LANGUAGES.index(r.get("language", "")) if r.get("language") in LANGUAGES else 99,
            SPLITS.index(r.get("split", "")) if r.get("split") in SPLITS else 99,
        )
    )

    current_lang = None
    for r in sorted_results:
        lang = r.get("language", "")
        split = r.get("split", "")

        if lang != current_lang:
            lang_display = LANGUAGE_NAMES.get(lang, lang)
            current_lang = lang
        else:
            lang_display = ""

        split_display = SPLIT_NAMES.get(split, split)

        # Extract planner diagnostics
        cr = r.get("candidate_recall", {})
        en_cr = cr.get("english", {}).get("aggregate", {}).get("recall@100", 0) * 100
        tgt_cr = cr.get("target", {}).get("aggregate", {}).get("recall@100", 0) * 100
        delta_cr = cr.get("aggregate_delta", {}).get("delta_recall@100", 0) * 100

        so = r.get("simul_only_comparison", {})
        en_so = so.get("english", {}).get("SimulOnly_NDCG@10", 0) * 100
        tgt_so = so.get("target", {}).get("SimulOnly_NDCG@10", 0) * 100
        delta_so = so.get("aggregate_delta", {}).get("delta_SimulOnly_NDCG@10", 0) * 100

        tok = r.get("token_overlap", {}).get("aggregate", {}).get("mean_jaccard", 0) * 100

        delta_cr_str = f"{delta_cr:+.1f}" if delta_cr != 0 else "0.0"
        delta_so_str = f"{delta_so:+.1f}" if delta_so != 0 else "0.0"

        lines.append(
            f"{lang_display} & {split_display} & "
            f"{en_cr:.1f} & {tgt_cr:.1f} & {delta_cr_str} & "
            f"{en_so:.1f} & {tgt_so:.1f} & {delta_so_str} & "
            f"{tok:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_pag_gain_table(results: List[Dict]) -> str:
    """
    Generate LaTeX table for PAG gain analysis.

    Shows how PAG gain (Stage2 - Stage1) changes across languages.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{PAG gain comparison. PAG Gain = Stage2 NDCG@10 - Stage1 NDCG@10. "
        r"Positive gain means the sequential decoder improves over planner-only.}",
        r"\label{tab:rq3-pag-gain}",
        r"\begin{tabular}{ll|cc|c}",
        r"\toprule",
        r"& & English & Target & \\",
        r"Lang & Split & PAG-Gain & PAG-Gain & Δ Gain \\",
        r"\midrule",
    ]

    sorted_results = sorted(
        results,
        key=lambda r: (
            LANGUAGES.index(r.get("language", "")) if r.get("language") in LANGUAGES else 99,
            SPLITS.index(r.get("split", "")) if r.get("split") in SPLITS else 99,
        )
    )

    current_lang = None
    for r in sorted_results:
        lang = r.get("language", "")
        split = r.get("split", "")

        if lang != current_lang:
            lang_display = LANGUAGE_NAMES.get(lang, lang)
            current_lang = lang
        else:
            lang_display = ""

        split_display = SPLIT_NAMES.get(split, split)

        en_gain = r.get("english_pag_gain", {}).get("PAG_Gain_NDCG@10", 0) * 100
        tgt_gain = r.get("naive_pag_gain", {}).get("PAG_Gain_NDCG@10", 0) * 100
        delta_gain = r.get("pag_gain_delta", {}).get("PAG_Gain_NDCG@10", 0) * 100

        delta_str = f"{delta_gain:+.1f}" if delta_gain != 0 else "0.0"

        lines.append(
            f"{lang_display} & {split_display} & "
            f"{en_gain:+.1f} & {tgt_gain:+.1f} & {delta_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_summary_by_language_table(results: List[Dict]) -> str:
    """Generate summary table aggregated by language."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{RQ3 summary by language (averaged across splits). "
        r"Δ = change relative to English baseline or Naive-PAG.}",
        r"\label{tab:rq3-summary}",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{Naive vs En} & \multicolumn{2}{c|}{Trans vs Naive} & \multicolumn{2}{c}{Swap vs Naive} \\",
        r"Language & Δ N@10 & Δ M@10 & Δ N@10 & Δ M@10 & Δ N@10 & Δ M@10 \\",
        r"\midrule",
    ]

    # Aggregate by language
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r.get("language")].append(r)

    for lang in LANGUAGES:
        if lang not in by_lang:
            continue

        lang_results = by_lang[lang]

        # Compute averages
        delta_naive_ndcg = []
        delta_naive_mrr = []
        delta_trans_ndcg = []
        delta_trans_mrr = []
        delta_swap_ndcg = []
        delta_swap_mrr = []

        for r in lang_results:
            en_ndcg = r.get("english_smt_metrics", {}).get("NDCG@10", 0)
            en_mrr = r.get("english_smt_metrics", {}).get("MRR@10", 0)
            naive_ndcg = r.get("naive_smt_metrics", {}).get("NDCG@10", 0)
            naive_mrr = r.get("naive_smt_metrics", {}).get("MRR@10", 0)
            trans_ndcg = r.get("translate_smt_metrics", {}).get("NDCG@10", 0)
            trans_mrr = r.get("translate_smt_metrics", {}).get("MRR@10", 0)
            swap_ndcg = r.get("swap_smt_metrics", {}).get("NDCG@10", 0)
            swap_mrr = r.get("swap_smt_metrics", {}).get("MRR@10", 0)

            if en_ndcg and naive_ndcg:
                delta_naive_ndcg.append(naive_ndcg - en_ndcg)
                delta_naive_mrr.append(naive_mrr - en_mrr)

            if naive_ndcg and trans_ndcg:
                delta_trans_ndcg.append(trans_ndcg - naive_ndcg)
                delta_trans_mrr.append(trans_mrr - naive_mrr)

            if naive_ndcg and swap_ndcg:
                delta_swap_ndcg.append(swap_ndcg - naive_ndcg)
                delta_swap_mrr.append(swap_mrr - naive_mrr)

        # Format averages
        def fmt(vals):
            if not vals:
                return "--"
            return f"{np.mean(vals)*100:+.1f}"

        lines.append(
            f"{LANGUAGE_NAMES[lang]} & "
            f"{fmt(delta_naive_ndcg)} & {fmt(delta_naive_mrr)} & "
            f"{fmt(delta_trans_ndcg)} & {fmt(delta_trans_mrr)} & "
            f"{fmt(delta_swap_ndcg)} & {fmt(delta_swap_mrr)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate RQ3 cross-lingual evaluation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "RQ3_crosslingual"),
        help="Directory containing RQ3 results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: results_dir)",
    )
    parser.add_argument(
        "--latex_only",
        action="store_true",
        help="Only generate LaTeX tables, skip other analysis",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_all_results(args.results_dir)
    if not results:
        print(f"[aggregate] No results found in {args.results_dir}")
        return

    # Compute aggregates
    aggregates = compute_aggregates(results)

    # Save aggregates
    agg_path = os.path.join(output_dir, "aggregates.json")
    with open(agg_path, "w") as f:
        json.dump(aggregates, f, indent=2, default=float)
    print(f"[aggregate] Wrote {agg_path}")

    # Generate LaTeX tables
    tables = {
        "table_main_results.tex": generate_main_results_table(results),
        "table_planner_diagnostics.tex": generate_planner_diagnostics_table(results),
        "table_pag_gain.tex": generate_pag_gain_table(results),
        "table_summary_by_language.tex": generate_summary_by_language_table(results),
    }

    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    for filename, content in tables.items():
        path = os.path.join(latex_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"[aggregate] Wrote {path}")

    # Print summary
    print("\n" + "="*70)
    print("RQ3 Results Summary")
    print("="*70)

    for lang in LANGUAGES:
        lang_results = [r for r in results if r.get("language") == lang]
        if not lang_results:
            continue

        print(f"\n{LANGUAGE_NAMES[lang]}:")
        for r in lang_results:
            split = r.get("split", "")
            en_ndcg = r.get("english_smt_metrics", {}).get("NDCG@10", 0) * 100
            naive_ndcg = r.get("naive_smt_metrics", {}).get("NDCG@10", 0) * 100
            trans_ndcg = r.get("translate_smt_metrics", {}).get("NDCG@10", 0) * 100
            swap_ndcg = r.get("swap_smt_metrics", {}).get("NDCG@10", 0) * 100

            print(f"  {SPLIT_NAMES.get(split, split):5s}: "
                  f"En={en_ndcg:.1f}, Naive={naive_ndcg:.1f} ({naive_ndcg-en_ndcg:+.1f}), "
                  f"Trans={trans_ndcg:.1f} ({trans_ndcg-naive_ndcg:+.1f}), "
                  f"Swap={swap_ndcg:.1f} ({swap_ndcg-naive_ndcg:+.1f})")

    print(f"\n[aggregate] Done. Results in {output_dir}/")


if __name__ == "__main__":
    main()

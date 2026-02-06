"""
RQ3 data loader: mMARCO multilingual queries + MS MARCO qrels.

Languages: zh, nl, fr, de
Data source: HuggingFace unicamp-dl/mmarco (cached locally)
Qrels: MS MARCO dev_qrel.json (local)

Key features:
- Downloads and caches mMARCO queries per language
- Validates qid overlap between query sets and qrels
- Produces per-language coverage reports
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "msmarco-full"
MMARCO_ROOT = REPO_ROOT / "cross_lingual" / "data" / "mmarco"

MMARCO_HF_REPO = "unicamp-dl/mmarco"

# Languages for RQ3 cross-lingual evaluation
RQ3_LANGUAGES = ["zh", "nl", "fr", "de"]

# MS MARCO dev-small queries (the only split we use for RQ3)
DEV_QUERY_PATH = DATA_ROOT / "dev_queries" / "raw.tsv"
DEV_QREL_PATH = DATA_ROOT / "dev_qrel.json"

# ---------------------------------------------------------------------------
# English queries and qrels
# ---------------------------------------------------------------------------

def load_english_queries() -> Dict[str, str]:
    """Load English MS MARCO dev-small queries.

    Returns:
        {qid: query_text}
    """
    queries = {}
    with open(DEV_QUERY_PATH) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_qrels() -> Dict[str, Dict[str, int]]:
    """Load MS MARCO dev qrels.

    Returns:
        {qid: {pid: relevance}}
    """
    with open(DEV_QREL_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# mMARCO query download and loading
# ---------------------------------------------------------------------------

def download_mmarco_queries(language: str, force: bool = False) -> Path:
    """Download mMARCO queries for *language* from HuggingFace.

    Caches to cross_lingual/data/mmarco/{lang}/queries.tsv.
    Returns the directory containing the downloaded file.
    """
    if language not in RQ3_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. Available: {RQ3_LANGUAGES}"
        )

    output_dir = MMARCO_ROOT / language
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_file = output_dir / "queries.tsv"

    if queries_file.exists() and not force:
        print(f"[data_loader] {language} queries already cached at {queries_file}")
        return output_dir

    from datasets import load_dataset

    print(f"[data_loader] Downloading {language} queries from {MMARCO_HF_REPO}...")
    dataset = load_dataset(
        MMARCO_HF_REPO,
        f"queries-{language}",
        split="train",
        trust_remote_code=True,
    )

    with open(queries_file, "w") as f:
        for row in dataset:
            qid = str(row["id"])
            text = row["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")

    print(f"[data_loader] Saved {len(dataset)} queries -> {queries_file}")
    return output_dir


def load_mmarco_queries(
    language: str,
    download: bool = True,
) -> Dict[str, str]:
    """Load all mMARCO queries for *language*.

    Returns:
        {qid: query_text}
    """
    queries_file = MMARCO_ROOT / language / "queries.tsv"
    if not queries_file.exists():
        if download:
            download_mmarco_queries(language)
        else:
            raise FileNotFoundError(f"mMARCO queries not found: {queries_file}")

    queries = {}
    with open(queries_file) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


# ---------------------------------------------------------------------------
# QID validation and coverage reporting
# ---------------------------------------------------------------------------

def validate_qid_overlap(
    language: str,
    download: bool = True,
) -> Dict:
    """Validate qid overlap between mMARCO queries and qrels.

    Returns a coverage report dict:
        {
            "language": str,
            "n_mmarco_queries": int,
            "n_qrel_qids": int,
            "n_english_queries": int,
            "n_intersection": int,
            "n_missing_from_mmarco": int,
            "n_missing_from_qrels": int,
            "intersection_qids": [str, ...],
        }
    """
    mmarco_queries = load_mmarco_queries(language, download=download)
    english_queries = load_english_queries()
    qrels = load_qrels()

    mmarco_qids = set(mmarco_queries.keys())
    english_qids = set(english_queries.keys())
    qrel_qids = set(qrels.keys())

    # The evaluation set is the intersection of:
    # 1. mMARCO queries for this language
    # 2. English dev-small queries (so we have parallel queries)
    # 3. Qrels (so we can evaluate)
    intersection = mmarco_qids & english_qids & qrel_qids

    # Queries in english dev-small that are missing from mMARCO
    missing_from_mmarco = (english_qids & qrel_qids) - mmarco_qids
    # Queries in mMARCO that have no qrels
    missing_from_qrels = (mmarco_qids & english_qids) - qrel_qids

    report = {
        "language": language,
        "n_mmarco_queries": len(mmarco_qids),
        "n_english_queries": len(english_qids),
        "n_qrel_qids": len(qrel_qids),
        "n_intersection": len(intersection),
        "n_missing_from_mmarco": len(missing_from_mmarco),
        "n_missing_from_qrels": len(missing_from_qrels),
        "intersection_qids": sorted(intersection),
    }

    print(f"[data_loader] Coverage report for {language}:")
    print(f"  #mMARCO queries:     {report['n_mmarco_queries']}")
    print(f"  #English queries:    {report['n_english_queries']}")
    print(f"  #qrel qids:          {report['n_qrel_qids']}")
    print(f"  #intersection:       {report['n_intersection']}")
    print(f"  #missing from mMARCO:{report['n_missing_from_mmarco']}")
    print(f"  #missing from qrels: {report['n_missing_from_qrels']}")

    return report


def save_coverage_report(report: Dict, output_dir: Path) -> Path:
    """Save coverage report to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"coverage_{report['language']}.json"
    # Save without the full qid list for readability
    summary = {k: v for k, v in report.items() if k != "intersection_qids"}
    summary["sample_qids"] = report["intersection_qids"][:10]
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[data_loader] Coverage report saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Prepare query files for PAG pipeline
# ---------------------------------------------------------------------------

def get_evaluation_qids(language: str, download: bool = True) -> Set[str]:
    """Get the set of qids for evaluation (intersection of all three sets)."""
    report = validate_qid_overlap(language, download=download)
    return set(report["intersection_qids"])


def prepare_query_files(
    language: str,
    output_base: Path,
    download: bool = True,
) -> Tuple[Path, Path, int, Dict]:
    """Prepare English and target-language query TSV files for PAG.

    Only includes queries in the validated intersection (have both
    mMARCO translations and qrels).

    Returns:
        (english_query_dir, target_query_dir, n_queries, coverage_report)
    """
    report = validate_qid_overlap(language, download=download)
    eval_qids = set(report["intersection_qids"])

    english_queries = load_english_queries()
    mmarco_queries = load_mmarco_queries(language, download=False)

    # Directories compatible with PAG pipeline
    english_dir = output_base / "english"
    target_dir = output_base / language

    english_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write English queries (only intersection)
    with open(english_dir / "raw.tsv", "w") as f:
        for qid in sorted(eval_qids, key=int):
            text = english_queries[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")

    # Write target language queries (only intersection)
    with open(target_dir / "raw.tsv", "w") as f:
        for qid in sorted(eval_qids, key=int):
            text = mmarco_queries[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")

    n = len(eval_qids)
    print(f"[data_loader] Prepared {n} query pairs for {language}")
    print(f"  English: {english_dir}")
    print(f"  {language.upper()}: {target_dir}")

    return english_dir, target_dir, n, report


# ---------------------------------------------------------------------------
# Filtered qrels for evaluation
# ---------------------------------------------------------------------------

def get_filtered_qrels(eval_qids: Set[str]) -> Dict[str, Dict[str, int]]:
    """Return qrels filtered to only the evaluation qids."""
    qrels = load_qrels()
    return {qid: docs for qid, docs in qrels.items() if qid in eval_qids}


def save_filtered_qrels(eval_qids: Set[str], output_path: Path) -> Path:
    """Save filtered qrels to a JSON file for use by the PAG pipeline."""
    filtered = get_filtered_qrels(eval_qids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f)
    print(f"[data_loader] Saved {len(filtered)} filtered qrels -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RQ3 data loader")
    parser.add_argument(
        "--languages", nargs="+", default=RQ3_LANGUAGES,
        help=f"Languages to process (default: {RQ3_LANGUAGES})"
    )
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "cross_lingual" / "results"
    )

    for lang in args.languages:
        report = validate_qid_overlap(lang, download=args.download)
        save_coverage_report(report, output_base / "coverage")

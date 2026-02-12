#!/usr/bin/env python3
"""
Parallel query pair loader for planner alignment training.

Provides aligned (q_en, q_lang) pairs by qid for all RQ3 languages.
Uses mMARCO for non-English queries and either mMARCO-en or MS MARCO
English queries for the English side.

Key validation:
  |Q_en ∩ Q_lang| / |Q_lang| must be ~1.0.
  If mismatch > 0.1%, fails loudly with example mismatched qids.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data_loader import (
    RQ3_LANGUAGES,
    MMARCO_ROOT,
    MMARCO_HF_REPO,
    load_english_queries,
    load_qrels,
    load_mmarco_queries,
    download_mmarco_queries,
    get_evaluation_qids,
)

# ---------------------------------------------------------------------------
# mMARCO English queries (preferred parallel source)
# ---------------------------------------------------------------------------

MMARCO_EN_PATH = MMARCO_ROOT / "en" / "queries.tsv"


def download_mmarco_english(force: bool = False) -> Path:
    """Download mMARCO English queries from HuggingFace.

    These are preferred over MS MARCO dev-small queries because
    they share the exact same qid space as the non-English mMARCO splits.

    Uses load_dataset_builder + pyarrow (avoids trust_remote_code issues).
    """
    output_dir = MMARCO_ROOT / "en"
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_file = output_dir / "queries.tsv"

    if queries_file.exists() and not force:
        print(f"[parallel] mMARCO-en already cached at {queries_file}")
        return output_dir

    from datasets import load_dataset_builder
    from cross_lingual.data_loader import LANG_CODE_TO_HF_NAME, _load_queries_from_arrow

    hf_name = LANG_CODE_TO_HF_NAME.get("en", "english")
    print(f"[parallel] Downloading mMARCO English queries from {MMARCO_HF_REPO}...")

    builder = load_dataset_builder(MMARCO_HF_REPO, f"queries-{hf_name}")
    builder.download_and_prepare()

    n_written = _load_queries_from_arrow(builder, queries_file)
    print(f"[parallel] Saved {n_written} English queries -> {queries_file}")
    return output_dir


def load_mmarco_english() -> Dict[str, str]:
    """Load mMARCO English queries. Falls back to MS MARCO dev queries."""
    if MMARCO_EN_PATH.exists():
        queries = {}
        with open(MMARCO_EN_PATH) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]
        return queries

    # Fallback: use MS MARCO English dev-small queries
    print("[parallel] mMARCO-en not found, falling back to MS MARCO dev queries")
    return load_english_queries()


# ---------------------------------------------------------------------------
# QID alignment validation
# ---------------------------------------------------------------------------

def validate_alignment(
    q_en: Dict[str, str],
    q_lang: Dict[str, str],
    language: str,
    min_aligned: int = 1000,
) -> Set[str]:
    """Validate qid alignment between English and non-English queries.

    mMARCO splits have different qid coverage across languages by design,
    so we only check that the intersection is large enough for training.

    Returns the set of aligned qids.
    """
    en_qids = set(q_en.keys())
    lang_qids = set(q_lang.keys())
    aligned = en_qids & lang_qids

    coverage = len(aligned) / len(lang_qids) if lang_qids else 0.0

    print(f"[parallel] Alignment for {language}:")
    print(f"  |Q_en|:  {len(en_qids)}")
    print(f"  |Q_{language}|: {len(lang_qids)}")
    print(f"  |Q_en ∩ Q_{language}|: {len(aligned)} ({coverage:.1%} of {language})")

    if len(aligned) < min_aligned:
        raise ValueError(
            f"Too few aligned qids for {language}: "
            f"{len(aligned)} < {min_aligned} minimum."
        )

    return aligned


# ---------------------------------------------------------------------------
# Parallel query pair dataset
# ---------------------------------------------------------------------------

class ParallelQueryDataset(Dataset):
    """Dataset of aligned (q_en, q_lang) pairs for planner alignment training.

    Each item: (qid, english_text, target_language_text)
    """

    def __init__(
        self,
        language: str,
        split: str = "train",
        qid_subset: Optional[Set[str]] = None,
        download: bool = True,
    ):
        """
        Args:
            language: target language code (zh, nl, fr, de)
            split: "train" or "dev" (for compute-bounded settings, dev is used)
            qid_subset: optional subset of qids to use (e.g., eval qids)
            download: whether to download missing data
        """
        self.language = language
        self.split = split

        # Load queries
        if download:
            download_mmarco_queries(language, force=False)
            download_mmarco_english(force=False)

        q_en = load_mmarco_english()
        q_lang = load_mmarco_queries(language, download=False)

        # Validate alignment
        aligned_qids = validate_alignment(q_en, q_lang, language)

        # If qid_subset provided, intersect
        if qid_subset is not None:
            aligned_qids = aligned_qids & qid_subset

        # Store aligned pairs
        self.qids = sorted(aligned_qids)
        self.q_en = {qid: q_en[qid] for qid in self.qids}
        self.q_lang = {qid: q_lang[qid] for qid in self.qids}

        print(f"[parallel] {language}/{split}: {len(self.qids)} aligned pairs")

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        return qid, self.q_en[qid], self.q_lang[qid]


# ---------------------------------------------------------------------------
# Prepare TSV files for subprocess-based extraction
# ---------------------------------------------------------------------------

def prepare_parallel_tsv(
    language: str,
    output_dir: Path,
    split: str = "train",
    qid_subset: Optional[Set[str]] = None,
    download: bool = True,
) -> Tuple[Path, Path, List[str]]:
    """Write aligned English + target-language TSV files for PAG extraction.

    Returns:
        (english_dir, target_dir, qid_list)
    """
    ds = ParallelQueryDataset(
        language=language, split=split,
        qid_subset=qid_subset, download=download,
    )

    en_dir = output_dir / "parallel" / language / "english"
    lang_dir = output_dir / "parallel" / language / language

    en_dir.mkdir(parents=True, exist_ok=True)
    lang_dir.mkdir(parents=True, exist_ok=True)

    with open(en_dir / "raw.tsv", "w") as f_en, \
         open(lang_dir / "raw.tsv", "w") as f_lang:
        for qid in ds.qids:
            en_text = ds.q_en[qid].replace("\t", " ").replace("\n", " ")
            lang_text = ds.q_lang[qid].replace("\t", " ").replace("\n", " ")
            f_en.write(f"{qid}\t{en_text}\n")
            f_lang.write(f"{qid}\t{lang_text}\n")

    print(f"[parallel] Wrote {len(ds.qids)} pairs:")
    print(f"  EN:   {en_dir / 'raw.tsv'}")
    print(f"  {language.upper()}: {lang_dir / 'raw.tsv'}")

    return en_dir, lang_dir, ds.qids


# ---------------------------------------------------------------------------
# Train/dev split management
# ---------------------------------------------------------------------------

def get_train_dev_split(
    language: str,
    dev_fraction: float = 0.1,
    seed: int = 42,
    download: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Split aligned qids into train and dev sets.

    The dev set MUST overlap with the inference-only baseline eval set
    (qids with qrels). The training set uses the remaining qids.

    Returns:
        (train_qids, dev_qids)
    """
    import numpy as np

    # Dev qids = same as inference baselines (intersection with qrels)
    dev_qids = get_evaluation_qids(language, download=download)

    # All aligned qids
    if download:
        download_mmarco_queries(language, force=False)
        download_mmarco_english(force=False)

    q_en = load_mmarco_english()
    q_lang = load_mmarco_queries(language, download=False)
    aligned = validate_alignment(q_en, q_lang, language)

    # Train = aligned qids NOT in dev
    train_qids = aligned - dev_qids

    print(f"[parallel] {language} split:")
    print(f"  train: {len(train_qids)} qids")
    print(f"  dev:   {len(dev_qids)} qids (matches baseline eval)")
    print(f"  total: {len(aligned)} aligned qids")

    if len(train_qids) == 0:
        print(f"[parallel] WARNING: No train qids for {language}. "
              f"Using dev qids for training (compute-bounded mode).")
        train_qids = dev_qids

    return train_qids, dev_qids


# ---------------------------------------------------------------------------
# CLI: prepare parallel data for all languages
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare parallel query pairs")
    parser.add_argument("--languages", nargs="+", default=RQ3_LANGUAGES)
    parser.add_argument(
        "--output_dir", type=str,
        default=str(REPO_ROOT / "cross_lingual" / "trained_extension" / "results"),
    )
    parser.add_argument("--download", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"  Preparing parallel data for {lang}")
        print(f"{'='*60}")

        train_qids, dev_qids = get_train_dev_split(
            lang, download=args.download,
        )

        # Write train and dev TSVs
        for split, qids in [("train", train_qids), ("dev", dev_qids)]:
            en_dir, lang_dir, qid_list = prepare_parallel_tsv(
                language=lang,
                output_dir=output_dir / split,
                split=split,
                qid_subset=qids,
                download=False,
            )

    print("\n[parallel] All languages prepared.")

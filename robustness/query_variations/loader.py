"""
Load pre-generated query variations and map qids to evaluation splits
(TREC DL 2019, TREC DL 2020, MS MARCO dev).
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "msmarco-full")

SPLIT_QUERY_PATHS = {
    "dl19": os.path.join(DATA_ROOT, "TREC_DL_2019", "queries_2019", "raw.tsv"),
    "dl20": os.path.join(DATA_ROOT, "TREC_DL_2020", "queries_2020", "raw.tsv"),
    "dev": os.path.join(DATA_ROOT, "dev_queries", "raw.tsv"),
}

SPLIT_QREL_PATHS = {
    "dl19": os.path.join(DATA_ROOT, "TREC_DL_2019", "qrel.json"),
    "dl19_binary": os.path.join(DATA_ROOT, "TREC_DL_2019", "qrel_binary.json"),
    "dl20": os.path.join(DATA_ROOT, "TREC_DL_2020", "qrel.json"),
    "dl20_binary": os.path.join(DATA_ROOT, "TREC_DL_2020", "qrel_binary.json"),
    "dev": os.path.join(DATA_ROOT, "dev_qrel.json"),
}

# Maps split name -> list of (qrel_path, metrics) tuples used by the PAG evaluator
SPLIT_EVAL_CONFIG = {
    "dl19": [
        (SPLIT_QREL_PATHS["dl19"], ["ndcg_cut"]),
        (SPLIT_QREL_PATHS["dl19_binary"], ["mrr_10", "recall"]),
    ],
    "dl20": [
        (SPLIT_QREL_PATHS["dl20"], ["ndcg_cut"]),
        (SPLIT_QREL_PATHS["dl20_binary"], ["mrr_10", "recall"]),
    ],
    "dev": [
        (SPLIT_QREL_PATHS["dev"], ["mrr_10", "recall"]),
    ],
}

# Where query-variation JSONs live
VARIATION_DIR = os.path.join(DATA_ROOT, "query_variations", "msmarco")

ATTACK_METHODS = ["mispelling", "ordering", "synonym", "paraphrase", "naturality"]
SEEDS = [1999, 5, 27, 2016, 2026]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_queries_tsv(tsv_path: str) -> Dict[str, str]:
    """Load a raw.tsv query file into {qid: query_text}."""
    queries = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_split_qids() -> Dict[str, Set[str]]:
    """Return {split_name: set_of_qids} for all three evaluation splits."""
    split_qids = {}
    for split_name, tsv_path in SPLIT_QUERY_PATHS.items():
        split_qids[split_name] = set(load_queries_tsv(tsv_path).keys())
    return split_qids


def load_variation_json(attack_method: str, seed: int,
                        variation_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Load a query-variation JSON file.
    Returns {qid: perturbed_query_text}.
    """
    variation_dir = variation_dir or VARIATION_DIR
    filename = f"msmarco_test_attacked_queries_seed_{seed}_attack_method_{attack_method}.json"
    path = os.path.join(variation_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Query variation file not found: {path}")
    with open(path) as f:
        return json.load(f)


def partition_by_split(
    variation_qids: Set[str],
    split_qids: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """
    Given a set of variation qids, return {split: qids_in_that_split}.
    If a qid appears in multiple splits, it is assigned to the first match
    in priority order: dl19 > dl20 > dev.
    """
    if split_qids is None:
        split_qids = load_split_qids()

    result: Dict[str, Set[str]] = {}
    assigned: Set[str] = set()
    for split in ["dl19", "dl20", "dev"]:
        matched = variation_qids & split_qids[split] - assigned
        if matched:
            result[split] = matched
            assigned |= matched
    return result


def write_perturbed_queries_tsv(
    queries: Dict[str, str],
    out_path: str,
) -> str:
    """
    Write a {qid: text} dict as a raw.tsv file (same format the PAG pipeline
    expects via CollectionDatasetPreLoad).
    Returns the directory containing the file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for qid, text in sorted(queries.items(), key=lambda x: int(x[0])):
            f.write(f"{qid}\t{text}\n")
    return os.path.dirname(out_path)


def prepare_split_queries(
    attack_method: str,
    seed: int,
    split: str,
    output_base: str,
    variation_dir: Optional[str] = None,
    max_variants_per_qid: int = 1,
) -> Tuple[str, str, int]:
    """
    Prepare clean and perturbed query TSV files for a single split.

    Returns (clean_query_dir, perturbed_query_dir, n_queries).
    """
    # Load the original queries for this split
    clean_queries = load_queries_tsv(SPLIT_QUERY_PATHS[split])
    split_qids = set(clean_queries.keys())

    # Load variations and filter to this split
    all_variations = load_variation_json(attack_method, seed, variation_dir)
    split_variations = {
        qid: text for qid, text in all_variations.items() if qid in split_qids
    }

    # Build clean subset (only qids that have variations)
    clean_subset = {qid: clean_queries[qid] for qid in split_variations}

    # Write TSV files
    # The subdirectory names MUST contain the string that
    # t5_pretrainer.utils.utils.get_dataset_name() checks for.
    # That function is case-sensitive: "TREC_DL_2019", "TREC_DL_2020",
    # and lowercase "msmarco" (not "MSMARCO").
    split_label = {
        "dl19": "TREC_DL_2019",
        "dl20": "TREC_DL_2020",
        "dev": "msmarco_dev",
    }[split]

    clean_dir = os.path.join(output_base, split_label, "clean")
    perturbed_dir = os.path.join(
        output_base, split_label, "perturbed", f"{attack_method}_seed_{seed}"
    )

    clean_tsv = os.path.join(clean_dir, "raw.tsv")
    perturbed_tsv = os.path.join(perturbed_dir, "raw.tsv")

    write_perturbed_queries_tsv(clean_subset, clean_tsv)
    write_perturbed_queries_tsv(split_variations, perturbed_tsv)

    return clean_dir, perturbed_dir, len(split_variations)

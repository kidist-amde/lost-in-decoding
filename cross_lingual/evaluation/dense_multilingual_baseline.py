#!/usr/bin/env python3
"""
Exact multilingual dense baseline for RQ3 using a SentenceTransformer encoder.

This script:
  - encodes the MS MARCO English corpus in batches,
  - caches corpus embeddings with --corpus_emb_cache,
  - encodes mMARCO queries per language,
  - runs exact FAISS inner-product search,
  - computes MRR@10 and NDCG@10,
  - writes summary.json and summary.csv in the RQ3 summary layout.

Default query files:
  experiments/RQ3_crosslingual/queries/msmarco_dev/{nl,fr,de,zh}/raw.tsv

Default outputs:
  experiments/RQ3_crosslingual/dense_multilingual_baseline/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / "data" / "msmarco-full"
MMARCO_ROOT = REPO_ROOT / "data" / "mmarco"
CORPUS_TSV = DATA_ROOT / "full_collection" / "raw.tsv"

SPLIT_TO_DATASET = {
    "dl19": "TREC_DL_2019",
    "dl20": "TREC_DL_2020",
    "dev": "MSMARCO",
}

SPLIT_LABELS = {
    "dl19": "TREC_DL_2019",
    "dl20": "TREC_DL_2020",
    "dev": "msmarco_dev",
}

SPLIT_QUERY_PATHS = {
    "dl19": DATA_ROOT / "TREC_DL_2019" / "queries_2019" / "raw.tsv",
    "dl20": DATA_ROOT / "TREC_DL_2020" / "queries_2020" / "raw.tsv",
    "dev": DATA_ROOT / "dev_queries" / "raw.tsv",
}

SPLIT_QREL_PATHS = {
    "dl19": DATA_ROOT / "TREC_DL_2019" / "qrel.json",
    "dl20": DATA_ROOT / "TREC_DL_2020" / "qrel.json",
    "dev": DATA_ROOT / "dev_qrel.json",
}

RQ3_LANGUAGES = ["nl", "fr", "de", "zh"]
DEFAULT_MODEL = "google/embeddinggemma-300m"

ISO_TO_LANG_NAME = {
    "ar": "arabic",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "vi": "vietnamese",
    "zh": "chinese",
}

RQ3_CSV_COLUMNS = [
    "language",
    "split",
    "n_queries",
    "en_lex_NDCG@10",
    "en_lex_MRR@10",
    "en_smt_NDCG@10",
    "en_smt_MRR@10",
    "naive_lex_NDCG@10",
    "naive_lex_MRR@10",
    "naive_smt_NDCG@10",
    "naive_smt_MRR@10",
    "seq_only_NDCG@10",
    "seq_only_MRR@10",
    "trans_lex_NDCG@10",
    "trans_lex_MRR@10",
    "trans_smt_NDCG@10",
    "trans_smt_MRR@10",
    "swap_smt_NDCG@10",
    "swap_smt_MRR@10",
    "delta_simul_NDCG@10",
    "delta_simul_MRR@10",
    "delta_pag_NDCG@10",
    "delta_pag_MRR@10",
    "trans_gain_NDCG@10",
    "trans_gain_MRR@10",
    "swap_gain_NDCG@10",
    "swap_gain_MRR@10",
    "en_PAG_Gain_NDCG@10",
    "en_PAG_Gain_MRR@10",
    "naive_PAG_Gain_NDCG@10",
    "naive_PAG_Gain_MRR@10",
    "delta_recall@10",
    "delta_recall@100",
    "delta_recall@1000",
    "delta_SimulOnly_NDCG@10",
    "delta_SimulOnly_MRR@10",
    "token_jaccard_mean",
    "token_jaccard_median",
    "trans_token_jaccard_mean",
    "trans_token_jaccard_median",
    "PlanSwap_Gain_NDCG@10",
    "PlanSwap_Gain_MRR@10",
    "dense_NDCG@10",
    "dense_MRR@10",
    "dense_model",
    "dense_num_evaluated",
    "corpus_emb_cache",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


def patch_sentence_transformers_import() -> None:
    """Support older sentence-transformers with newer huggingface_hub."""
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            from huggingface_hub import hf_hub_download

            huggingface_hub.cached_download = hf_hub_download
    except Exception:
        pass


def numeric_qid_sort_key(qid: str) -> Tuple[int, str]:
    try:
        return int(qid), qid
    except ValueError:
        return sys.maxsize, qid


def read_tsv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                values[str(parts[0])] = parts[1]
    return values


def load_corpus(path: Path) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            doc_ids.append(str(parts[0]))
            texts.append(parts[1])
    LOG.info("Loaded %d corpus passages from %s", len(doc_ids), path)
    return doc_ids, texts


def load_qrels(split: str) -> Dict[str, Dict[str, int]]:
    with open(SPLIT_QREL_PATHS[split]) as f:
        raw = json.load(f)
    return {
        str(qid): {str(docid): int(rel) for docid, rel in rels.items()}
        for qid, rels in raw.items()
    }


def download_mmarco_queries(language: str, force: bool = False) -> Path:
    if language not in ISO_TO_LANG_NAME:
        raise ValueError(f"Unsupported mMARCO language: {language}")

    output_dir = MMARCO_ROOT / language
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_file = output_dir / "queries.tsv"
    if queries_file.exists() and not force:
        return queries_file

    from datasets import load_dataset_builder
    import pyarrow as pa

    lang_name = ISO_TO_LANG_NAME[language]
    LOG.info("Downloading mMARCO %s queries into %s", language, queries_file)
    builder = load_dataset_builder("unicamp-dl/mmarco", f"queries-{lang_name}")
    builder.download_and_prepare()

    seen = set()
    n_written = 0
    with open(queries_file, "w") as out:
        for arrow_name in ["mmarco-train.arrow", "mmarco-dev.arrow", "mmarco-dev.full.arrow"]:
            arrow_path = Path(builder.cache_dir) / arrow_name
            if not arrow_path.exists():
                continue
            table = pa.ipc.open_stream(str(arrow_path)).read_all()
            for i in range(len(table)):
                qid = str(table.column("id")[i].as_py())
                if qid in seen:
                    continue
                seen.add(qid)
                text = str(table.column("text")[i].as_py())
                text = text.replace("\t", " ").replace("\n", " ")
                out.write(f"{qid}\t{text}\n")
                n_written += 1
    LOG.info("Saved %d mMARCO queries to %s", n_written, queries_file)
    return queries_file


def load_language_queries(
    language: str,
    split: str,
    qrels: Dict[str, Dict[str, int]],
    download: bool,
) -> Tuple[List[str], List[str]]:
    english_queries = read_tsv(SPLIT_QUERY_PATHS[split])
    prepared_query_path = (
        REPO_ROOT
        / "experiments"
        / "RQ3_crosslingual"
        / "queries"
        / SPLIT_LABELS[split]
        / language
        / "raw.tsv"
    )

    if language == "en":
        lang_queries = english_queries
    elif prepared_query_path.exists():
        LOG.info("Using prepared query file: %s", prepared_query_path)
        lang_queries = read_tsv(prepared_query_path)
    else:
        queries_file = MMARCO_ROOT / language / "queries.tsv"
        if not queries_file.exists():
            if not download:
                raise FileNotFoundError(
                    f"Missing mMARCO query file: {queries_file}. "
                    "Run with downloads enabled or pre-populate the query TSVs."
                )
            download_mmarco_queries(language)
        lang_queries = read_tsv(queries_file)

    qids = sorted(
        set(english_queries) & set(lang_queries) & set(qrels),
        key=numeric_qid_sort_key,
    )
    texts = [lang_queries[qid] for qid in qids]
    LOG.info("Loaded %d %s queries for split=%s", len(qids), language, split)
    return qids, texts


def load_sentence_transformer(model_name: str, device: str):
    patch_sentence_transformers_import()
    from sentence_transformers import SentenceTransformer

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    LOG.info("Loading SentenceTransformer %s on %s", model_name, device)
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        error_text = str(exc).lower()
        if "gated repo" in error_text or "access to model" in error_text:
            raise RuntimeError(
                f"Cannot load model '{model_name}' because it is gated on Hugging Face. "
                f"Request access for that repo or pass an open multilingual model such as "
                f"'{DEFAULT_MODEL}' via --model."
            ) from exc
        raise
    has_model_card_api = hasattr(model, "encode_query") and hasattr(model, "encode_document")
    if not has_model_card_api:
        LOG.warning(
            "Installed sentence-transformers does not expose encode_query/encode_document; "
            "falling back to normalized encode(). Upgrade sentence-transformers for exact model-card API."
        )
    return model, has_model_card_api


def uses_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


def prepare_texts_for_encoding(
    texts: Sequence[str],
    mode: str,
    model_name: str,
    has_model_card_api: bool,
) -> List[str]:
    prepared = list(texts)
    if has_model_card_api:
        return prepared
    if uses_e5_prefixes(model_name):
        prefix = "query: " if mode == "query" else "passage: "
        return [text if text.startswith(prefix) else f"{prefix}{text}" for text in prepared]
    return prepared


def encode_with_model_card_api(
    model,
    texts: Sequence[str],
    mode: str,
    batch_size: int,
    normalize: bool,
    show_progress: bool,
    has_model_card_api: bool,
    model_name: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if has_model_card_api:
        if mode == "query":
            embeddings = model.encode_query(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
            )
        elif mode == "document":
            embeddings = model.encode_document(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
            )
        else:
            raise ValueError(f"Unknown encoding mode: {mode}")
    else:
        prepared_texts = prepare_texts_for_encoding(texts, mode, model_name, has_model_card_api)
        embeddings = model.encode(
            prepared_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

    return np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))


def cache_sidecar(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


def model_cache_stem(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-") or "model"


def default_corpus_emb_cache(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_cache_stem(model_name)}_corpus_embs.npy"


def load_cached_embeddings(
    cache_path: Path,
    expected_count: int,
    expected_model: str,
    expected_normalize: bool,
) -> Optional[np.ndarray]:
    if not cache_path.exists():
        return None

    meta_path = cache_sidecar(cache_path, ".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        cached_model = meta.get("model")
        cached_normalize = meta.get("normalization")
        if cached_model and cached_model != expected_model:
            raise ValueError(
                f"Corpus cache {cache_path} was created for model '{cached_model}', "
                f"not '{expected_model}'. Use a different --corpus_emb_cache path or "
                "--force_corpus_reencode."
            )
        if cached_normalize is not None and bool(cached_normalize) != bool(expected_normalize):
            raise ValueError(
                f"Corpus cache {cache_path} was created with normalize={cached_normalize}, "
                f"not normalize={expected_normalize}. Use a different --corpus_emb_cache path "
                f"or --force_corpus_reencode."
            )

    LOG.info("Loading cached corpus embeddings from %s", cache_path)
    with open(cache_path, "rb") as f:
        embeddings = np.load(f)
    if embeddings.shape[0] != expected_count:
        raise ValueError(
            f"Corpus cache row count {embeddings.shape[0]} does not match corpus size {expected_count}"
        )
    return np.ascontiguousarray(embeddings.astype(np.float32, copy=False))


def save_cached_embeddings(cache_path: Path, embeddings: np.ndarray, doc_ids: Sequence[str], meta: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        np.save(f, embeddings.astype(np.float32, copy=False))
    with open(cache_sidecar(cache_path, ".docids.json"), "w") as f:
        json.dump(list(doc_ids), f)
    with open(cache_sidecar(cache_path, ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    LOG.info("Saved corpus embeddings to %s", cache_path)


def parse_gpu_devices(device_spec: str) -> Optional[List[int]]:
    if not device_spec or device_spec == "all":
        return None
    return [int(part) for part in device_spec.split(",") if part.strip()]


def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool,
    gpu_devices: str,
    add_batch_size: int,
):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        devices = parse_gpu_devices(gpu_devices)
        if devices is None:
            index = faiss.index_cpu_to_all_gpus(index, co=co)
            LOG.info("Moved exact FAISS index to all visible GPUs with sharding")
        elif len(devices) == 1:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, devices[0], index)
            LOG.info("Moved exact FAISS index to GPU %s", devices[0])
        else:
            resources = [faiss.StandardGpuResources() for _ in devices]
            index = faiss.index_cpu_to_gpu_multiple_py(resources, index, co=co, gpus=devices)
            LOG.info("Moved exact FAISS index to GPUs %s with sharding", devices)

    for start in range(0, embeddings.shape[0], add_batch_size):
        batch = np.ascontiguousarray(embeddings[start:start + add_batch_size], dtype=np.float32)
        index.add(batch)
        LOG.info(
            "Added %d / %d corpus vectors to FAISS",
            min(start + add_batch_size, embeddings.shape[0]),
            embeddings.shape[0],
        )
    return index


def faiss_search(index, query_embeddings: np.ndarray, topk: int, search_batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    all_scores: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    for start in range(0, query_embeddings.shape[0], search_batch_size):
        batch = np.ascontiguousarray(query_embeddings[start:start + search_batch_size], dtype=np.float32)
        scores, indices = index.search(batch, topk)
        all_scores.append(scores)
        all_indices.append(indices)
        LOG.info("Searched %d / %d queries", min(start + search_batch_size, query_embeddings.shape[0]), query_embeddings.shape[0])
    return np.vstack(all_scores), np.vstack(all_indices)


def build_run(
    qids: Sequence[str],
    doc_ids: Sequence[str],
    scores: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    run: Dict[str, Dict[str, float]] = {}
    for row_idx, qid in enumerate(qids):
        doc_scores: Dict[str, float] = {}
        for score, doc_idx in zip(scores[row_idx], indices[row_idx]):
            if doc_idx < 0:
                continue
            doc_scores[str(doc_ids[int(doc_idx)])] = float(score)
        run[str(qid)] = doc_scores
    return run


def dcg(relevances: Iterable[int]) -> float:
    value = 0.0
    for rank, rel in enumerate(relevances, start=1):
        value += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    return value


def compute_metrics(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict[str, float]:
    mrr_values: List[float] = []
    ndcg_values: List[float] = []

    for qid, rels in qrels.items():
        if qid not in run:
            continue
        ranked_docs = sorted(run[qid].items(), key=lambda item: item[1], reverse=True)
        top_docs = [docid for docid, _ in ranked_docs[:k]]

        rr = 0.0
        for rank, docid in enumerate(top_docs, start=1):
            if rels.get(docid, 0) > 0:
                rr = 1.0 / rank
                break
        mrr_values.append(rr)

        observed_rels = [rels.get(docid, 0) for docid in top_docs]
        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        ideal = dcg(ideal_rels)
        ndcg_values.append(0.0 if ideal <= 0 else dcg(observed_rels) / ideal)

    return {
        f"MRR@{k}": float(np.mean(mrr_values)) if mrr_values else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "n_evaluated": len(mrr_values),
    }


def evaluate_one(
    language: str,
    split: str,
    qids: Sequence[str],
    query_texts: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    doc_ids: Sequence[str],
    faiss_index,
    model,
    has_model_card_api: bool,
    args,
    output_dir: Path,
) -> Dict:
    LOG.info("Encoding/evaluating language=%s split=%s", language, split)
    query_embeddings = encode_with_model_card_api(
        model,
        query_texts,
        mode="query",
        batch_size=args.query_batch_size,
        normalize=args.normalize,
        show_progress=args.show_progress,
        has_model_card_api=has_model_card_api,
        model_name=args.model,
    )

    topk = min(args.topk, len(doc_ids))
    scores, indices = faiss_search(
        faiss_index,
        query_embeddings,
        topk=topk,
        search_batch_size=args.search_batch_size,
    )
    run = build_run(qids, doc_ids, scores, indices)
    metrics = compute_metrics(run, qrels, k=10)

    run_dir = output_dir / language / split
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run.json", "w") as f:
        json.dump(run, f)

    result = {
        "language": language,
        "split": split,
        "dataset": SPLIT_TO_DATASET[split],
        "model": args.model,
        "n_queries": len(qids),
        "dense_metrics": {
            "NDCG@10": metrics["NDCG@10"],
            "MRR@10": metrics["MRR@10"],
            "n_evaluated": metrics["n_evaluated"],
        },
        "dense_smt_metrics": {
            "NDCG@10": metrics["NDCG@10"],
            "MRR@10": metrics["MRR@10"],
            "n_evaluated": metrics["n_evaluated"],
        },
        "corpus_emb_cache": str(args.corpus_emb_cache),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    LOG.info(
        "language=%s split=%s NDCG@10=%.5f MRR@10=%.5f n=%d",
        language,
        split,
        metrics["NDCG@10"],
        metrics["MRR@10"],
        metrics["n_evaluated"],
    )
    return result


def summary_row(result: Dict) -> Dict[str, object]:
    metrics = result.get("dense_metrics", {})
    row = {column: "" for column in RQ3_CSV_COLUMNS}
    row.update(
        {
            "language": result.get("language"),
            "split": result.get("split"),
            "n_queries": result.get("n_queries"),
            "dense_NDCG@10": metrics.get("NDCG@10"),
            "dense_MRR@10": metrics.get("MRR@10"),
            "dense_model": result.get("model"),
            "dense_num_evaluated": metrics.get("n_evaluated"),
            "corpus_emb_cache": result.get("corpus_emb_cache"),
        }
    )
    return row


def write_summary(results: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RQ3_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(summary_row(result))
    LOG.info("Wrote %s and %s", output_dir / "summary.json", output_dir / "summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact FAISS dense multilingual baseline for RQ3")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model ID or local path")
    parser.add_argument("--languages", nargs="+", default=RQ3_LANGUAGES, help="Languages to evaluate, or 'all'")
    parser.add_argument("--splits", nargs="+", default=["dev"], help="Splits to evaluate: dev, dl19, dl20, or 'all'")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "experiments" / "RQ3_crosslingual" / "dense_multilingual_baseline"),
        help="Output directory for runs and summary files",
    )
    parser.add_argument(
        "--corpus_emb_cache",
        default=None,
        help="Path to cached corpus embeddings. Defaults to an output-dir cache keyed by --model.",
    )
    parser.add_argument("--force_corpus_reencode", action="store_true", help="Ignore an existing corpus embedding cache")
    parser.add_argument("--no_download", action="store_true", help="Do not download missing mMARCO query files")
    parser.add_argument("--corpus_batch_size", type=int, default=512, help="Batch size for corpus encoding")
    parser.add_argument("--query_batch_size", type=int, default=512, help="Batch size for query encoding")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True, help="L2-normalize embeddings before IP search")
    parser.add_argument("--topk", type=int, default=100, help="Retrieval depth for each query")
    parser.add_argument("--search_batch_size", type=int, default=1024, help="FAISS query search batch size")
    parser.add_argument("--faiss_add_batch_size", type=int, default=500000, help="FAISS corpus add batch size")
    parser.add_argument("--faiss_gpu", action="store_true", help="Move the exact IndexFlatIP search index to GPU(s)")
    parser.add_argument("--faiss_gpu_devices", default="1,2,3", help="Comma-separated GPU IDs for FAISS search, or 'all'")
    parser.add_argument("--device", default="cuda:0", help="Torch device for SentenceTransformer encoding")
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True, help="Show SentenceTransformers progress bars")
    args = parser.parse_args()

    if "all" in args.languages:
        args.languages = RQ3_LANGUAGES
    if "all" in args.splits:
        args.splits = ["dl19", "dl20", "dev"]

    invalid_splits = sorted(set(args.splits) - set(SPLIT_TO_DATASET))
    if invalid_splits:
        parser.error(f"invalid split(s): {invalid_splits}; use dev, dl19, dl20, or all")

    args.output_dir = Path(args.output_dir)
    if args.corpus_emb_cache:
        args.corpus_emb_cache = Path(args.corpus_emb_cache)
    else:
        args.corpus_emb_cache = default_corpus_emb_cache(args.output_dir, args.model)
    return args


def main() -> None:
    args = parse_args()
    LOG.info("Dense multilingual baseline")
    LOG.info("model=%s", args.model)
    LOG.info("languages=%s", args.languages)
    LOG.info("splits=%s", args.splits)
    LOG.info("output_dir=%s", args.output_dir)
    LOG.info("corpus_emb_cache=%s", args.corpus_emb_cache)

    model, has_model_card_api = load_sentence_transformer(args.model, args.device)

    doc_ids, doc_texts = load_corpus(CORPUS_TSV)
    corpus_embeddings = None
    if not args.force_corpus_reencode:
        corpus_embeddings = load_cached_embeddings(
            args.corpus_emb_cache,
            expected_count=len(doc_ids),
            expected_model=args.model,
            expected_normalize=args.normalize,
        )

    if corpus_embeddings is None:
        start = time.time()
        corpus_embeddings = encode_with_model_card_api(
            model,
            doc_texts,
            mode="document",
            batch_size=args.corpus_batch_size,
            normalize=args.normalize,
            show_progress=args.show_progress,
            has_model_card_api=has_model_card_api,
            model_name=args.model,
        )
        meta = {
            "model": args.model,
            "normalization": args.normalize,
            "shape": list(corpus_embeddings.shape),
            "encoded_seconds": time.time() - start,
            "sentence_transformers_model_card_api": has_model_card_api,
        }
        save_cached_embeddings(args.corpus_emb_cache, corpus_embeddings, doc_ids, meta)

    faiss_index = build_faiss_index(
        corpus_embeddings,
        use_gpu=args.faiss_gpu,
        gpu_devices=args.faiss_gpu_devices,
        add_batch_size=args.faiss_add_batch_size,
    )
    del corpus_embeddings

    results: List[Dict] = []
    for split in args.splits:
        qrels = load_qrels(split)
        for language in args.languages:
            qids, query_texts = load_language_queries(
                language=language,
                split=split,
                qrels=qrels,
                download=not args.no_download,
            )
            if not qids:
                LOG.warning("Skipping language=%s split=%s because no evaluable queries were found", language, split)
                continue
            result = evaluate_one(
                language=language,
                split=split,
                qids=qids,
                query_texts=query_texts,
                qrels=qrels,
                doc_ids=doc_ids,
                faiss_index=faiss_index,
                model=model,
                has_model_card_api=has_model_card_api,
                args=args,
                output_dir=args.output_dir,
            )
            results.append(result)
            write_summary(results, args.output_dir)

    write_summary(results, args.output_dir)
    LOG.info("Done")


if __name__ == "__main__":
    main()

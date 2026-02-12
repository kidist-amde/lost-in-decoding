#!/usr/bin/env python3
"""Measure precise latency and GPU memory for Table 3 efficiency columns.

Uses torch.cuda.Event for timing and torch.cuda.memory_allocated for memory.
Reuses existing pipeline functions; writes results to JSON.

Usage:
    python scripts/measure_efficiency.py [--warmup_batches 5]
"""
import argparse
import json
import os
import sys
import time

import torch
import ujson
from tqdm import tqdm
from transformers import AutoTokenizer

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t5_pretrainer.modeling.t5_generative_retriever import LexicalRipor
from t5_pretrainer.tasks.evaluator import TermEncoderRetriever
from t5_pretrainer.dataset.dataloader import (
    T5SpladeCollectionDataLoader,
    LexicalConditionCollectionDataLoader,
)
from t5_pretrainer.dataset.dataset import CollectionDatasetPreLoad
from t5_pretrainer.evaluate import lexical_tmp_rescore_decode_doc
from t5_pretrainer.utils.utils import (
    get_dataset_name,
    get_qid_smtid_scores,
    convert_ptsmtids_to_strsmtid,
)
from t5_pretrainer.utils.prefixer import BatchPrefixerForLexInc
from t5_pretrainer.utils.sequence_rescorer import BatchLexTmpReScorer
from t5_pretrainer.tasks.generation import generate_for_lex_tmp_rescore

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
PRETRAINED = os.path.join(MODEL_DIR, "checkpoint")
SMT_DOCID_PATH = "data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json"
LEX_BASE = "data/experiments-splade/t5-splade-0-12l"
DEV_QUERIES = "data/msmarco-full/dev_queries/"
MAX_NEW_TOKEN = 8
LEX_TOPK = 1000


def lex_bow_path(m: int) -> str:
    if m == 64:
        return os.path.join(LEX_BASE, "top_bow", "docid_to_tokenids.json")
    return os.path.join(LEX_BASE, f"top_bow_{m}", "docid_to_tokenids.json")


def stage1_run_path(m: int) -> str:
    return os.path.join(
        MODEL_DIR, f"table3_dev/m_{m}/lex_ret_{LEX_TOPK}/MSMARCO/run.json"
    )


# ── helpers ──────────────────────────────────────────────────────────────────
def cuda_timer():
    """Return (start, end) CUDA events."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


# ── Stage 1: simultaneous decoding ──────────────────────────────────────────
def measure_stage1(model, lex_docid_to_smtids, device, warmup_batches=5):
    """Return (index_mem_gb, simul_gpu_ms_per_query, simul_wall_ms_per_query)."""
    model.base_model.mode = "lex_retrieval"

    # ── build index tensor on GPU and measure memory ─────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    doc_encodings = []
    docids = []
    for docid, smtids in lex_docid_to_smtids.items():
        doc_encodings.append(smtids)
        docids.append(docid)
    doc_encodings = torch.LongTensor(doc_encodings).to(device)

    torch.cuda.synchronize(device)
    index_mem_bytes = torch.cuda.memory_allocated(device) - mem_before
    index_mem_gb = index_mem_bytes / 1e9
    m = doc_encodings.shape[1]
    print(f"  Index tensor shape: {doc_encodings.shape}, GPU mem: {index_mem_gb:.3f} GB")

    # ── build query dataloader ───────────────────────────────────────────
    q_collection = CollectionDatasetPreLoad(data_dir=DEV_QUERIES, id_style="row_id")
    q_loader = T5SpladeCollectionDataLoader(
        dataset=q_collection,
        tokenizer_type=PRETRAINED,
        max_length=128,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )
    retriever = TermEncoderRetriever(model, None)

    # ── warmup ───────────────────────────────────────────────────────────
    warmup_iter = iter(q_loader)
    for _ in range(warmup_batches):
        batch = next(warmup_iter)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}
            preds = model.encode(**inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            _ = retriever.get_doc_scores(preds, doc_encodings)
    torch.cuda.synchronize(device)
    warmup_queries = warmup_batches * 8
    print(f"  Warmup done ({warmup_batches} batches, {warmup_queries} queries)")

    # ── timed run (remaining batches) ────────────────────────────────────
    start_ev, end_ev = cuda_timer()
    timed_queries = 0

    wall_start = time.perf_counter()
    start_ev.record()
    for i, batch in enumerate(q_loader):
        if i < warmup_batches:
            continue
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}
            preds = model.encode(**inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            scores = retriever.get_doc_scores(preds, doc_encodings)
            _ = torch.topk(scores, k=LEX_TOPK, dim=-1)
        bz = len(batch["id"]) if isinstance(batch["id"], list) else batch["id"].shape[0]
        timed_queries += bz
    end_ev.record()
    torch.cuda.synchronize(device)
    wall_total_ms = (time.perf_counter() - wall_start) * 1000.0

    gpu_total_ms = start_ev.elapsed_time(end_ev)
    gpu_ms_per_query = gpu_total_ms / timed_queries
    wall_ms_per_query = wall_total_ms / timed_queries
    print(
        f"  Stage 1 GPU-time: {gpu_total_ms:.1f} ms total, {timed_queries} queries, "
        f"{gpu_ms_per_query:.3f} ms/query"
    )
    print(
        f"  Stage 1 E2E wall-time: {wall_total_ms:.1f} ms total, {timed_queries} queries, "
        f"{wall_ms_per_query:.3f} ms/query"
    )

    # cleanup index from GPU
    del doc_encodings
    torch.cuda.empty_cache()

    return index_mem_gb, gpu_ms_per_query, wall_ms_per_query


# ── Stage 2: sequential decoding ────────────────────────────────────────────
def measure_stage2(model, m, k, device, warmup_batches=2):
    """Return (seq_gpu_ms_per_query, seq_wall_ms_per_query)."""
    model.base_model.mode = "smt_retrieval"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

    # ── load smt index ───────────────────────────────────────────────────
    with open(SMT_DOCID_PATH) as f:
        docid_to_tokenids = ujson.load(f)

    smtid_to_docids = {}
    docid_to_sub_tokenids = {}
    for docid, tokenids in docid_to_tokenids.items():
        sid = "_".join(str(x) for x in tokenids[:MAX_NEW_TOKEN])
        smtid_to_docids.setdefault(sid, []).append(docid)
        docid_to_sub_tokenids[docid] = tokenids[:MAX_NEW_TOKEN]

    # ── load Stage 1 run.json ────────────────────────────────────────────
    run_path = stage1_run_path(m)
    with open(run_path) as f:
        qid_to_rankdata = ujson.load(f)
    print(f"  Loaded Stage 1 results: {len(qid_to_rankdata)} queries from {run_path}")

    # ── build query dataloader ───────────────────────────────────────────
    q_collection = CollectionDatasetPreLoad(data_dir=DEV_QUERIES, id_style="row_id")
    q_loader = LexicalConditionCollectionDataLoader(
        dataset=q_collection,
        tokenizer_type=PRETRAINED,
        max_length=128,
        batch_size=16,
        shuffle=False,
        num_workers=1,
    )

    # ── prepare lex scoring ──────────────────────────────────────────────
    lex_qid_to_smtid_to_score = get_qid_smtid_scores(qid_to_rankdata, docid_to_sub_tokenids)

    # ── warmup ───────────────────────────────────────────────────────────
    warmup_iter = iter(q_loader)
    for w in range(warmup_batches):
        batch = next(warmup_iter)
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {ki: v.to(device) for ki, v in batch.items() if ki != "id"}
            batch_prefixer = BatchPrefixerForLexInc(
                docid_to_tokenids=docid_to_sub_tokenids,
                qid_to_rankdata=qid_to_rankdata,
                qids=batch_qids,
                tokenizer=tokenizer,
                pooling="max",
            )
            batch_lex_rescorer = BatchLexTmpReScorer(batch_prefixer, num_beams=k)
            _ = generate_for_lex_tmp_rescore(
                model.base_model,
                prefix_allowed_tokens_fn=batch_prefixer,
                logit_tmp_rescorer=batch_lex_rescorer,
                input_ids=inputs["input_ids"].long(),
                attention_mask=inputs["attention_mask"].long(),
                decoder_input_ids=inputs["decoder_input_ids"].long(),
                max_new_tokens=MAX_NEW_TOKEN,
                output_scores=True,
                return_dict=True,
                return_dict_in_generate=True,
                num_beams=k,
                num_return_sequences=k,
            )
    torch.cuda.synchronize(device)
    warmup_queries = warmup_batches * 16
    print(f"  Warmup done ({warmup_batches} batches, {warmup_queries} queries)")

    # ── timed run ────────────────────────────────────────────────────────
    start_ev, end_ev = cuda_timer()
    timed_queries = 0

    wall_start = time.perf_counter()
    start_ev.record()
    for i, batch in enumerate(q_loader):
        if i < warmup_batches:
            continue
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {ki: v.to(device) for ki, v in batch.items() if ki != "id"}
            batch_prefixer = BatchPrefixerForLexInc(
                docid_to_tokenids=docid_to_sub_tokenids,
                qid_to_rankdata=qid_to_rankdata,
                qids=batch_qids,
                tokenizer=tokenizer,
                pooling="max",
            )
            batch_lex_rescorer = BatchLexTmpReScorer(batch_prefixer, num_beams=k)
            _ = generate_for_lex_tmp_rescore(
                model.base_model,
                prefix_allowed_tokens_fn=batch_prefixer,
                logit_tmp_rescorer=batch_lex_rescorer,
                input_ids=inputs["input_ids"].long(),
                attention_mask=inputs["attention_mask"].long(),
                decoder_input_ids=inputs["decoder_input_ids"].long(),
                max_new_tokens=MAX_NEW_TOKEN,
                output_scores=True,
                return_dict=True,
                return_dict_in_generate=True,
                num_beams=k,
                num_return_sequences=k,
            )
        timed_queries += len(batch_qids)
    end_ev.record()
    torch.cuda.synchronize(device)
    wall_total_ms = (time.perf_counter() - wall_start) * 1000.0

    gpu_total_ms = start_ev.elapsed_time(end_ev)
    gpu_ms_per_query = gpu_total_ms / timed_queries
    wall_ms_per_query = wall_total_ms / timed_queries
    print(
        f"  Stage 2 (k={k}) GPU-time: {gpu_total_ms:.1f} ms total, {timed_queries} queries, "
        f"{gpu_ms_per_query:.3f} ms/query"
    )
    print(
        f"  Stage 2 (k={k}) E2E wall-time: {wall_total_ms:.1f} ms total, {timed_queries} queries, "
        f"{wall_ms_per_query:.3f} ms/query"
    )

    return gpu_ms_per_query, wall_ms_per_query


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_batches_s1", type=int, default=5)
    parser.add_argument("--warmup_batches_s2", type=int, default=2)
    parser.add_argument("--m_values", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--out", type=str,
                        default="experiments/table3_mk_sweep/efficiency_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda:0")

    # ── load model once ──────────────────────────────────────────────────
    print("Loading model...")
    model = LexicalRipor.from_pretrained(PRETRAINED)
    model.eval()
    model.to(device)

    results = {}

    for m in args.m_values:
        print(f"\n{'='*60}")
        print(f" m = {m}")
        print(f"{'='*60}")

        # ── load BOW index ───────────────────────────────────────────
        bow_path = lex_bow_path(m)
        print(f"  Loading BOW: {bow_path}")
        with open(bow_path) as f:
            lex_docid_to_smtids = ujson.load(f)
        print(f"  {len(lex_docid_to_smtids):,} documents, m={m} tokens each")

        # ── Stage 1 ─────────────────────────────────────────────────
        print("\n  --- Stage 1: Simultaneous Decoding ---")
        index_mem_gb, simul_gpu_ms, simul_wall_ms = measure_stage1(
            model, lex_docid_to_smtids, device,
            warmup_batches=args.warmup_batches_s1,
        )

        entry = {
            "index_mem_gb": round(index_mem_gb, 3),
            "simul_gpu_ql_ms": round(simul_gpu_ms, 3),
            "simul_wall_ql_ms": round(simul_wall_ms, 3),
        }

        # ── Stage 2 for each k ──────────────────────────────────────
        for k in args.k_values:
            print(f"\n  --- Stage 2: Sequential Decoding (k={k}) ---")
            seq_gpu_ms, seq_wall_ms = measure_stage2(
                model, m, k, device,
                warmup_batches=args.warmup_batches_s2,
            )
            entry[f"seq_gpu_ql_k{k}_ms"] = round(seq_gpu_ms, 3)
            entry[f"seq_wall_ql_k{k}_ms"] = round(seq_wall_ms, 3)

        results[f"m{m}"] = entry
        print(f"\n  Results for m={m}: {json.dumps(entry, indent=2)}")

    # ── write results ────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")

    # ── summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EFFICIENCY RESULTS SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'m':>4}  {'Index (GB)':>10}  {'SimGPU':>10}  {'SimWall':>10}  "
        f"{'Seq10GPU':>10}  {'Seq10Wall':>10}  {'Seq100GPU':>10}  {'Seq100Wall':>10}"
    )
    print(
        f"{'----':>4}  {'----------':>10}  {'------':>10}  {'-------':>10}  "
        f"{'--------':>10}  {'---------':>10}  {'---------':>10}  {'----------':>10}"
    )
    for m in args.m_values:
        e = results[f"m{m}"]
        seq10_gpu = e.get("seq_gpu_ql_k10_ms", float("nan"))
        seq10_wall = e.get("seq_wall_ql_k10_ms", float("nan"))
        seq100_gpu = e.get("seq_gpu_ql_k100_ms", float("nan"))
        seq100_wall = e.get("seq_wall_ql_k100_ms", float("nan"))
        print(
            f"{m:>4}  {e['index_mem_gb']:>10.3f}  {e['simul_gpu_ql_ms']:>10.3f}  "
            f"{e['simul_wall_ql_ms']:>10.3f}  {seq10_gpu:>10.3f}  {seq10_wall:>10.3f}  "
            f"{seq100_gpu:>10.3f}  {seq100_wall:>10.3f}"
        )


if __name__ == "__main__":
    main()

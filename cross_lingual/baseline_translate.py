#!/usr/bin/env python3
"""
Baseline C: Translate-at-inference PAG.

Translates non-English queries to English offline (NLLB-200 distilled
or Opus-MT), then runs the standard English PAG pipeline on the
translated queries.

Produces:
- Stage 1 run file (planner-only / SimulOnly) in TREC format
- Stage 2 run file (full PAG) in TREC format
- Metrics JSON (MRR@10, Recall@10, nDCG@10, latency, error_rate)
- Translation log CSV: qid, src_query, translated_query, status/error
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cross_lingual.data_loader import (
    RQ3_LANGUAGES,
    prepare_query_files,
    save_coverage_report,
    save_filtered_qrels,
    load_mmarco_queries,
    get_evaluation_qids,
)
from cross_lingual.retrieval_engine import (
    run_stage1,
    run_stage2,
    count_queries,
    load_lexical_run,
    load_sequential_run,
    run_json_to_trec,
    _get_dataset_name,
)
from cross_lingual.evaluate import compute_metrics, save_metrics


# ---------------------------------------------------------------------------
# Translation with logging
# ---------------------------------------------------------------------------

def translate_queries_with_log(
    queries: Dict[str, str],
    language: str,
    output_dir: Path,
    model_type: str = "nllb",
    batch_size: int = 32,
    device: str = "cuda",
) -> Tuple[Dict[str, str], Path]:
    """Translate queries and produce a detailed translation log.

    Returns:
        (translated_queries, log_path)
    """
    from cross_lingual.utils.translator import CachedTranslator

    translator = CachedTranslator(model_type=model_type, device=device)

    t0 = time.time()
    translated = translator.translate_queries(
        queries=queries,
        src_lang=language,
        tgt_lang="en",
        batch_size=batch_size,
    )
    translate_time = time.time() - t0

    # Write translation log
    log_path = output_dir / "translation_log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "src_query", "translated_query", "status"])
        n_ok = 0
        n_err = 0
        for qid in sorted(queries.keys(), key=lambda x: int(x) if x.isdigit() else x):
            src = queries[qid]
            if qid in translated and translated[qid].strip():
                writer.writerow([qid, src, translated[qid], "ok"])
                n_ok += 1
            else:
                writer.writerow([qid, src, "", "error:empty_translation"])
                n_err += 1

    print(f"[translate] {n_ok} ok, {n_err} errors in {translate_time:.1f}s")
    print(f"[translate] Log saved -> {log_path}")

    return translated, log_path


def prepare_translated_query_dir(
    translated: Dict[str, str],
    output_dir: Path,
) -> Path:
    """Write translated queries to a PAG-compatible directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "raw.tsv", "w") as f:
        for qid in sorted(translated.keys(), key=lambda x: int(x) if x.isdigit() else x):
            text = translated[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")
    return output_dir


# ---------------------------------------------------------------------------
# Baseline C runner
# ---------------------------------------------------------------------------

def run_translate_baseline(
    language: str,
    output_dir: str,
    lex_topk: int = 1000,
    smt_topk: int = 100,
    batch_size: int = 8,
    n_gpu: int = 1,
    translation_model: str = "nllb",
    translation_batch_size: int = 32,
    device: str = "cuda",
    skip_existing: bool = True,
) -> dict:
    """Run Baseline C (translate-at-inference) for one language."""
    out = Path(output_dir) / language / "translate"
    out.mkdir(parents=True, exist_ok=True)

    # Prepare data
    queries_dir = Path(output_dir) / "queries"
    en_dir, tgt_dir, n_queries, coverage = prepare_query_files(
        language, queries_dir, download=True,
    )
    save_coverage_report(coverage, out)

    eval_qids = set(coverage["intersection_qids"])
    qrel_path = out / "qrels.json"
    save_filtered_qrels(eval_qids, qrel_path)

    result = {
        "baseline": "translate",
        "language": language,
        "n_queries": n_queries,
        "translation_model": translation_model,
    }

    # --- Translation ---
    translated_dir = out / "translated_queries"
    translated_tsv = translated_dir / "raw.tsv"

    if translated_tsv.exists() and skip_existing:
        print(f"[translate] Translated queries exist, loading from cache")
        translated = {}
        with open(translated_tsv) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    translated[parts[0]] = parts[1]
        result["translation_time_s"] = 0
    else:
        # Load source-language queries (only eval qids)
        all_mmarco = load_mmarco_queries(language, download=True)
        src_queries = {qid: all_mmarco[qid] for qid in eval_qids if qid in all_mmarco}

        t0 = time.time()
        translated, log_path = translate_queries_with_log(
            queries=src_queries,
            language=language,
            output_dir=out,
            model_type=translation_model,
            batch_size=translation_batch_size,
            device=device,
        )
        result["translation_time_s"] = time.time() - t0
        result["translation_log"] = str(log_path)

        prepare_translated_query_dir(translated, translated_dir)

    dataset_name = _get_dataset_name(str(translated_dir))
    lex_out_dir = str(out / "lex_ret")
    smt_out_dir = str(out / "smt_ret")

    # --- Stage 1 (planner on translated queries) ---
    lex_run_path = os.path.join(lex_out_dir, dataset_name, "run.json")
    if os.path.exists(lex_run_path) and skip_existing:
        print(f"[translate] Stage 1 exists, skipping")
        result["stage1_time_s"] = 0
    else:
        rc1, t1 = run_stage1(
            str(translated_dir), lex_out_dir, lex_topk, batch_size,
        )
        result["stage1_time_s"] = t1
        if rc1 != 0:
            result["error"] = f"stage1_failed_rc={rc1}"
            return result

    # --- Stage 2 (full PAG on translated queries) ---
    smt_run_path = os.path.join(smt_out_dir, dataset_name, "run.json")
    if os.path.exists(smt_run_path) and skip_existing:
        print(f"[translate] Stage 2 exists, skipping")
        result["stage2_time_s"] = 0
    else:
        rc2, t2 = run_stage2(
            str(translated_dir), smt_out_dir, lex_out_dir,
            smt_topk, batch_size * 2, n_gpu,
        )
        result["stage2_time_s"] = t2
        if rc2 != 0:
            result["error"] = f"stage2_failed_rc={rc2}"
            return result

    # --- Merge & evaluate ---
    from robustness.utils.pag_inference import merge_and_evaluate
    merge_and_evaluate(str(translated_dir), smt_out_dir, str(qrel_path))

    # Load runs, save TREC
    lex_run = load_lexical_run(lex_out_dir, dataset_name)
    smt_run = load_sequential_run(smt_out_dir, dataset_name)

    if lex_run:
        trec_s1 = str(out / "stage1_run.trec")
        run_json_to_trec(lex_run, trec_s1, f"translate_{language}_s1")

    if smt_run:
        trec_s2 = str(out / "stage2_run.trec")
        run_json_to_trec(smt_run, trec_s2, f"translate_{language}_s2")

    # Metrics
    qrels = json.load(open(qrel_path))
    total_time = sum(result.get(k, 0) for k in [
        "translation_time_s", "stage1_time_s", "stage2_time_s"
    ])

    if lex_run:
        s1_metrics = compute_metrics(lex_run, qrels, n_queries, total_time)
        result["stage1_metrics"] = s1_metrics
        save_metrics(s1_metrics, out / "stage1_metrics.json")

    if smt_run:
        s2_metrics = compute_metrics(smt_run, qrels, n_queries, total_time)
        result["stage2_metrics"] = s2_metrics
        save_metrics(s2_metrics, out / "stage2_metrics.json")

    with open(out / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Baseline C: Translate-at-inference PAG"
    )
    parser.add_argument("--language", type=str, required=True,
                        help=f"Language: {RQ3_LANGUAGES} or 'all'")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "cross_lingual" / "results"))
    parser.add_argument("--lex_topk", type=int, default=1000)
    parser.add_argument("--smt_topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--translation_model", type=str, default="nllb",
                        choices=["nllb", "m2m100"])
    parser.add_argument("--translation_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = RQ3_LANGUAGES if args.language == "all" else [args.language]

    for lang in languages:
        print(f"\n{'='*60}")
        print(f"[translate] Running Baseline C for {lang}")
        print(f"{'='*60}")
        result = run_translate_baseline(
            language=lang,
            output_dir=args.output_dir,
            lex_topk=args.lex_topk,
            smt_topk=args.smt_topk,
            batch_size=args.batch_size,
            n_gpu=args.n_gpu,
            translation_model=args.translation_model,
            translation_batch_size=args.translation_batch_size,
            device=args.device,
            skip_existing=not args.force,
        )
        print(f"[translate] {lang} done. Errors: {result.get('error', 'none')}")


if __name__ == "__main__":
    main()

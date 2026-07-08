#!/usr/bin/env python3
"""Score RQ2_REDECODE_2026-07-07 runs with pytrec_eval, matching
t5_pretrainer.utils.metrics conventions (top-10 truncation for MRR/recip_rank)."""
import json
import os
import statistics as stats
import sys

import pytrec_eval

REPO_ROOT = "/gpfs/work4/0/prjs1037/dpo-exp/pag-repro"
OUT_ROOT = os.path.join(REPO_ROOT, "data/experiments-full-lexical-ripor",
                         "lexical_ripor_direct_lng_knp_seq2seq_1", "RQ2_REDECODE_2026-07-07")

ATTACK_METHODS = ["mispelling", "ordering", "synonym", "paraphrase", "naturality"]
SEEDS = [1999, 5, 27, 2016, 2026]

SPLIT_DATASET = {"dl19": "TREC_DL_2019", "dl20": "TREC_DL_2020", "dev": "MSMARCO"}
SPLIT_GRADED_QREL = {
    "dl19": os.path.join(REPO_ROOT, "data/msmarco-full/TREC_DL_2019/qrel.json"),
    "dl20": os.path.join(REPO_ROOT, "data/msmarco-full/TREC_DL_2020/qrel.json"),
    "dev": os.path.join(REPO_ROOT, "data/msmarco-full/dev_qrel.json"),
}
SPLIT_BINARY_QREL = {
    "dl19": os.path.join(REPO_ROOT, "data/msmarco-full/TREC_DL_2019/qrel_binary.json"),
    "dl20": os.path.join(REPO_ROOT, "data/msmarco-full/TREC_DL_2020/qrel_binary.json"),
    "dev": os.path.join(REPO_ROOT, "data/msmarco-full/dev_qrel.json"),
}
SPLIT_PRIMARY_METRIC = {"dl19": "ndcg_cut_10", "dl20": "ndcg_cut_10", "dev": "recip_rank"}


def truncate_run(run, k):
    out = {}
    for qid, docs in run.items():
        ordered = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out[qid] = dict(ordered)
    return out


def score_run(run_path, split):
    with open(run_path) as f:
        run = json.load(f)
    ds = SPLIT_DATASET[split]

    with open(SPLIT_BINARY_QREL[split]) as f:
        qrel_bin = json.load(f)
    with open(SPLIT_GRADED_QREL[split]) as f:
        qrel_graded = json.load(f)

    truncated = truncate_run(run, 10)

    mrr_eval = pytrec_eval.RelevanceEvaluator(qrel_bin, {"recip_rank"})
    mrr_res = mrr_eval.evaluate(truncated)
    n_mrr = len(mrr_res)
    mrr10 = sum(d["recip_rank"] for d in mrr_res.values()) / n_mrr if n_mrr else float("nan")

    rec_eval = pytrec_eval.RelevanceEvaluator(qrel_bin, {"recall"})
    rec_res = rec_eval.evaluate(run)
    n_rec = len(rec_res)
    recall10 = sum(d.get("recall_10", 0.0) for d in rec_res.values()) / n_rec if n_rec else float("nan")

    result = {"mrr_10": mrr10, "recall_10": recall10, "n": n_mrr}

    if split in ("dl19", "dl20"):
        ndcg_eval = pytrec_eval.RelevanceEvaluator(qrel_graded, {"ndcg_cut"})
        ndcg_res = ndcg_eval.evaluate(run)
        n_ndcg = len(ndcg_res)
        ndcg10 = sum(d["ndcg_cut_10"] for d in ndcg_res.values()) / n_ndcg if n_ndcg else float("nan")
        result["ndcg_cut_10"] = ndcg10
        result["n_ndcg"] = n_ndcg

    return result


def run_json_path(split, condition_dir):
    ds = SPLIT_DATASET[split]
    return os.path.join(OUT_ROOT, split, condition_dir, "smt_ret", ds, "run.json")


def main(splits):
    all_results = {}
    for split in splits:
        all_results[split] = {}

        clean_path = run_json_path(split, "clean")
        if not os.path.exists(clean_path):
            print(f"MISSING clean run for {split}: {clean_path}", file=sys.stderr)
            continue
        clean_scores = score_run(clean_path, split)
        all_results[split]["clean"] = clean_scores

        for attack in ATTACK_METHODS:
            per_seed = []
            for seed in SEEDS:
                cond_dir = os.path.join(attack, f"seed_{seed}")
                rp = run_json_path(split, cond_dir)
                if not os.path.exists(rp):
                    print(f"MISSING: {split}/{cond_dir} -> {rp}", file=sys.stderr)
                    continue
                sc = score_run(rp, split)
                sc["seed"] = seed
                per_seed.append(sc)
            all_results[split][attack] = per_seed

    return all_results


if __name__ == "__main__":
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["dl19", "dl20"]
    results = main(splits)

    primary_key = {"dl19": "ndcg_cut_10", "dl20": "ndcg_cut_10", "dev": "mrr_10"}
    primary_label = {"dl19": "NDCG@10", "dl20": "NDCG@10", "dev": "MRR@10"}

    csv_rows = [("split", "clean_primary", "clean_recall_10", "attack",
                 "primary_metric", "primary_mean", "primary_std",
                 "recall10_mean", "recall10_std", "delta_primary", "n_seeds")]
    md_lines = ["# RQ2 re-decode summary (Table 3 pipeline)", ""]

    for split in splits:
        if "clean" not in results.get(split, {}):
            continue
        clean = results[split]["clean"]
        pk = primary_key[split]
        pk_label = primary_label[split]

        header = f"\n===== {split.upper()} ====="
        clean_line = f"CLEAN: {pk}={clean[pk]:.6f}  recall_10={clean['recall_10']:.6f}  (n={clean['n']})"
        print(header)
        print(clean_line)
        print(f"{'attack':<12} {'seed_mean_'+pk:<18} {'std':<10} {'recall10_mean':<15} {'std':<10} {'delta_'+pk:<12}")

        md_lines.append(f"## {split.upper()}")
        md_lines.append("")
        md_lines.append(f"Clean: **{pk_label} = {clean[pk]:.4f}**, Recall@10 = {clean['recall_10']:.4f} (n={clean['n']})")
        md_lines.append("")
        md_lines.append(f"| Attack | {pk_label} mean±std | Δ vs clean | Recall@10 mean±std | n seeds |")
        md_lines.append("|---|---|---|---|---|")

        for attack in ATTACK_METHODS:
            seeds = results[split].get(attack, [])
            if not seeds:
                print(f"{attack:<12} NO DATA")
                md_lines.append(f"| {attack} | NO DATA | | | |")
                csv_rows.append((split, clean[pk], clean["recall_10"], attack, pk, "", "", "", "", "", 0))
                continue
            pk_vals = [s[pk] for s in seeds]
            rec_vals = [s["recall_10"] for s in seeds]
            pk_mean = stats.mean(pk_vals)
            pk_std = stats.pstdev(pk_vals) if len(pk_vals) > 1 else 0.0
            rec_mean = stats.mean(rec_vals)
            rec_std = stats.pstdev(rec_vals) if len(rec_vals) > 1 else 0.0
            delta = clean[pk] - pk_mean
            print(f"{attack:<12} {pk_mean:<18.6f} {pk_std:<10.6f} {rec_mean:<15.6f} {rec_std:<10.6f} {delta:<12.6f}")
            md_lines.append(
                f"| {attack} | {pk_mean:.4f} ± {pk_std:.4f} | {delta:+.4f} | "
                f"{rec_mean:.4f} ± {rec_std:.4f} | {len(seeds)} |"
            )
            csv_rows.append((split, clean[pk], clean["recall_10"], attack, pk,
                              pk_mean, pk_std, rec_mean, rec_std, delta, len(seeds)))
        md_lines.append("")

    json_path = os.path.join(OUT_ROOT, "scored_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(OUT_ROOT, "rq2_redecode_summary.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    md_path = os.path.join(OUT_ROOT, "rq2_redecode_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

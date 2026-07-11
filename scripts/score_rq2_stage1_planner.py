#!/usr/bin/env python3
"""
Score Stage 1 (SimulOnly / lexical planner) outputs from
RQ2_REDECODE_2026-07-07, alongside Stage 2 (end-to-end PAG) for comparison.

For each split (dl19, dl20, dev) and each of the 26 conditions (clean + 5
attacks x 5 seeds), computes BOTH NDCG@10 (graded qrel) and MRR@10 (binary /
dev qrel) on:
  - Stage 1: <split>/<condition>/lex_ret/<dataset>/run.json   (planner-only)
  - Stage 2: <split>/<condition>/smt_ret/<dataset>/run.json   (end-to-end PAG)

Reports two forms of delta per attack:
  - delta_of_means:      clean_mean - perturbed_mean   (matches README convention)
  - per_seed_delta_mean/std: mean/std of (clean - perturbed_i) computed per seed,
                              i.e. the LaTeX table's mu +/- sigma of Delta.
    NOTE: clean has no seed axis (it is a single re-decode), so the per-seed
    delta here is (single clean value - perturbed_seed_i), and its std is
    therefore identical to the std of the perturbed values themselves.

Writes stage1_stage2_scored_results.json and a flat CSV alongside the
existing Stage-2-only artifacts, without modifying them.
"""
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


def truncate_run(run, k):
    out = {}
    for qid, docs in run.items():
        ordered = sorted(docs.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out[qid] = dict(ordered)
    return out


def score_run(run_path, split):
    with open(run_path) as f:
        run = json.load(f)

    with open(SPLIT_BINARY_QREL[split]) as f:
        qrel_bin = json.load(f)
    with open(SPLIT_GRADED_QREL[split]) as f:
        qrel_graded = json.load(f)

    truncated = truncate_run(run, 10)

    mrr_eval = pytrec_eval.RelevanceEvaluator(qrel_bin, {"recip_rank"})
    mrr_res = mrr_eval.evaluate(truncated)
    n_mrr = len(mrr_res)
    mrr10 = sum(d["recip_rank"] for d in mrr_res.values()) / n_mrr if n_mrr else float("nan")

    ndcg_eval = pytrec_eval.RelevanceEvaluator(qrel_graded, {"ndcg_cut"})
    ndcg_res = ndcg_eval.evaluate(run)
    n_ndcg = len(ndcg_res)
    ndcg10 = sum(d["ndcg_cut_10"] for d in ndcg_res.values()) / n_ndcg if n_ndcg else float("nan")

    return {"mrr_10": mrr10, "ndcg_cut_10": ndcg10, "n_mrr": n_mrr, "n_ndcg": n_ndcg}


def run_json_path(split, condition_dir, stage_subdir):
    ds = SPLIT_DATASET[split]
    return os.path.join(OUT_ROOT, split, condition_dir, stage_subdir, ds, "run.json")


def score_stage(split, stage_subdir):
    clean_path = run_json_path(split, "clean", stage_subdir)
    if not os.path.exists(clean_path):
        print(f"MISSING clean {stage_subdir} run for {split}: {clean_path}", file=sys.stderr)
        return None
    result = {"clean": score_run(clean_path, split)}

    for attack in ATTACK_METHODS:
        per_seed = []
        for seed in SEEDS:
            cond_dir = os.path.join(attack, f"seed_{seed}")
            rp = run_json_path(split, cond_dir, stage_subdir)
            if not os.path.exists(rp):
                print(f"MISSING: {split}/{cond_dir}/{stage_subdir} -> {rp}", file=sys.stderr)
                continue
            sc = score_run(rp, split)
            sc["seed"] = seed
            per_seed.append(sc)
        result[attack] = per_seed
    return result


def main(splits):
    all_results = {}
    for split in splits:
        all_results[split] = {
            "stage1_lex_ret": score_stage(split, "lex_ret"),
            "stage2_smt_ret": score_stage(split, "smt_ret"),
        }
    return all_results


def summarize(results, splits):
    csv_rows = [("split", "stage", "clean_ndcg10", "clean_mrr10", "attack",
                 "ndcg10_mean", "ndcg10_std", "ndcg10_delta_of_means",
                 "ndcg10_perseed_delta_mean", "ndcg10_perseed_delta_std",
                 "mrr10_mean", "mrr10_std", "mrr10_delta_of_means",
                 "mrr10_perseed_delta_mean", "mrr10_perseed_delta_std", "n_seeds")]

    for split in splits:
        for stage_key, stage_label in [("stage1_lex_ret", "Stage1_SimulOnly"),
                                        ("stage2_smt_ret", "Stage2_PAG")]:
            stage_data = results[split].get(stage_key)
            if not stage_data or "clean" not in stage_data:
                continue
            clean = stage_data["clean"]
            print(f"\n===== {split.upper()} / {stage_label} =====")
            print(f"CLEAN: NDCG@10={clean['ndcg_cut_10']:.6f}  MRR@10={clean['mrr_10']:.6f}")

            for attack in ATTACK_METHODS:
                seeds = stage_data.get(attack, [])
                if not seeds:
                    csv_rows.append((split, stage_label, clean["ndcg_cut_10"], clean["mrr_10"],
                                      attack, "", "", "", "", "", "", "", "", "", "", 0))
                    continue

                ndcg_vals = [s["ndcg_cut_10"] for s in seeds]
                mrr_vals = [s["mrr_10"] for s in seeds]

                ndcg_mean = stats.mean(ndcg_vals)
                ndcg_std = stats.pstdev(ndcg_vals) if len(ndcg_vals) > 1 else 0.0
                mrr_mean = stats.mean(mrr_vals)
                mrr_std = stats.pstdev(mrr_vals) if len(mrr_vals) > 1 else 0.0

                ndcg_delta_of_means = clean["ndcg_cut_10"] - ndcg_mean
                mrr_delta_of_means = clean["mrr_10"] - mrr_mean

                # per-seed delta: clean is a single value (no seed axis), so
                # per-seed delta_i = clean - perturbed_i; mean/std of that
                # series. Mean equals delta_of_means; std equals std of the
                # perturbed values themselves (clean is constant per seed).
                ndcg_perseed_deltas = [clean["ndcg_cut_10"] - v for v in ndcg_vals]
                mrr_perseed_deltas = [clean["mrr_10"] - v for v in mrr_vals]
                ndcg_perseed_mean = stats.mean(ndcg_perseed_deltas)
                ndcg_perseed_std = stats.pstdev(ndcg_perseed_deltas) if len(ndcg_perseed_deltas) > 1 else 0.0
                mrr_perseed_mean = stats.mean(mrr_perseed_deltas)
                mrr_perseed_std = stats.pstdev(mrr_perseed_deltas) if len(mrr_perseed_deltas) > 1 else 0.0

                print(f"  {attack:<12} NDCG@10={ndcg_mean:.4f}±{ndcg_std:.4f} (Δ={ndcg_delta_of_means:+.4f})  "
                      f"MRR@10={mrr_mean:.4f}±{mrr_std:.4f} (Δ={mrr_delta_of_means:+.4f})")

                csv_rows.append((split, stage_label, clean["ndcg_cut_10"], clean["mrr_10"], attack,
                                  ndcg_mean, ndcg_std, ndcg_delta_of_means,
                                  ndcg_perseed_mean, ndcg_perseed_std,
                                  mrr_mean, mrr_std, mrr_delta_of_means,
                                  mrr_perseed_mean, mrr_perseed_std, len(seeds)))
    return csv_rows


if __name__ == "__main__":
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["dl19", "dl20", "dev"]
    results = main(splits)
    csv_rows = summarize(results, splits)

    json_path = os.path.join(OUT_ROOT, "stage1_stage2_scored_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(OUT_ROOT, "stage1_stage2_summary.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

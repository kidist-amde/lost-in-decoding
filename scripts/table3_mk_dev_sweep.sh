#!/bin/bash
#SBATCH --job-name=table3_mk_dev
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=experiments/table3_mk_sweep/logs/%x-%j.out
#SBATCH --error=experiments/table3_mk_sweep/logs/%x-%j.err
#SBATCH --chdir=/gpfs/work4/0/prjs1037/dpo-exp/pag-repro/

# =============================================================================
# Table 3 reproduction (DEV ONLY): effectiveness & efficiency with different m and k
# on MSMARCO Dev.
#
# Sweeps:  m ∈ {16, 32, 64}  ×  k ∈ {10, 100}
# =============================================================================

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

LOG_DIR=experiments/table3_mk_sweep/logs
mkdir -p "$LOG_DIR"

# ── Paths (shared across all runs) ──────────────────────────────────────────
model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
pretrained_path="$model_dir/checkpoint/"

smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
smt_docid_to_smtid_path="$smt_data_dir/aq_smtid/docid_to_tokenids.json"

lex_base_dir="./data/experiments-splade/t5-splade-0-12l"

# DEV ONLY
q_collection_paths='["./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json"]'

# This must match the folder name produced by get_dataset_name(dev_queries_path)
# If your output is lex_out_dir/dev_queries/run.json then set DEV_NAME="dev_queries"
DEV_NAME="MSMARCO"

max_new_token_for_docid=8
lex_topk=1000
lexical_constrained=lexical_tmp_rescore

# ── Sweep ────────────────────────────────────────────────────────────────────
for m in 16 32 64; do
    echo ""
    echo "============================================================"
    echo " m = $m"
    echo "============================================================"

    # Resolve BOW docid file for this m value
    if [ "$m" -eq 64 ]; then
        lex_docid_to_smtid_path="$lex_base_dir/top_bow/docid_to_tokenids.json"
    else
        lex_docid_to_smtid_path="$lex_base_dir/top_bow_${m}/docid_to_tokenids.json"
    fi

    if [ ! -f "$lex_docid_to_smtid_path" ]; then
        echo "ERROR: BOW file missing for m=$m: $lex_docid_to_smtid_path"
        exit 1
    fi

    # ── Output directories (m-tagged) ────────────────────────────────────
    lex_out_dir="$model_dir/table3_dev/m_${m}/lex_ret_${lex_topk}"

    # ── Stage 1: Lexical retrieval (simultaneous decoding) ───────────────
    stage1_marker="$lex_out_dir/$DEV_NAME/run.json"
    if [ -f "$stage1_marker" ]; then
        echo "[Stage 1] m=$m — already complete, skipping."
    else
        echo "[Stage 1] m=$m — running lexical retrieval (top-$lex_topk)..."
        python -m t5_pretrainer.evaluate \
            --pretrained_path="$pretrained_path" \
            --out_dir="$lex_out_dir" \
            --lex_out_dir="$lex_out_dir" \
            --task=lexical_constrained_retrieve_and_rerank \
            --q_collection_paths="$q_collection_paths" \
            --batch_size=8 \
            --topk="$lex_topk" \
            --lex_docid_to_smtid_path="$lex_docid_to_smtid_path" \
            --smt_docid_to_smtid_path="$smt_docid_to_smtid_path" \
            --max_length=128 \
            --eval_qrel_path="$eval_qrel_path" \
            2>&1 | tee "$LOG_DIR/stage1_dev_m${m}.log"
    fi

    for k in 10 100; do
        echo ""
        echo "------------------------------------------------------------"
        echo " m = $m, k = $k"
        echo "------------------------------------------------------------"

        smt_out_dir="$lex_out_dir/ltmp_smt_ret_${k}"

        # ── Stage 2: Sequential decoding ────────────────────────────────
        stage2_marker="$smt_out_dir/$DEV_NAME/run.json"
        if [ -f "$stage2_marker" ]; then
            echo "[Stage 2] m=$m, k=$k — already complete, skipping."
        else
            echo "[Stage 2] m=$m, k=$k — running sequential decoding (beam=$k)..."
            python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
                --pretrained_path="$pretrained_path" \
                --out_dir="$smt_out_dir" \
                --lex_out_dir="$lex_out_dir" \
                --task=lexical_constrained_retrieve_and_rerank_2 \
                --q_collection_paths="$q_collection_paths" \
                --batch_size=16 \
                --topk="$k" \
                --lex_docid_to_smtid_path="$lex_docid_to_smtid_path" \
                --smt_docid_to_smtid_path="$smt_docid_to_smtid_path" \
                --max_length=128 \
                --max_new_token_for_docid="$max_new_token_for_docid" \
                --eval_qrel_path="$eval_qrel_path" \
                --lex_constrained="$lexical_constrained" \
                2>&1 | tee "$LOG_DIR/stage2_dev_m${m}_k${k}.log"
        fi

        # ── Stage 3: Merge shards & evaluate ─────────────────────────────
        echo "[Stage 3] m=$m, k=$k — merging and evaluating..."
        python -m t5_pretrainer.evaluate \
            --task=lexical_constrained_retrieve_and_rerank_3 \
            --out_dir="$smt_out_dir" \
            --q_collection_paths="$q_collection_paths" \
            --eval_qrel_path="$eval_qrel_path" \
            2>&1 | tee "$LOG_DIR/stage3_dev_m${m}_k${k}.log"
    done
done

# ── Collect results ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Table 3 Results Summary (MSMARCO Dev ONLY)"
echo "============================================================"
echo ""
printf "%-6s %-6s %-12s %-12s\n" "m" "k" "MRR@10" "Recall@10"
printf "%-6s %-6s %-12s %-12s\n" "----" "----" "----------" "----------"

for m in 16 32 64; do
    lex_dir="$model_dir/table3_dev/m_${m}/lex_ret_${lex_topk}"
    for k in 10 100; do
        perf_file="$lex_dir/ltmp_smt_ret_${k}/perf_all_datasets.json"
        if [ -f "$perf_file" ]; then
            mrr=$(python -c "import json; d=json.load(open('$perf_file')); print(f\"{d['MSMARCO']['mrr_10']:.4f}\")")
            rec=$(python -c "import json; d=json.load(open('$perf_file')); print(f\"{d['MSMARCO']['recall_10']:.4f}\")")
            printf "%-6s %-6s %-12s %-12s\n" "$m" "$k" "$mrr" "$rec"
        else
            printf "%-6s %-6s %-12s %-12s\n" "$m" "$k" "N/A" "N/A"
        fi
    done
done

echo ""
echo "Paper-reported values for comparison (MSMARCO Dev):"
echo "  m=16, k=10:   MRR@10=.342   Recall@10=.577"
echo "  m=32, k=10:   MRR@10=.367   Recall@10=.626"
echo "  m=64, k=10:   MRR@10=.379   Recall@10=.641"
echo "  m=16, k=100:  MRR@10=.355   Recall@10=.620"
echo "  m=32, k=100:  MRR@10=.372   Recall@10=.652"
echo "  m=64, k=100:  MRR@10=.385   Recall@10=.670"
echo ""
echo "Logs in: $LOG_DIR/"
echo "Done."

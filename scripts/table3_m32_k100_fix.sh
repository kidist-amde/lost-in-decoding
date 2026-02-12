#!/bin/bash
#SBATCH --job-name=table3_m32k100_fix
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=experiments/table3_mk_sweep/logs/%x-%j.out
#SBATCH --error=experiments/table3_mk_sweep/logs/%x-%j.err
#SBATCH --chdir=.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pag-env

model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
pretrained_path="$model_dir/checkpoint/"

smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
smt_docid_to_smtid_path="$smt_data_dir/aq_smtid/docid_to_tokenids.json"

lex_docid_to_smtid_path="./data/experiments-splade/t5-splade-0-12l/top_bow_32/docid_to_tokenids.json"
lex_out_dir="$model_dir/table3/m_32/lex_ret_1000"
out_dir="$lex_out_dir/ltmp_smt_ret_100_fix"

q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'

python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
  --pretrained_path="$pretrained_path" \
  --out_dir="$out_dir" \
  --lex_out_dir="$lex_out_dir" \
  --task=lexical_constrained_retrieve_and_rerank_2 \
  --q_collection_paths="$q_collection_paths" \
  --batch_size=16 \
  --topk=100 \
  --lex_docid_to_smtid_path="$lex_docid_to_smtid_path" \
  --smt_docid_to_smtid_path="$smt_docid_to_smtid_path" \
  --max_length=128 \
  --max_new_token_for_docid=8 \
  --eval_qrel_path="$eval_qrel_path" \
  --lex_constrained=lexical_tmp_rescore

python -m t5_pretrainer.evaluate \
  --task=lexical_constrained_retrieve_and_rerank_3 \
  --out_dir="$out_dir" \
  --q_collection_paths="$q_collection_paths" \
  --eval_qrel_path="$eval_qrel_path"


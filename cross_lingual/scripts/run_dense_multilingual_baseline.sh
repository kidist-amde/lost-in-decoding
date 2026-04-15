#!/bin/bash
#SBATCH --job-name=dense_rq3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_h100
#SBATCH --time=01-12:00:00
#SBATCH --output=experiments/RQ3_crosslingual/%x-%j.out
#SBATCH --error=experiments/RQ3_crosslingual/%x-%j.err

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/RQ3_crosslingual/dense_multilingual_baseline}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pag-env}"

LANGUAGES="${1:-${LANGUAGES:-fr de zh nl}}"
SPLITS="${2:-${SPLITS:-dev}}"
MODEL="${MODEL:-microsoft/harrier-oss-v1-27b}"
CORPUS_EMB_CACHE="${CORPUS_EMB_CACHE:-$OUTPUT_DIR/corpus_embs.npy}"

CORPUS_BATCH_SIZE="${CORPUS_BATCH_SIZE:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
CORPUS_MAX_LENGTH="${CORPUS_MAX_LENGTH:-512}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-128}"
POOLING="${POOLING:-last}"
QUERY_PREFIX="${QUERY_PREFIX:-Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: }"
PASSAGE_PREFIX="${PASSAGE_PREFIX:-}"
TOPK="${TOPK:-100}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
FAISS_GPU="${FAISS_GPU:-1}"
FAISS_GPU_DEVICES="${FAISS_GPU_DEVICES:-1,2,3}"

FAISS_ARGS=()
if [ "$FAISS_GPU" = "1" ]; then
    FAISS_ARGS+=(--faiss_gpu --faiss_gpu_devices "$FAISS_GPU_DEVICES")
fi

echo "========================================"
echo "Dense multilingual RQ3 baseline"
echo "  model:            $MODEL"
echo "  languages:        $LANGUAGES"
echo "  splits:           $SPLITS"
echo "  output_dir:       $OUTPUT_DIR"
echo "  corpus_emb_cache: $CORPUS_EMB_CACHE"
echo "========================================"
nvidia-smi || true

python -m cross_lingual.evaluation.dense_multilingual_baseline \
    --model "$MODEL" \
    --languages $LANGUAGES \
    --splits $SPLITS \
    --output_dir "$OUTPUT_DIR" \
    --corpus_emb_cache "$CORPUS_EMB_CACHE" \
    --corpus_batch_size "$CORPUS_BATCH_SIZE" \
    --query_batch_size "$QUERY_BATCH_SIZE" \
    --corpus_max_length "$CORPUS_MAX_LENGTH" \
    --query_max_length "$QUERY_MAX_LENGTH" \
    --pooling "$POOLING" \
    --query_prefix "$QUERY_PREFIX" \
    --passage_prefix "$PASSAGE_PREFIX" \
    --topk "$TOPK" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${FAISS_ARGS[@]}"

echo "Done. Summary: $OUTPUT_DIR/summary.csv"

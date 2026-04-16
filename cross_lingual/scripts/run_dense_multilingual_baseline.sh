#!/bin/bash
#SBATCH --job-name=dense_rq3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_h100
#SBATCH --time=5:00:00
#SBATCH --output=experiments/RQ3_crosslingual/dense_multilingual_baseline/%x-%j.out
#SBATCH --error=experiments/RQ3_crosslingual/dense_multilingual_baseline/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/../..}")"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/RQ3_crosslingual/dense_multilingual_baseline}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pag-env}"

LANGUAGES="${1:-${LANGUAGES:-nl fr de zh}}"
SPLITS="${2:-${SPLITS:-dev}}"
MODEL="${MODEL:-intfloat/multilingual-e5-base}"
MODEL_CACHE_STEM="${MODEL//\//_}"
MODEL_CACHE_STEM="${MODEL_CACHE_STEM//:/_}"
CORPUS_EMB_CACHE="${CORPUS_EMB_CACHE:-$OUTPUT_DIR/${MODEL_CACHE_STEM}_corpus_embs.npy}"

CORPUS_BATCH_SIZE="${CORPUS_BATCH_SIZE:-512}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-512}"
TOPK="${TOPK:-100}"
DEVICE="${DEVICE:-cuda:0}"
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
    --topk "$TOPK" \
    --device "$DEVICE" \
    "${FAISS_ARGS[@]}"

echo "Done. Summary: $OUTPUT_DIR/summary.csv"

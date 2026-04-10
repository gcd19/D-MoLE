#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DMOLE_ENV="${DMOLE_ENV:-/home/jpgtex/.venvs/dmole-research}"
PYTHON_BIN="${DMOLE_ENV}/bin/python"

export PATH="${DMOLE_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-34229}"
export LAUNCHER="${LAUNCHER:-pytorch}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export DMOLE_MODEL_DTYPE="${DMOLE_MODEL_DTYPE:-float32}"
BF16_FLAG="${DMOLE_BF16:-False}"
ZC_FORCE_IMAGE_SIZE="${DMOLE_ZC_FORCE_IMAGE_SIZE:-224}"
ZC_MAX_DYNAMIC_PATCH="${DMOLE_ZC_MAX_DYNAMIC_PATCH:-1}"
ZC_MAX_SEQ_LENGTH="${DMOLE_ZC_MAX_SEQ_LENGTH:-512}"
ZC_BATCH_SIZE="${DMOLE_ZC_BATCH_SIZE:-1}"
ZC_GRAD_CHECKPOINT="${DMOLE_ZC_GRAD_CHECKPOINT:-True}"

TASK_NAME="vizwiz_caption"
TASK_ID=1
SAMPLE_SIZE="${DMOLE_SAMPLE_SIZE:-256}"
META_PATH="${REPO_ROOT}/shell/dmole_internal/vizwiz_caption_minimal.json"
EMBEDDING_PATH="${REPO_ROOT}/embeddings/vizwiz_caption_minimal/embeddings.pt"
AUTOENCODER_ROOT="${REPO_ROOT}/autoencoder_models"
SCORE_PATH="${REPO_ROOT}/results/zc_scores/1_InternVL2-2B_vizwiz_caption_score.csv"
ARCH_PATH="${REPO_ROOT}/dmole_arch/1_InternVL2-2B_vizwiz_caption_arch.json"
MODEL_PATH="${REPO_ROOT}/pretrained/InternVL2-2B"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "FATAL: required file is missing: $path" >&2
    exit 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "FATAL: executable is missing: $path" >&2
    exit 1
  fi
}

require_executable "$PYTHON_BIN"
require_file "${MODEL_PATH}/config.json"

"$PYTHON_BIN" "${REPO_ROOT}/scripts/preprocess/reconstruct_public_dmole_assets.py" \
  --task vizwiz_caption \
  --repo-root "${REPO_ROOT}" \
  --sample-size "${SAMPLE_SIZE}"

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/internvl/scorer/compute_seq_rep.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir none \
  --conv_style "internlm2-chat" \
  --meta_path "${META_PATH}" \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --bf16 "${BF16_FLAG}" \
  --num_train_epochs 1 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2

require_file "${EMBEDDING_PATH}"

"$PYTHON_BIN" "${REPO_ROOT}/scripts/preprocess/train_autoencoder_single_task.py" \
  --task-name "${TASK_NAME}" \
  --embedding-path "${EMBEDDING_PATH}" \
  --output-dir "${AUTOENCODER_ROOT}"

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=1 \
  --master_port="$((MASTER_PORT + 1))" \
  "${REPO_ROOT}/internvl/scorer/compute_zc_score.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir none \
  --conv_style "internlm2-chat" \
  --meta_path "${META_PATH}" \
  --force_image_size "${ZC_FORCE_IMAGE_SIZE}" \
  --max_dynamic_patch "${ZC_MAX_DYNAMIC_PATCH}" \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --vision_select_layer -1 \
  --bf16 "${BF16_FLAG}" \
  --num_train_epochs 1 \
  --max_seq_length "${ZC_MAX_SEQ_LENGTH}" \
  --per_device_train_batch_size "${ZC_BATCH_SIZE}" \
  --grad_checkpoint "${ZC_GRAD_CHECKPOINT}" \
  --zc_proxy_score_portion 0.01 \
  --zc_proxy_score_save_path "${SCORE_PATH}"

require_file "${SCORE_PATH}"

"$PYTHON_BIN" "${REPO_ROOT}/scripts/preprocess/get_dmole_arch_single_task.py" \
  --task-name "${TASK_NAME}" \
  --task-id "${TASK_ID}" \
  --score-path "${SCORE_PATH}" \
  --output-path "${ARCH_PATH}"

require_file "${ARCH_PATH}"
require_file "${AUTOENCODER_ROOT}/${TASK_NAME}/autoencoder.pt"
require_file "${AUTOENCODER_ROOT}/${TASK_NAME}/threshold.txt"
require_file "${AUTOENCODER_ROOT}/reconstruction_loss_quantiles.csv"

echo "minimal VizWiz caption D-MoLE preprocessing completed successfully."

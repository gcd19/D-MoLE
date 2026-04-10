#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DMOLE_ENV="${DMOLE_ENV:-/home/jpgtex/.venvs/dmole-research}"
export PATH="${DMOLE_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-34229}"
export LAUNCHER="${LAUNCHER:-pytorch}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export ENABLE_EVALUATION="${ENABLE_EVALUATION:-false}"

PYTHON_BIN="${DMOLE_ENV}/bin/python"
OUTPUT_ROOT="${SOULFORGE_OUTPUT_DIR:?SOULFORGE_OUTPUT_DIR must be set}"
BASE_MODEL_DIR="${DMOLE_BASE_MODEL_DIR:-${REPO_ROOT}/pretrained/InternVL2-2B}"
DATA_DIR="${DMOLE_DATA_DIR:-${REPO_ROOT}/data}"
ARCH_DIR="${DMOLE_ARCH_DIR:-${REPO_ROOT}/dmole_arch}"
AUTOENCODER_DIR="${DMOLE_AUTOENCODER_DIR:-${REPO_ROOT}/autoencoder_models}"
META_PATH="${DMOLE_META_PATH:-${REPO_ROOT}/shell/dmole/vizwiz_caption.json}"
TASK_NAME="${DMOLE_TASK_NAME:-vizwiz_caption}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK_NAME}_1gpu"
DMOLE_ARCH_PATH="${DMOLE_ARCH_PATH:-${ARCH_DIR}/1_InternVL2-2B_${TASK_NAME}_arch.json}"
TRAIN_LOG="${OUTPUT_DIR}/training_log.txt"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "FATAL: required file is missing: $path" >&2
    exit 1
  fi
}

require_nonempty_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "FATAL: required directory is missing: $path" >&2
    exit 1
  fi
  if ! find "$path" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "FATAL: required directory is empty: $path" >&2
    exit 1
  fi
}

require_findable_file() {
  local root="$1"
  local pattern="$2"
  if ! find "$root" -type f -name "$pattern" | read -r _; then
    echo "FATAL: expected to find $pattern beneath $root" >&2
    exit 1
  fi
}

require_interpreter() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: dedicated D-MoLE interpreter is missing: $PYTHON_BIN" >&2
    exit 1
  fi
}

verify_runtime() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util

required = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "bitsandbytes",
    "trl",
    "accelerate",
    "deepspeed",
    "einops",
    "einops_exts",
    "timm",
    "shortuuid",
    "sentencepiece",
    "torchvision",
    "pydantic",
    "imageio",
    "decord",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"FATAL: dedicated D-MoLE environment is missing modules: {', '.join(missing)}")
PY
}

stage_checks() {
  require_nonempty_dir "$BASE_MODEL_DIR"
  require_nonempty_dir "$DATA_DIR"
  require_nonempty_dir "$ARCH_DIR"
  require_nonempty_dir "$AUTOENCODER_DIR"
  require_file "$META_PATH"
  require_file "$DMOLE_ARCH_PATH"
  require_file "${BASE_MODEL_DIR}/config.json"
  if ! find "$DATA_DIR" -type f \( -name '*.json' -o -name '*.jsonl' \) | read -r _; then
    echo "FATAL: missing any dataset manifest (.json/.jsonl) beneath $DATA_DIR" >&2
    exit 1
  fi
  if [[ ! -f "${AUTOENCODER_DIR}/reconstruction_loss_quantiles.csv" ]]; then
    echo "FATAL: missing reconstruction_loss_quantiles.csv in $AUTOENCODER_DIR" >&2
    exit 1
  fi
  require_findable_file "$AUTOENCODER_DIR" 'autoencoder.pt'
}

write_receipts() {
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
task_name = os.environ["TASK_NAME"]
dmole_arch_path = os.environ["DMOLE_ARCH_PATH"]
base_model_dir = os.environ["BASE_MODEL_DIR"]
meta_path = os.environ["META_PATH"]

router_payload = {
    "route": "dmole_minimal_1gpu",
    "task_name": task_name,
    "dmole_arch_path": dmole_arch_path,
    "base_model_dir": base_model_dir,
    "meta_path": meta_path,
}
expert_payload = {
    "task_name": task_name,
    "output_dir": str(output_dir),
    "training_log": str(output_dir / "training_log.txt"),
}

(output_dir / "dmole_router.json").write_text(json.dumps(router_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output_dir / "expert_manifest.json").write_text(json.dumps(expert_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

require_interpreter
verify_runtime
stage_checks

mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR TASK_NAME DMOLE_ARCH_PATH BASE_MODEL_DIR META_PATH

"$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/internvl/train/internvl_chat_finetune.py" \
  --model_name_or_path "${BASE_MODEL_DIR}" \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 8 \
  --use_backbone_lora 8 \
  --use_dmole True \
  --dmole_arch_path "${DMOLE_ARCH_PATH}" \
  --autoencoder_path "${AUTOENCODER_DIR}" \
  --task_id 1 \
  --vision_select_layer -1 \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "none" \
  2>&1 | tee "${TRAIN_LOG}"

write_receipts

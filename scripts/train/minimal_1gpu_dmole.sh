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
export DMOLE_REQUIRE_FINITE_TRAIN_SIGNAL="${DMOLE_REQUIRE_FINITE_TRAIN_SIGNAL:-1}"
export DMOLE_REQUIRE_FINITE_INITIAL_SIGNAL="${DMOLE_REQUIRE_FINITE_INITIAL_SIGNAL:-1}"

PYTHON_BIN="${DMOLE_ENV}/bin/python"
OUTPUT_ROOT="${SOULFORGE_OUTPUT_DIR:?SOULFORGE_OUTPUT_DIR must be set}"
BASE_MODEL_DIR="${DMOLE_BASE_MODEL_DIR:-${REPO_ROOT}/pretrained/InternVL2-2B}"
DATA_DIR="${DMOLE_DATA_DIR:-${REPO_ROOT}/data}"
ARCH_DIR="${DMOLE_ARCH_DIR:-${REPO_ROOT}/dmole_arch}"
AUTOENCODER_DIR="${DMOLE_AUTOENCODER_DIR:-${REPO_ROOT}/autoencoder_models}"
META_PATH="${DMOLE_META_PATH:-${REPO_ROOT}/shell/dmole_internal/vizwiz_caption_minimal.json}"
TASK_NAME="${DMOLE_TASK_NAME:-vizwiz_caption}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK_NAME}_1gpu"
DMOLE_ARCH_PATH="${DMOLE_ARCH_PATH:-${ARCH_DIR}/1_InternVL2-2B_${TASK_NAME}_arch.json}"
TRAIN_LOG="${OUTPUT_ROOT}/${TASK_NAME}_1gpu_training.log"
SIGNAL_PROBE_JSON="${OUTPUT_ROOT}/${TASK_NAME}_initial_signal_probe.json"

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

validate_training_signal() {
  if [[ "${DMOLE_REQUIRE_FINITE_TRAIN_SIGNAL}" == "0" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - <<'PY'
import ast
import math
import os
from pathlib import Path

train_log = Path(os.environ["TRAIN_LOG"])
if not train_log.is_file():
    raise SystemExit(f"FATAL: training log is missing: {train_log}")

metric_rows = []
for raw_line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
    start = raw_line.find("{")
    end = raw_line.rfind("}")
    if start < 0 or end <= start:
        continue
    candidate = raw_line[start : end + 1]
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        continue
    if isinstance(parsed, dict) and "loss" in parsed:
        metric_rows.append(parsed)

if not metric_rows:
    raise SystemExit("FATAL: no per-step training metrics were captured in the training log.")

finite_positive_loss = False
nonfinite_grad_rows = []
for row in metric_rows:
    loss_value = row.get("loss")
    grad_value = row.get("grad_norm")
    try:
        if math.isfinite(float(loss_value)) and float(loss_value) > 0.0:
            finite_positive_loss = True
    except (TypeError, ValueError):
        pass
    if grad_value is not None:
        try:
            grad_float = float(grad_value)
        except (TypeError, ValueError):
            nonfinite_grad_rows.append(row)
        else:
            if not math.isfinite(grad_float):
                nonfinite_grad_rows.append(row)

if not finite_positive_loss:
    raise SystemExit(
        "FATAL: degenerate training signal detected; no strictly positive finite loss was observed."
    )

if nonfinite_grad_rows:
    raise SystemExit(
        f"FATAL: non-finite grad_norm detected in {len(nonfinite_grad_rows)} logged step(s); "
        "authoritative training evidence is invalid."
    )
PY
}

probe_initial_signal() {
  if [[ "${DMOLE_REQUIRE_FINITE_INITIAL_SIGNAL}" == "0" ]]; then
    return 0
  fi

  "$PYTHON_BIN" "${REPO_ROOT}/scripts/train/probe_initial_dmole_signal.py" \
    --model-name-or-path "${BASE_MODEL_DIR}" \
    --meta-path "${META_PATH}" \
    --dmole-arch-path "${DMOLE_ARCH_PATH}" \
    --autoencoder-path "${AUTOENCODER_DIR}" \
    --output-json "${SIGNAL_PROBE_JSON}" \
    --force-image-size 448 \
    --down-sample-ratio 0.5 \
    --conv-style "internlm2-chat" \
    --min-dynamic-patch 1 \
    --max-dynamic-patch 6 \
    --use-llm-lora 8 \
    --use-backbone-lora 8 \
    --task-id 1 \
    --max-seq-length 2048
}

require_interpreter
verify_runtime
stage_checks

mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR TASK_NAME DMOLE_ARCH_PATH BASE_MODEL_DIR META_PATH TRAIN_LOG SIGNAL_PROBE_JSON

probe_initial_signal

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
  --eval_strategy "no" \
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
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "none" \
  2>&1 | tee "${TRAIN_LOG}"

validate_training_signal
write_receipts

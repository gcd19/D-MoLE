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
export DMOLE_REQUIRE_LONESTAR_PHYSICS="${DMOLE_REQUIRE_LONESTAR_PHYSICS:-1}"
export DMOLE_FAIL_ON_SANITIZED_SCORE="${DMOLE_FAIL_ON_SANITIZED_SCORE:-1}"
BF16_FLAG="${DMOLE_BF16:-False}"
SEQ_FORCE_IMAGE_SIZE="${DMOLE_SEQ_FORCE_IMAGE_SIZE:-224}"
SEQ_MAX_DYNAMIC_PATCH="${DMOLE_SEQ_MAX_DYNAMIC_PATCH:-1}"
SEQ_MAX_SEQ_LENGTH="${DMOLE_SEQ_MAX_SEQ_LENGTH:-1024}"
SEQ_BATCH_SIZE="${DMOLE_SEQ_BATCH_SIZE:-1}"
ZC_FORCE_IMAGE_SIZE="${DMOLE_ZC_FORCE_IMAGE_SIZE:-224}"
ZC_MAX_DYNAMIC_PATCH="${DMOLE_ZC_MAX_DYNAMIC_PATCH:-1}"
ZC_MAX_SEQ_LENGTH="${DMOLE_ZC_MAX_SEQ_LENGTH:-1024}"
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

require_python_module() {
  local module_name="$1"
  if ! "$PYTHON_BIN" - "$module_name" "${REPO_ROOT}" <<'PY'
import importlib
import importlib.util
import os
import sys
from pathlib import Path

def load_lonestar_physics_from_candidates(repo_root: Path) -> bool:
    candidate_paths = []
    explicit_path = os.environ.get("LONESTAR_PHYSICS_EXTENSION", "").strip()
    if explicit_path:
        candidate_paths.append(Path(explicit_path))
    workspace_root = repo_root.parent
    candidate_paths.extend(
        [
            workspace_root / "lonestar-physics" / "target" / "maturin" / "liblonestar_physics.so",
            workspace_root / "lonestar-physics" / "target" / "release" / "liblonestar_physics.so",
            workspace_root / "lonestar-physics" / "target" / "debug" / "liblonestar_physics.so",
        ]
    )
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("lonestar_physics", candidate_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["lonestar_physics"] = module
        spec.loader.exec_module(module)
        return True
    return False

module_name = sys.argv[1]
repo_root = Path(sys.argv[2]).resolve()

try:
    importlib.import_module(module_name)
except Exception:
    if module_name != "lonestar_physics" or not load_lonestar_physics_from_candidates(
        repo_root
    ):
        raise
PY
  then
    echo "FATAL: required Python module is unavailable: ${module_name}" >&2
    exit 1
  fi
}

require_executable "$PYTHON_BIN"
require_file "${MODEL_PATH}/config.json"
if [[ "${DMOLE_REQUIRE_LONESTAR_PHYSICS}" == "1" ]]; then
  require_python_module "lonestar_physics"
fi

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
  --force_image_size "${SEQ_FORCE_IMAGE_SIZE}" \
  --max_dynamic_patch "${SEQ_MAX_DYNAMIC_PATCH}" \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --bf16 "${BF16_FLAG}" \
  --num_train_epochs 1 \
  --max_seq_length "${SEQ_MAX_SEQ_LENGTH}" \
  --per_device_train_batch_size "${SEQ_BATCH_SIZE}"

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
require_file "${SCORE_PATH%.csv}.manifest.json"

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

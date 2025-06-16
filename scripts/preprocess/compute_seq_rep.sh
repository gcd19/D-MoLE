#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# Calculate number of GPUs
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
GPUS=$((GPUS + 1))

# Basic configuration
PRETRAINED_MODEL_PATH="pretrained/InternVL2-2B"

# Task configurations: name:metapath:batch_size
TASK_CONFIGS=(
  "vizwiz_caption:./shell/dmole/vizwiz_caption.json:128"
  "skvg:./shell/dmole/skvg.json:128"
  "textcaps:./shell/dmole/textcaps.json:128"
  "iconqa:./shell/dmole/iconqa.json:128"
  "ocrvqa:./shell/dmole/ocrvqa.json:128"
  "flickr30k:./shell/dmole/flickr30k.json:128"
  "vizwiz:./shell/dmole/vizwiz.json:128"
  "kvqa:./shell/dmole/kvqa.json:128"
  "pmcvqa:./shell/dmole/pmcvqa.json:128"
)

# Function to parse task configuration
parse_task_config() {
  local config=$1
  IFS=":" read -r TASK_NAME META_PATH BATCH_SIZE <<< "$config"
}

# Main loop to compute embeddings
for config in "${TASK_CONFIGS[@]}"; do
  # Skip commented tasks
  if [[ $config == \#* ]]; then
    continue
  fi
  
  # Parse current task configuration
  parse_task_config "$config"
  
  echo "=========================================="
  echo "Computing embeddings for: $TASK_NAME"
  echo "Meta Path: $META_PATH"
  echo "Batch Size: $BATCH_SIZE"
  echo "=========================================="
  
  # Compute sequence representations
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/scorer/compute_seq_rep.py \
    --model_name_or_path ${PRETRAINED_MODEL_PATH} \
    --output_dir none \
    --conv_style "internlm2-chat" \
    --meta_path ${META_PATH} \
    --force_image_size 448 \
    --max_dynamic_patch 6 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --freeze_llm True \
    --freeze_mlp True \
    --freeze_backbone True \
    --vision_select_layer -1 \
    --bf16 True \
    --num_train_epochs 1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    2>&1 | tee "embeddings/${TASK_NAME}/training_log.txt"

  echo "Embedding computation for $TASK_NAME completed!"
  echo ""
done

echo "All embedding computations completed successfully!"
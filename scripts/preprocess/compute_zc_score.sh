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
BASE_MODEL="InternVL2-2B"
PRETRAINED_MODEL_PATH="pretrained/${BASE_MODEL}"
OUTPUT_DIR="results/zc_scores"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Task configurations: name:metapath:batch_size
TASK_CONFIGS=(
  "vizwiz_caption:./shell/dmole/vizwiz_caption.json:2"
  "skvg:./shell/dmole/skvg.json:2"
  "textcaps:./shell/dmole/textcaps.json:2"
  "iconqa:./shell/dmole/iconqa.json:2"
  "ocrvqa:./shell/dmole/ocrvqa.json:2"
  "flickr30k:./shell/dmole/flickr30k.json:2"
  "vizwiz:./shell/dmole/vizwiz.json:2"
  "kvqa:./shell/dmole/kvqa.json:2"
  "pmcvqa:./shell/dmole/pmcvqa.json:2"
)

# Function to parse task configuration
parse_task_config() {
  local config=$1
  IFS=":" read -r TASK_NAME META_PATH BATCH_SIZE <<< "$config"
}

# Main loop to compute zero-cost proxy scores
for ((i=0; i<${#TASK_CONFIGS[@]}; i++)); do
  config="${TASK_CONFIGS[$i]}"
  
  # Skip commented tasks
  if [[ $config == \#* ]]; then
    continue
  fi
  
  # Parse current task configuration
  parse_task_config "$config"
  
  echo "=========================================="
  echo "Computing zero-cost proxy score for Task $((i+1)): $TASK_NAME"
  echo "Meta Path: $META_PATH"
  echo "Batch Size: $BATCH_SIZE"
  echo "Output File: ${OUTPUT_DIR}/$((i+1))_${BASE_MODEL}_${TASK_NAME}_score.csv"
  echo "=========================================="

  # Compute zero-cost proxy scores
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/scorer/compute_zc_score.py \
    --model_name_or_path ${PRETRAINED_MODEL_PATH} \
    --output_dir none \
    --conv_style "internlm2-chat" \
    --meta_path ${META_PATH} \
    --force_image_size 448 \
    --max_dynamic_patch 6 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --vision_select_layer -1 \
    --bf16 True \
    --num_train_epochs 1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --zc_proxy_score_portion 0.01 \
    --zc_proxy_score_save_path "${OUTPUT_DIR}/$((i+1))_${BASE_MODEL}_${TASK_NAME}_score.csv" \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

  echo "Zero-cost proxy score computation for Task $((i+1)): $TASK_NAME completed!"
  echo ""
done

echo "All zero-cost proxy score computations completed successfully!"
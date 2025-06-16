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
INITIAL_MODEL_PATH="pretrained/${BASE_MODEL}"
CONTINUAL_BASE_DIR='checkpoints/dmole'
AUTOENCODER_PATH="autoencoder_models"
ENABLE_EVALUATION=${ENABLE_EVALUATION:-true}

# Create output directory
mkdir -p "$CONTINUAL_BASE_DIR"

# Task configurations: name:metapath:batch_size:epochs
TASK_CONFIGS=(
  "vizwiz_caption:./shell/dmole/vizwiz_caption.json:10:1"
  "skvg:./shell/dmole/skvg.json:10:5"
  "textcaps:./shell/dmole/textcaps.json:10:1"
  "iconqa:./shell/dmole/iconqa.json:10:1"
  "ocrvqa:./shell/dmole/ocrvqa.json:8:1"
  "flickr30k:./shell/dmole/flickr30k.json:10:1"
  "vizwiz:./shell/dmole/vizwiz.json:10:5"
  "kvqa:./shell/dmole/kvqa.json:10:1"
  "pmcvqa:./shell/dmole/pmcvqa.json:10:1"
)

# Evaluation datasets
EVALUATION_DATASETS=(
  "caption-vizwiz-val"
  "grouding-skvg-test"
  "caption-textcaps-val"
  "vqa-iconqa-test"
  "vqa-ocrvqa-val"
  "caption-flickr30k"
  "vqa-vizwiz-val"
  "vqa-kvqa-test"
  "vqa-pmcvqa-test-clean"
)

# Function to parse task configuration
parse_task_config() {
  local config=$1
  IFS=":" read -r TASK_NAME META_PATH BATCH_SIZE EPOCHS <<< "$config"
}

# Main training loop
PREVIOUS_MODEL=$INITIAL_MODEL_PATH

for i in "${!TASK_CONFIGS[@]}"; do
  # Parse current task configuration
  parse_task_config "${TASK_CONFIGS[$i]}"
  
  # Set output directory and DMOLE arch path
  OUTPUT_DIR="${CONTINUAL_BASE_DIR}/$((i+1))_${BASE_MODEL}-${TASK_NAME}"
  DMOLE_ARCH_PATH="dmole_arch/$((i+1))_${BASE_MODEL}_${TASK_NAME}_arch.json"
  mkdir -p "$OUTPUT_DIR"
  
  echo "=========================================="
  echo "Training Task $((i+1))/${#TASK_CONFIGS[@]}: $TASK_NAME"
  echo "Meta Path: $META_PATH"
  echo "Batch Size: $BATCH_SIZE"
  echo "Epochs: $EPOCHS"
  echo "Previous Model: $PREVIOUS_MODEL"
  echo "Output Dir: $OUTPUT_DIR"
  echo "DMOLE Arch Path: $DMOLE_ARCH_PATH"
  echo "=========================================="

  # Training
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/train/internvl_chat_finetune.py \
    --model_name_or_path ${PREVIOUS_MODEL} \
    --conv_style "internlm2-chat" \
    --output_dir ${OUTPUT_DIR} \
    --meta_path ${META_PATH} \
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
    --dmole_arch_path ${DMOLE_ARCH_PATH} \
    --autoencoder_path autoencoder_models \
    --task_id $((i+1)) \
    --vision_select_layer -1 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
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
    --report_to "wandb" \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"

  # Evaluation
  if [ "$ENABLE_EVALUATION" = "true" ]; then
    for eval_dataset in "${EVALUATION_DATASETS[@]}"; do
      echo "Evaluating Task $((i+1)): ${eval_dataset}"
      GPUS=${GPUS} bash evaluate.sh ${OUTPUT_DIR} ${eval_dataset} --dynamic
    done
  else
    echo "Evaluation skipped (ENABLE_EVALUATION=$ENABLE_EVALUATION)"
  fi

  # Update input model for next training
  PREVIOUS_MODEL="$OUTPUT_DIR"
  
  echo "Task $((i+1)) completed successfully!"
  echo ""
done

echo "All tasks completed successfully!"
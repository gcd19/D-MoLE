#!/bin/bash

set -e

# Parse arguments
MODE="continual"           # pretrained or continual
SELECTED_MODELS=()         # pretrained or continual model
SELECTED_TASKS=()          # task indices or "all"

while [[ $# -gt 0 ]]; do
  case $1 in
    pretrained|continual) MODE="$1" ;;                                          # Evaluation mode
    --models) shift; IFS=',' read -ra SELECTED_MODELS <<< "$1" ;;            # Model selection
    --tasks) shift; IFS=',' read -ra SELECTED_TASKS <<< "$1" ;;              # Task selection
    *) echo "Usage: $0 [pretrained|continual] [--models MODEL_LIST] [--tasks TASK_LIST]"; exit 1 ;;
  esac
  shift
done

# Configuration
export CUDA_VISIBLE_DEVICES=0                               # GPU devices
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') # Number of GPUs
BASE_MODEL="InternVL2-2B"                                                     # Base model name
CONTINUAL_BASE_PATH="checkpoints/dmole"                                       # Base path for continual models
LOG_FILE="results/evaluation/$(date +%Y%m%d_%H%M%S).log"                     # Log file path
mkdir -p "$(dirname "$LOG_FILE")"

# Task configuration: "task_name:evaluation_dataset:batch_size"
TASKS=(
  "vizwiz_caption:caption-vizwiz-val:16"
  "skvg:grouding-skvg-test:8"
  "textcaps:caption-textcaps-val:8"
  "iconqa:vqa-iconqa-test:8"
  "ocrvqa:vqa-ocrvqa-val:8"
  "flickr30k:caption-flickr30k:8"
  "vizwiz:vqa-vizwiz-val:8"
  "kvqa:vqa-kvqa-test:16"
  "pmcvqa:vqa-pmcvqa-test-clean:8" 
)

# Set defaults
case $MODE in
  pretrained)                                                                 # Evaluate pretrained model on all tasks
    [[ ${#SELECTED_MODELS[@]} -eq 0 ]] && SELECTED_MODELS=(pretrained)
    [[ ${#SELECTED_TASKS[@]} -eq 0 ]] && SELECTED_TASKS=(all)
    ;;
  continual)                                                                  # Evaluate all continual models on all tasks
    [[ ${#SELECTED_MODELS[@]} -eq 0 ]] && SELECTED_MODELS=(0 1 2 3 4 5 6 7 8)
    [[ ${#SELECTED_TASKS[@]} -eq 0 ]] && SELECTED_TASKS=(all)
    ;;
  *)                                                                          # Default: skvg model on skvg task
    [[ ${#SELECTED_MODELS[@]} -eq 0 ]] && SELECTED_MODELS=(1)
    [[ ${#SELECTED_TASKS[@]} -eq 0 ]] && SELECTED_TASKS=(1)
    ;;
esac

# Get model paths
get_models() {
  for model in "${SELECTED_MODELS[@]}"; do
    if [[ "$model" == "pretrained" ]]; then
      echo "pretrained/$BASE_MODEL"                                          # Base pretrained model path
    elif [[ "$model" =~ ^[0-8]$ ]]; then
      local task=$(echo "${TASKS[$model]}" | cut -d: -f1)
      echo "$CONTINUAL_BASE_PATH/$((model+1))_${BASE_MODEL}-${task}"   # Continual learning model path
    fi
  done
}

# Get task indices
get_tasks() {
  for task in "${SELECTED_TASKS[@]}"; do
    if [[ "$task" == "all" ]]; then
      seq 0 $((${#TASKS[@]}-1))                                              # All task indices (0-8)
      return
    elif [[ "$task" =~ ^[0-8]$ ]]; then
      echo "$task"                                                           # Specific task index
    fi
  done
}

# Run evaluation
readarray -t models < <(get_models)
readarray -t task_indices < <(get_tasks)

for model in "${models[@]}"; do
  [[ ! -d "$model" ]] && continue                                            # Skip if model directory doesn't exist
  for task_idx in "${task_indices[@]}"; do
    IFS=: read -r task_name dataset batch <<< "${TASKS[$task_idx]}"           # Parse task config: name:dataset:batch_size
    echo "Evaluating $(basename "$model") on $dataset"
    GPUS=$GPUS bash evaluate.sh "$model" "$dataset" --dynamic --batch-size "$batch" 2>&1 | tee -a "$LOG_FILE"
  done
done
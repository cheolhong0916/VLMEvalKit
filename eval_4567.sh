#!/bin/bash

# =================================================================
# VLMEvalKit Sequential Evaluation Script
# -----------------------------------------------------------------
# This script evaluates multiple models one by one. Each evaluation
# job is expected to utilize all available GPUs, making sequential
# execution the most efficient approach.
#
# To see the exact commands being run, uncomment the line below.
# set -x
# =================================================================

# --- Configuration ---

# The dataset to use for evaluation.
DATASET="RoboSpatial"

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.
# MODELS=(
#     # "Qwen2.5-VL-7B-Instruct-2d-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-3d-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-dynamic-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-perception-1epoch"
#     "Qwen2.5-VL-7B-Instruct-real-1epoch"
#     "Qwen2.5-VL-7B-Instruct-reasoning-1epoch"
#     "Qwen2.5-VL-7B-Instruct-static-1epoch"
#     "Qwen2.5-VL-7B-Instruct-synthetic-1epoch"
# )

MODELS=(
    "Qwen2.5-VL-32B-Instruct-real"
    "Qwen2.5-VL-32B-Instruct-reasoning"
    "Qwen2.5-VL-32B-Instruct-static"
    "Qwen2.5-VL-32B-Instruct-synthetic"
)

# MODELS=(
#     # "llava_next_vicuna_7b_lora_rank_64_2d"
#     # "llava_next_vicuna_7b_lora_rank_64_3d"
#     # "llava_next_vicuna_7b_lora_rank_64_dynamic"
#     # "llava_next_vicuna_7b_lora_rank_64_perception"
#     "llava_next_vicuna_7b_lora_rank_64_real"
#     "llava_next_vicuna_7b_lora_rank_64_reasoning"
#     "llava_next_vicuna_7b_lora_rank_64_static"
#     "llava_next_vicuna_7b_lora_rank_64_synthetic"
# )

# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"
    
    # Run the evaluation command for the current model.
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python run.py --data "${DATASET}" --model "${model_name}"
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --master_port=30000 run.py --data "${DATASET}" --model "${model_name}"

    # Check the exit code of the last command.
    # If it's non-zero, an error occurred.
    if [ $? -ne 0 ]; then
        echo "--> ERROR: Evaluation failed for model ${model_name}. Halting script."
        exit 1
    fi

    echo "--> Finished evaluation for model: ${model_name}"
    echo "----------------------------------------------------------------"
done

DURATION=$((SECONDS - START_TIME))

echo ""
echo "All evaluations completed successfully!"
echo "Total execution time: $((DURATION / 3600))h $(((DURATION / 60) % 60))m $((DURATION % 60))s"
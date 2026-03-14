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
# DATASET="ERQA"
# DATASET="EmbSpatialBench"
DATASET="BLINK"
# DATASET="CV-Bench-2D"
# DATASET="CV-Bench-3D"
# DATASET="Spatial457"
# DATASET="RoboSpatial"

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.

# MODELS=(
#     "molmo-7B-O-0924-spatial_relation_in_80k"
#     "molmo-7B-O-0924-non_spatial_relation_in_80k"
# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-2d"
#     "Qwen2.5-VL-3B-Instruct-3d"
#     "Qwen2.5-VL-3B-Instruct-dynamic"
#     "Qwen2.5-VL-3B-Instruct-perception"
#     "Qwen2.5-VL-3B-Instruct-real"
#     "Qwen2.5-VL-3B-Instruct-reasoning"
#     "Qwen2.5-VL-3B-Instruct-static"
#     "Qwen2.5-VL-3B-Instruct-synthetic"
# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-data_scale_exp_2m"
# )

# MODELS=(
#     "SpatialLadder-3B"
# )

# MODELS=(
#     "Qwen2.5-VL-7B-Instruct-2d"
#     "Qwen2.5-VL-7B-Instruct-3d"
#     "Qwen2.5-VL-7B-Instruct-dynamic"
#     "Qwen2.5-VL-7B-Instruct-perception"
#     # "Qwen2.5-VL-7B-Instruct-real"
#     # "Qwen2.5-VL-7B-Instruct-reasoning"
#     # "Qwen2.5-VL-7B-Instruct-static"
#     # "Qwen2.5-VL-7B-Instruct-synthetic"
# )

# MODELS=(
#     "Qwen2.5-VL-7B-Instruct-2d-1epoch"
#     "Qwen2.5-VL-7B-Instruct-3d-1epoch"
#     "Qwen2.5-VL-7B-Instruct-dynamic-1epoch"
#     "Qwen2.5-VL-7B-Instruct-perception-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-real-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-reasoning-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-static-1epoch"
#     # "Qwen2.5-VL-7B-Instruct-synthetic-1epoch"
# )

# 7B models without RoboSpatial (20250927_081915)
# MODELS=(
#     "Qwen2.5-VL-7B-Instruct-real_woRS"
#     "Qwen2.5-VL-7B-Instruct-static_woRS"
#     "Qwen2.5-VL-7B-Instruct-reasoning_woRS"
#     "Qwen2.5-VL-7B-Instruct-3d_woRS"
# )

# MODELS=(
#     "Qwen2.5-VL-32B-Instruct-2d"
#     "Qwen2.5-VL-32B-Instruct-3d"
#     "Qwen2.5-VL-32B-Instruct-dynamic"
#     "Qwen2.5-VL-32B-Instruct-perception"
# )

# MODELS=(
#     "llava_next_vicuna_7b_lora_rank_64_2d"
#     "llava_next_vicuna_7b_lora_rank_64_3d"
#     "llava_next_vicuna_7b_lora_rank_64_dynamic"
#     "llava_next_vicuna_7b_lora_rank_64_perception"
#     # "llava_next_vicuna_7b_lora_rank_64_real"
#     # "llava_next_vicuna_7b_lora_rank_64_reasoning"
#     # "llava_next_vicuna_7b_lora_rank_64_static"
#     # "llava_next_vicuna_7b_lora_rank_64_synthetic"
# )

MODELS=(
    "NVILA-Lite-2B"
    "NVILA-Lite-2B-data-scale-exp-80k"
    "NVILA-Lite-2B-data-scale-exp-400k"
    "NVILA-Lite-2B-data-scale-exp-800k"
    # "NVILA-Lite-2B-data-scale-exp-2m"
    # "NVILA-Lite-2B-single_prism_80k"
    # "NVILA-Lite-2B-single_refspatial_80k"
    # "NVILA-Lite-2B-single_robospatial_80k"
    # "NVILA-Lite-2B-single_sat_80k"
    # "NVILA-Lite-2B-single_spar7m_80k"
    # "NVILA-Lite-2B-single_spatial457_23k"
)

# MODELS=(
#     "RoboRefer-2B-SFT"
# )

# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"
    MASTER_PORT=$((RANDOM % 40001 + 20000))
    # Run the evaluation command for the current model.
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port=${MASTER_PORT} run.py --data "${DATASET}" --model "${model_name}"

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
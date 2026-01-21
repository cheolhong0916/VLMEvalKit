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
# DATASET="BLINK"
# DATASET="CV-Bench-2D"
# DATASET="CV-Bench-3D"
# DATASET="RoboSpatial"
DATASET="Spatial457"

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.
# MODELS=(
# #     "Qwen2.5-VL-3B-Instruct-2d_erqa"
# #     "Qwen2.5-VL-3B-Instruct-3d_erqa"
# #     "Qwen2.5-VL-3B-Instruct-dynamic_erqa"
# #     "Qwen2.5-VL-3B-Instruct-perception_erqa"
#     "Qwen2.5-VL-3B-Instruct-real_erqa"
#     "Qwen2.5-VL-3B-Instruct-reasoning_erqa"
#     "Qwen2.5-VL-3B-Instruct-static_erqa"
#     "Qwen2.5-VL-3B-Instruct-synthetic_erqa"
# )

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-2d-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-3d-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-dynamic-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-perception-1epoch"
#     "Qwen2.5-VL-3B-Instruct-real-1epoch"
#     "Qwen2.5-VL-3B-Instruct-reasoning-1epoch"
#     "Qwen2.5-VL-3B-Instruct-static-1epoch"
#     "Qwen2.5-VL-3B-Instruct-synthetic-1epoch"
# )


# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-single_prism_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_refspatial_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_robospatial_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_sat_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_spar7m_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_spatial457_23k"
#     # "Qwen2.5-VL-3B-Instruct-data_scale_exp_80k"
#     # "Qwen2.5-VL-3B-Instruct-data_scale_exp_400k"
#     # "Qwen2.5-VL-3B-Instruct-data_scale_exp_800k"
#     "Qwen2.5-VL-3B-Instruct-data_scale_exp_2m"
#     # "Qwen2.5-VL-3B-Instruct"
# )

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-single_prism_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_refspatial_80k"
#     # "Qwen2.5-VL-3B-Instruct-single_robospatial_80k"
#     "Qwen2.5-VL-3B-Instruct-single_sat_80k"
#     "Qwen2.5-VL-3B-Instruct-single_spar7m_80k"
#     "Qwen2.5-VL-3B-Instruct-single_spatial457_23k"
# )

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-top3_action_reasoning"
#     # "Qwen2.5-VL-3B-Instruct-top3_multi-view-reasoning"
#     # "Qwen2.5-VL-3B-Instruct-top3_other"
#     # "Qwen2.5-VL-3B-Instruct-top3_pointing"
#     "Qwen2.5-VL-3B-Instruct-top3_spatial_reasoning"
#     "Qwen2.5-VL-3B-Instruct-top3_state_estimation"
#     "Qwen2.5-VL-3B-Instruct-top3_task_reasoning"
#     "Qwen2.5-VL-3B-Instruct-top3_trajectory_reasoning"
# )

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-top3_action_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_multi-view_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_other_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_pointing_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_spatial_reasoning_80k"
#     "Qwen2.5-VL-3B-Instruct-top3_state_estimation_80k"
#     "Qwen2.5-VL-3B-Instruct-top3_task_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_trajectory_reasoning_80k"
# )

# MODELS=(
#     # "Qwen2.5-VL-7B-Instruct-2d"
#     # "Qwen2.5-VL-7B-Instruct-3d"
#     # "Qwen2.5-VL-7B-Instruct-dynamic"
#     # "Qwen2.5-VL-7B-Instruct-perception"
#     "Qwen2.5-VL-7B-Instruct-real"
#     "Qwen2.5-VL-7B-Instruct-reasoning"
#     "Qwen2.5-VL-7B-Instruct-static"
#     "Qwen2.5-VL-7B-Instruct-synthetic"
# )

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

# MODELS=(
#     "Qwen2.5-VL-32B-Instruct-real"
#     "Qwen2.5-VL-32B-Instruct-reasoning"
#     "Qwen2.5-VL-32B-Instruct-static"
#     "Qwen2.5-VL-32B-Instruct-synthetic"
# )

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

# MODELS=(
#     "NVILA-Lite-2B-data-scale-exp-80k"
#     "NVILA-Lite-2B-data-scale-exp-400k"
#     "NVILA-Lite-2B-data-scale-exp-800k"
#     "NVILA-Lite-2B-single_prism_80k"
#     "NVILA-Lite-2B-single_refspatial_80k"
#     "NVILA-Lite-2B-single_robospatial_80k"
#     "NVILA-Lite-2B-single_sat_80k"
#     "NVILA-Lite-2B-single_spar7m_80k"
#     "NVILA-Lite-2B-single_spatial457_23k"
# )
 
# MODELS=(
#     "prism-dinosiglip+7b-data-scale-exp-80k"
#     "prism-dinosiglip+7b-data-scale-exp-400k"
#     "prism-dinosiglip+7b-data-scale-exp-800k"
# )

MODELS=(
    "molmo-7B-O-0924"
    "molmo-7B-O-0924-single_prism"
    "molmo-7B-O-0924-single_refspatial"
    "molmo-7B-O-0924-single_robospatial"
    "molmo-7B-O-0924-single_sat"
    "molmo-7B-O-0924-single_spar"
    "molmo-7B-O-0924-single_spatial457"
    "molmo-7B-O-0924-data_scale_exp_80k"
    "molmo-7B-O-0924-data_scale_exp_400k"
    "molmo-7B-O-0924-data_scale_exp_800k"
    "molmo-7B-O-0924-data_scale_exp_2m"
)

# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"
    MASTER_PORT=$((RANDOM % 40001 + 20000))
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc-per-node=6 --master_port=${MASTER_PORT} run.py --data "${DATASET}" --model "${model_name}"

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
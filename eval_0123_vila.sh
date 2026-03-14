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
DATASET="CV-Bench-3D"
# DATASET="RoboSpatial"
# DATASET="Spatial457"

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.

# MODELS=(
#     # "molmo-7B-O-0924-spatial_relation_in_80k"
#     # "molmo-7B-O-0924-non_spatial_relation_in_80k"
#     # "molmo-7B-O-0924-spatial_relation_in_400k"
#     # "molmo-7B-O-0924-non_spatial_relation_in_400k"
#     "molmo-7B-O-0924-spatial_relation_in_80k_2"
#     "molmo-7B-O-0924-non_spatial_relation_in_80k_2"    
# )

# MODELS=(
#     "molmo-7B-O-0924-stage1_hard_in_80k_fixed"
#     "molmo-7B-O-0924-stage2_easy_in_80k_fixed"
#     "molmo-7B-O-0924-stage2_hard_cot_in_80k_fixed"
# )

# MODELS=(
#     "molmo-7B-O-0924-stage1_easy_in_80k"
#     "molmo-7B-O-0924-stage1_easy_in_80k_fixed"
#     "molmo-7B-O-0924-stage1_easy_in_400k"
#     "molmo-7B-O-0924-stage1_hard_in_80k_fixed"
#     "molmo-7B-O-0924-stage1_hard_in_134.4k"
#     "molmo-7B-O-0924-stage1_hard_in_400k"

#     "molmo-7B-O-0924-stage2_easy_in_80k"
#     # "molmo-7B-O-0924-stage2_easy_in_80k_fixed"
#     # "molmo-7B-O-0924-stage2_easy_in_400k"
#     # "molmo-7B-O-0924-stage2_hard_in_80k"
#     # "molmo-7B-O-0924-stage2_hard_in_80k_fixed"
#     # "molmo-7B-O-0924-stage2_hard_in_400k"
    
#     # "molmo-7B-O-0924-stage2_hard_cot_in_80k_fixed"
#     # "molmo-7B-O-0924-stage2_hard_cot_in_400k"

# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-2d_erqa"
#     "Qwen2.5-VL-3B-Instruct-3d_erqa"
#     "Qwen2.5-VL-3B-Instruct-dynamic_erqa"
#     "Qwen2.5-VL-3B-Instruct-perception_erqa"
#     # "Qwen2.5-VL-3B-Instruct-real_erqa"
#     # "Qwen2.5-VL-3B-Instruct-reasoning_erqa"
#     # "Qwen2.5-VL-3B-Instruct-static_erqa"
#     # "Qwen2.5-VL-3B-Instruct-synthetic_erqa"
# )


# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-2d-1epoch"
#     "Qwen2.5-VL-3B-Instruct-3d-1epoch"
#     "Qwen2.5-VL-3B-Instruct-dynamic-1epoch"
#     "Qwen2.5-VL-3B-Instruct-perception-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-real-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-reasoning-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-static-1epoch"
#     # "Qwen2.5-VL-3B-Instruct-synthetic-1epoch"
# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-single_prism"
#     "Qwen2.5-VL-3B-Instruct-single_refspatial"
#     "Qwen2.5-VL-3B-Instruct-single_robospatial"
#     # "Qwen2.5-VL-3B-Instruct-single_sat"
#     # "Qwen2.5-VL-3B-Instruct-single_spar7m"
#     # "Qwen2.5-VL-3B-Instruct-single_spatial457"
# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct"
#     "Qwen2.5-VL-3B-Instruct-single_prism_80k"
#     "Qwen2.5-VL-3B-Instruct-single_refspatial_80k"
#     "Qwen2.5-VL-3B-Instruct-single_robospatial_80k"
#     "Qwen2.5-VL-3B-Instruct-single_sat_80k"
#     "Qwen2.5-VL-3B-Instruct-single_spar7m_80k"
#     "Qwen2.5-VL-3B-Instruct-single_spatial457_23k"
# )

# MODELS=(
#     "Qwen2.5-VL-3B-Instruct-top3_action_reasoning"
#     "Qwen2.5-VL-3B-Instruct-top3_multi-view_reasoning"
#     "Qwen2.5-VL-3B-Instruct-top3_other"
#     "Qwen2.5-VL-3B-Instruct-top3_pointing"
#     # "Qwen2.5-VL-3B-Instruct-top3_spatial_reasoning"
#     # "Qwen2.5-VL-3B-Instruct-top3_state_estimation"
#     # "Qwen2.5-VL-3B-Instruct-top3_task_reasoning"
#     # "Qwen2.5-VL-3B-Instruct-top3_trajectory_reasoning"
# )

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-top3_action_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_multi-view_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_other_80k"
#     "Qwen2.5-VL-3B-Instruct-top3_pointing_80k"
#     "Qwen2.5-VL-3B-Instruct-top3_spatial_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_state_estimation_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_task_reasoning_80k"
#     # "Qwen2.5-VL-3B-Instruct-top3_trajectory_reasoning_80k"
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
    # "NVILA-Lite-2B"
    "NVILA-Lite-2B-data-scale-exp-80k-refspatial"
    "NVILA-Lite-2B-data-scale-exp-400k-refspatial"
    "NVILA-Lite-2B-data-scale-exp-800k-refspatial"
    "NVILA-Lite-2B-data-scale-exp-2m-refspatial"
    # "NVILA-Lite-2B-single_prism_80k"
    # "NVILA-Lite-2B-single_refspatial_80k"
    # "NVILA-Lite-2B-single_robospatial_80k"
    # "NVILA-Lite-2B-single_sat_80k"
    # "NVILA-Lite-2B-single_spar7m_80k"
    # "NVILA-Lite-2B-single_spatial457_23k"
)

# MODELS=(
#     # "Qwen2.5-VL-3B-Instruct-data_scale_exp_80k"
#     # "Qwen2.5-VL-3B-Instruct-data_scale_exp_400k"
#     "Qwen2.5-VL-3B-Instruct-data_scale_exp_800k"
# )


# MODELS=(
#     RoboRefer-2B-SFT
#     RoboRefer-8B-SFT
# )
# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"
    
    # Run the evaluation command for the current model.
    MASTER_PORT=$((RANDOM % 10001 + 20000))
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master_port=${MASTER_PORT} run.py --data "${DATASET}" --model "${model_name}"

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
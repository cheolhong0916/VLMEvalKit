#!/bin/bash

# =================================================================
# VLMEvalKit Sequential Evaluation Script (Multi-Dataset Version)
# -----------------------------------------------------------------
# This script evaluates multiple models on multiple datasets.
# It iterates through all combinations of datasets and models.
#
# To see the exact commands being run, uncomment the line below.
# set -x
# =================================================================

# --- Configuration ---

# An array of datasets to evaluate.
DATASETS=(
    # "EmbSpatialBench"
    # "BLINK"
    "BLINK_Spatial"
    # "CV-Bench-2D"
    # "CV-Bench-3D"
    # "Spatial457"
    # "ERQA"
    # "Spatial457"
)

# An array of model names to evaluate.
# MODELS=(
#     # "molmo-7B-O-0924-stage1_hard_in_400k"
#     # "molmo-7B-O-0924-stage2_easy_in_400k"
#     # "molmo-7B-O-0924-stage2_hard_cot_in_400k"
#     # "molmo-7B-O-0924-stage1_hard_in_134.4k"
#     # "molmo-7B-O-0924-single_sat_bias_fixed"
#     # "molmo-7B-O-0924-stage2_hard_in_400k"
#     "molmo-7B-O-0924-single_sat_bias_fixed_2"
# )
# # MODELS=(
# #     # "Qwen2.5-VL-3B-Instruct-single_sat_80k_bias_fixed"
# # )
# MODELS=(
#     # "NVILA-Lite-2B"
#     # "NVILA-Lite-2B-data-scale-exp-80k"
#     # "NVILA-Lite-2B-data-scale-exp-400k"
#     # "NVILA-Lite-2B-data-scale-exp-800k"
#     "NVILA-Lite-2B-data-scale-exp-2m"
#     # "NVILA-Lite-2B-single_prism_80k"
#     # "NVILA-Lite-2B-single_refspatial_80k"
#     # "NVILA-Lite-2B-single_robospatial_80k"
#     # "NVILA-Lite-2B-single_sat_80k"
#     # "NVILA-Lite-2B-single_spar7m_80k"
#     # "NVILA-Lite-2B-single_spatial457_23k"
# )
# MODELS=(
#     RoboRefer-2B-SFT
#     RoboRefer-8B-SFT
# )
# MODELS=(
#     # "Qwen3-VL-32B-Instruct"
#     # "Qwen3-VL-8B-Instruct"
#     # "Qwen3-VL-4B-Instruct"
#     # "Qwen3-VL-2B-Instruct"
#     "molmo2-8B"
# )
# MODELS=(
#     # "NVILA-Lite-2B-synthetic-mix-5pct-80k"
#     # "NVILA-Lite-2B-synthetic-mix-5pct-400k"
#     # "NVILA-Lite-2B-synthetic-mix-5pct-730k"
#     # "NVILA-Lite-2B-synthetic-mix-10pct-80k"
#     # "NVILA-Lite-2B-synthetic-mix-20pct-80k"
#     # "NVILA-Lite-2B-synthetic-mix-30pct-80k"
#     # "NVILA-Lite-2B-synthetic-mix-5pct-800k"
#     "NVILA-Lite-2B-ST-80k-5pct"
#     "NVILA-Lite-2B-ST-400k-5pct"
#     "NVILA-Lite-2B-ST-800k-5pct"
# )
# MODELS=(
#     "RoboRefer-2B-SFT-depth"
# )
# MODELS=(
#     Qwen3-VL-235B-A22B-Instruct
# )
# MODELS=(
#         "NVILA-Lite-2B-data-scale-exp-4m-refspatial"
# )
MODELS=(
    "molmo-72B-0924"
)
# --- Execution ---
DATASETS_STR="${DATASETS[*]}"
TOTAL_COMBINATIONS=$((${#DATASETS[@]} * ${#MODELS[@]}))


TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

echo "Starting optimized sequential evaluation"
echo "  Datasets to evaluate at once: ${DATASETS_STR}"
echo "  Models: ${TOTAL_MODELS} (${MODELS[*]})"
echo "================================================================"

START_TIME=$SECONDS
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKIP_ERR=1


for model_name in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================"
    echo "--> [${CURRENT}/${TOTAL_MODELS}] Evaluating Model: ${model_name}"
    echo "    On Datasets: ${DATASETS_STR}"
    echo "========================================"
    
    MASTER_PORT=$((RANDOM % 40001 + 20000))
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port=${MASTER_PORT} run.py --data ${DATASETS_STR} --model "${model_name}"
    
    # Qwen3 235B
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --data ${DATASETS_STR} --model "${model_name}"

    if [ $? -ne 0 ]; then
        echo "--> ERROR: Evaluation failed for model ${model_name}. Halting script."
        exit 1
    fi

    echo "--> Finished: ${model_name} on all datasets"
    echo "----------------------------------------------------------------"
done

DURATION=$((SECONDS - START_TIME))

echo ""
echo "================================================================"
echo "All evaluations completed successfully!"
echo "  Total models evaluated: ${TOTAL_MODELS}"
echo "  Total execution time: $((DURATION / 3600))h $(((DURATION / 60) % 60))m $((DURATION % 60))s"
echo "================================================================"

















# echo "Starting sequential evaluation"
# echo "  Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
# echo "  Models: ${#MODELS[@]} (${MODELS[*]})"
# echo "  Total combinations: ${TOTAL_COMBINATIONS}"
# echo "================================================================"

# START_TIME=$SECONDS
# export NCCL_TIMEOUT=3600
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# for dataset in "${DATASETS[@]}"; do
#     echo ""
#     echo "========================================"
#     echo "Dataset: ${dataset}"
#     echo "========================================"
    
#     for model_name in "${MODELS[@]}"; do
#         CURRENT=$((CURRENT + 1))
#         echo ""
#         echo "--> [${CURRENT}/${TOTAL_COMBINATIONS}] Evaluating: ${model_name} on ${dataset}"
        
#         # Run the evaluation command for the current model and dataset.
#         MASTER_PORT=$((RANDOM % 40001 + 20000))
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port=${MASTER_PORT} run.py --data "${dataset}" --model "${model_name}"
        
#         # For Qwen3 235B model
#         # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --data "${dataset}" --model "${model_name}"

#         # Check the exit code of the last command.
#         if [ $? -ne 0 ]; then
#             echo "--> ERROR: Evaluation failed for model ${model_name} on dataset ${dataset}. Halting script."
#             exit 1
#         fi

#         echo "--> Finished: ${model_name} on ${dataset}"
#         echo "----------------------------------------------------------------"
#     done
# done

# DURATION=$((SECONDS - START_TIME))

# echo ""
# echo "================================================================"
# echo "All evaluations completed successfully!"
# echo "  Total combinations evaluated: ${TOTAL_COMBINATIONS}"
# echo "  Total execution time: $((DURATION / 3600))h $(((DURATION / 60) % 60))m $((DURATION % 60))s"
# echo "================================================================"

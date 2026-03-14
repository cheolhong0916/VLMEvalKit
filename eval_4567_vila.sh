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
DATASET="EmbSpatialBench"
# DATASET="BLINK"
# DATASET="CV-Bench-2D"
# DATASET="CV-Bench-3D"
# DATASET="RoboSpatial"
# DATASET="Spatial457"

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.


# MODELS=(
#     "NVILA-Lite-2B-data-scale-exp-80k"
#     "NVILA-Lite-2B-data-scale-exp-400k"
#     "NVILA-Lite-2B-data-scale-exp-800k"
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
MODELS=(
    "NVILA-Lite-2B-synthetic-mix-5pct-80k"
    "NVILA-Lite-2B-synthetic-mix-5pct-400k"
    "NVILA-Lite-2B-synthetic-mix-5pct-730k"
    # "NVILA-Lite-2B-synthetic-mix-10pct-80k"
    # "NVILA-Lite-2B-synthetic-mix-20pct-80k"
    # "NVILA-Lite-2B-synthetic-mix-30pct-80k"
)
# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

# Set NCCL timeout to 3 hours (10800 seconds)
export NCCL_TIMEOUT=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"

    # Run the evaluation command for the current model.
    MASTER_PORT=$((RANDOM % 40001 + 20000))
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --master_port=${MASTER_PORT} --rdzv-conf timeout=1800 run.py --data "${DATASET}" --model "${model_name}"
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc-per-node=6 --master_port=42100 run.py --data "${DATASET}" --model "${model_name}"

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
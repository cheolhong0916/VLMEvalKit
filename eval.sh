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

# An array of model names to evaluate.
# These must match the keys in your VLMEvalKit config file.
MODELS=(
    "Qwen2.5-VL-7B-2d"
    "Qwen2.5-VL-7B-3d"
    "Qwen2.5-VL-7B-dynamic"
    "Qwen2.5-VL-7B-perception"
    "Qwen2.5-VL-7B-real"
    "Qwen2.5-VL-7B-reasoning"
    "Qwen2.5-VL-7B-static"
    "Qwen2.5-VL-7B-synthetic"
)

# The dataset to use for evaluation.
DATASET="RoboSpatial"

# --- Execution ---

echo "Starting sequential evaluation for ${#MODELS[@]} models on dataset: ${DATASET}"
echo "================================================================"

START_TIME=$SECONDS

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "--> Starting evaluation for model: ${model_name}"
    
    # Run the evaluation command for the current model.
    python run.py --data "${DATASET}" --model "${model_name}"

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
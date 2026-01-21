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
    "ERQA"
    "EmbSpatialBench"
    "BLINK"
    "CV-Bench-2D"
    "CV-Bench-3D"
)

# An array of model names to evaluate.
MODELS=(
    "molmo-7B-O-0924-data_scale_exp_80k_warmup_60"
)

# --- Execution ---

TOTAL_COMBINATIONS=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT=0

echo "Starting sequential evaluation"
echo "  Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
echo "  Models: ${#MODELS[@]} (${MODELS[*]})"
echo "  Total combinations: ${TOTAL_COMBINATIONS}"
echo "================================================================"

START_TIME=$SECONDS

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Dataset: ${dataset}"
    echo "========================================"
    
    for model_name in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--> [${CURRENT}/${TOTAL_COMBINATIONS}] Evaluating: ${model_name} on ${dataset}"
        
        # Run the evaluation command for the current model and dataset.
        MASTER_PORT=$((RANDOM % 40001 + 20000))
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --master_port=${MASTER_PORT} run.py --data "${dataset}" --model "${model_name}"

        # Check the exit code of the last command.
        if [ $? -ne 0 ]; then
            echo "--> ERROR: Evaluation failed for model ${model_name} on dataset ${dataset}. Halting script."
            exit 1
        fi

        echo "--> Finished: ${model_name} on ${dataset}"
        echo "----------------------------------------------------------------"
    done
done

DURATION=$((SECONDS - START_TIME))

echo ""
echo "================================================================"
echo "All evaluations completed successfully!"
echo "  Total combinations evaluated: ${TOTAL_COMBINATIONS}"
echo "  Total execution time: $((DURATION / 3600))h $(((DURATION / 60) % 60))m $((DURATION % 60))s"
echo "================================================================"

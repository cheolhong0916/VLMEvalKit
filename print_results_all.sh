#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <data> <date> <base_model>"
    echo "Example: $0 ERQA 20250926 Qwen2.5-VL-32B-Instruct"
    exit 1
fi

# Get arguments
DATA=$1
DATE=$2
BASE_MODEL=$3

# Define fine-tuned model labels
MODELS=("synthetic" "real" "static" "dynamic" "perception" "reasoning" "2d" "3d")

echo "Running print_results.py for all fine-tuned models..."
echo "Data: $DATA"
echo "Date: $DATE"
echo "Base model: $BASE_MODEL"
echo "----------------------------------------"

# Loop through each fine-tuned model
for model_label in "${MODELS[@]}"; do
    full_model_name="${BASE_MODEL}-${model_label}"
    echo "Running with model: $full_model_name"
    
    python print_results.py --data "$DATA" --date "$DATE" --model "$full_model_name"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $full_model_name"
    else
        echo "✗ Failed: $full_model_name"
    fi
    echo "----------------------------------------"
done

echo "All models processed!"
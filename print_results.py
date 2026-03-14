import os
import json
import argparse
import glob
import pandas as pd
import ast
import re
import numpy as np

parser = argparse.ArgumentParser(
    description='Save score json w/o answers')
parser.add_argument('--data', type=str, help='Names of Dataset')
parser.add_argument('--model', type=str, help='Names of Model')
parser.add_argument('--date', type=str, help='Date of the experiment')
parser.add_argument('format', nargs='?', default=None, help='Format to read (xlsx or json)')
args = parser.parse_args()

# Base path: outputs/model_name
base_path = os.path.join('outputs', args.model)
folder_pattern = os.path.join(base_path, f"T{args.date}_G*")

found_folders = glob.glob(folder_pattern)

base_filename = None
search_dirs = []

if found_folders:
    # Case 1: Dated folder exists
    folder_name = os.path.basename(found_folders[0])
    try:
        commit_id = folder_name.split('_')[1]
    except IndexError:
        print(f"ERROR: Could not extract Commit ID from folder name: {folder_name}")
        exit()

    print(f"Found target folder: {folder_name}")
    print(f"Extracted Commit ID: {commit_id}")

    # Paths
    path_in_folder = os.path.join(base_path, folder_name, args.model + '_' + args.data)
    path_in_root = os.path.join(base_path, args.model + '_' + args.data)
    
    base_filename = path_in_folder

    # BLINK special logic: Check root if not in folder
    if 'BLINK' in args.data:
        if not os.path.exists(path_in_folder + '_acc.csv') and os.path.exists(path_in_root + '_acc.csv'):
            base_filename = path_in_root
            
    search_dirs = [os.path.join(base_path, folder_name), base_path]

else:
    # Case 2: No dated folder found -> Check root directory directly
    print(f"WARNING: No dated folder found for {args.date}. Checking root output folder...")
    
    direct_path = os.path.join(base_path, args.model + '_' + args.data)
    
    # Check if any result file exists in root
    # (acc.csv, score.json, or any xlsx result)
    if os.path.exists(direct_path + '_acc.csv') or \
       os.path.exists(direct_path + '_score.json') or \
       glob.glob(os.path.join(base_path, "*result.xlsx")):
        
        print(f"Found files in root: {base_path}")
        base_filename = direct_path
        search_dirs = [base_path]
        folder_name = "" 
    else:
        print(f"ERROR: No matching folder found for pattern: {folder_pattern}")
        print(f"       And no direct file found at: {direct_path}_acc.csv")
        exit()

# Define file paths based on determined base_filename
json_path = base_filename + '_score.json'
acc_csv_path = base_filename + '_acc.csv'
acc_by_relation_csv_path = base_filename + '_acc_by_relation.csv'
xlsx_path = base_filename + '.xlsx'

# --- EmbSpatialBench Special Fallback Logic ---
# If specific score files are not found in the dated folder, fallback to checking root directory
if 'EmbSpatialBench' in args.data:
    if not (os.path.exists(json_path) or os.path.exists(acc_csv_path)):
        print(f"WARNING: EmbSpatialBench files not found in dated folder. Checking root: {base_path}")
        root_base_filename = os.path.join(base_path, args.model + '_' + args.data)
        
        # Check if root files exist
        if os.path.exists(root_base_filename + '_score.json') or \
           os.path.exists(root_base_filename + '_acc.csv'):
            print(f"Found EmbSpatialBench files in root directory.")
            base_filename = root_base_filename
            json_path = base_filename + '_score.json'
            acc_csv_path = base_filename + '_acc.csv'
            acc_by_relation_csv_path = base_filename + '_acc_by_relation.csv'
            xlsx_path = base_filename + '.xlsx'
            # Add root to search dirs if not already there
            if base_path not in search_dirs:
                search_dirs.append(base_path)

# --- Flexible File Search Logic ---
cvbench_result_path = None
found_files = []

for search_dir in search_dirs:
    # Look for any *result.xlsx file
    pattern = os.path.join(search_dir, "*result.xlsx")
    files = glob.glob(pattern)
    found_files.extend(files)

if found_files:
    # 1. Try to find a file that specifically contains the dataset name (args.data)
    matching_files = [f for f in found_files if args.data in os.path.basename(f)]
    
    if matching_files:
        # Pick the latest one if multiple match
        cvbench_result_path = sorted(matching_files)[-1]
    else:
        # 2. Fallback to the latest available result file
        cvbench_result_path = sorted(found_files)[-1]
        print(f"WARNING: Exact match for '{args.data}' not found. Using found result file: {os.path.basename(cvbench_result_path)}")

data = {}

# Function to extract answer from prediction for xlsx
def extract_answer_xlsx(pred):
    """Extract answer from prediction string or dict for xlsx files"""
    if pd.isna(pred): return None
    if isinstance(pred, str):
        if pred.strip().startswith('{'):
            try:
                pred_dict = ast.literal_eval(pred)
                if isinstance(pred_dict, dict) and 'Answer' in pred_dict:
                    return pred_dict['Answer']
            except: pass
        pattern = r'^([A-D])\.\s'
        match = re.match(pattern, pred.strip())
        if match: return match.group(1)
        if len(pred.strip()) == 1: return pred.strip()
    if isinstance(pred, dict) and 'Answer' in pred: return pred['Answer']
    return None

def evaluate_erqa_xlsx(xlsx_path):
    """Evaluate ERQA benchmark from xlsx file"""
    print(f"Reading file from: {xlsx_path}")
    
    # Read xlsx file
    df = pd.read_excel(xlsx_path)
    
    # Find prediction column (could be 'prediction', 'pred', or similar)
    pred_col = None
    for col in df.columns:
        if 'prediction' in col.lower() or col.lower() == 'pred':
            pred_col = col
            break
    
    if pred_col is None:
        print("ERROR: Could not find prediction column in xlsx file")
        exit()
    
    # Find answer column (ground truth)
    answer_col = None
    for col in df.columns:
        if col.lower() in ['answer', 'gt', 'ground_truth', 'label']:
            answer_col = col
            break
    
    if answer_col is None:
        print("ERROR: Could not find answer column in xlsx file")
        exit()
    
    print(f"Using prediction column: {pred_col}")
    print(f"Using answer column: {answer_col}")
    
    # Extract answers from predictions using xlsx-specific extraction
    predictions = []
    for pred in df[pred_col]:
        answer = extract_answer_xlsx(pred)
        predictions.append(answer)
    
    df['extracted_answer'] = predictions
    
    # Calculate overall accuracy
    # Count all rows where answer is not None (valid ground truth)
    correct = 0
    total = 0
    for pred, gt in zip(df['extracted_answer'], df[answer_col]):
        if not pd.isna(gt):  # Only check if ground truth exists
            total += 1
            if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                correct += 1
    
    overall_acc = (correct / total) if total > 0 else 0
    
    # Initialize results with overall stats
    results = {
        'Correct': correct,
        'Total': total,
        'Accuracy': overall_acc
    }
    
    # Find image type column
    image_type_col = None
    for col in df.columns:
        if 'image' in col.lower() and 'type' in col.lower():
            image_type_col = col
            break
    
    # Find question type column
    question_type_col = None
    for col in df.columns:
        if 'question' in col.lower() and 'type' in col.lower():
            question_type_col = col
            break
    
    # Collect stats for both image types and question types
    image_stats = {}
    question_stats = {}
    
    # Calculate Single_Image and Multi_Image accuracy
    if image_type_col:
        for img_type in ['Single_Image', 'Multi_Image']:
            img_df = df[df[image_type_col] == img_type]
            img_correct = 0
            img_total = 0
            for pred, gt in zip(img_df['extracted_answer'], img_df[answer_col]):
                if not pd.isna(gt):  # Only check if ground truth exists
                    img_total += 1
                    if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                        img_correct += 1
            
            img_acc = (img_correct / img_total) if img_total > 0 else 0
            image_stats[img_type] = {
                'accuracy': img_acc,
                'correct': img_correct,
                'total': img_total
            }
    
    # Calculate accuracy by question type
    if question_type_col:
        question_types = df[question_type_col].unique()
        
        for q_type in question_types:
            if pd.isna(q_type):
                continue
            q_df = df[df[question_type_col] == q_type]
            q_correct = 0
            q_total = 0
            for pred, gt in zip(q_df['extracted_answer'], q_df[answer_col]):
                if not pd.isna(gt):  # Only check if ground truth exists
                    q_total += 1
                    if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                        q_correct += 1
            
            q_acc = (q_correct / q_total) if q_total > 0 else 0
            
            q_name = str(q_type).strip()
            question_stats[q_name] = {
                'accuracy': q_acc,
                'correct': q_correct,
                'total': q_total
            }
    
    # Define the order of question types as in JSON
    question_type_order = [
        'Trajectory Reasoning',
        'Action Reasoning',
        'Pointing',
        'State Estimation',
        'Spatial Reasoning',
        'Multi-view Reasoning',
        'Task Reasoning',
        'Other'
    ]
    
    # Add image type accuracies
    for img_type in ['Single_Image', 'Multi_Image']:
        if img_type in image_stats:
            results[f'{img_type}_Accuracy'] = image_stats[img_type]['accuracy']
    
    # Add question type accuracies in order
    for q_name in question_type_order:
        if q_name in question_stats:
            results[f'{q_name}_Accuracy'] = question_stats[q_name]['accuracy']
    
    # Add image type counts
    for img_type in ['Single_Image', 'Multi_Image']:
        if img_type in image_stats:
            results[f'{img_type}_Correct'] = image_stats[img_type]['correct']
            results[f'{img_type}_Total'] = image_stats[img_type]['total']
    
    # Add question type counts in order
    for q_name in question_type_order:
        if q_name in question_stats:
            results[f'{q_name}_Correct'] = question_stats[q_name]['correct']
            results[f'{q_name}_Total'] = question_stats[q_name]['total']
    
    return results

# Function to evaluate EmbSpatial from xlsx
def evaluate_embspatial_xlsx(xlsx_path):
    print(f"Reading file from: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    results = {'Overall': 0.0}
    return results

# Function to evaluate CV-Bench using xlsx results
def evaluate_cvbench(result_path):
    """
    Evaluate CV-Bench using the 'hit' column.
    Uses 'category' column to breakdown tasks as requested.
    """
    print(f"Reading CV-Bench result from: {result_path}")
    df = pd.read_excel(result_path)

    if 'hit' not in df.columns:
        print("ERROR: 'hit' column not found in the excel file.")
        exit()

    results = {}
    
    # 1. Overall Accuracy
    results['Overall'] = df['hit'].mean()
    
    # 2. Breakdown by 'category'
    if 'category' in df.columns:
        print("Breaking down accuracy by 'category'...")
        cat_stats = df.groupby('category')['hit'].mean()
        for cat, score in cat_stats.items():
            results[str(cat)] = score
    else:
        print("WARNING: 'category' column not found. Unable to provide task breakdown.")

    return results

# --- Main Logic ---

is_cvbench = 'CV-Bench' in args.data
target_file = xlsx_path

# Use the searched result file if it's CV-Bench and it exists
if is_cvbench and cvbench_result_path:
    target_file = cvbench_result_path

# 1. Process as XLSX
# Force if format is xlsx, OR if it's CV-Bench and we found a result file
if args.format == 'xlsx' or (is_cvbench and cvbench_result_path):
    
    if os.path.exists(target_file):
        if 'ERQA' in args.data.upper():
            data = evaluate_erqa_xlsx(target_file)
        elif 'EMBSPATIAL' in args.data.upper() or 'EMB_SPATIAL' in args.data.upper():
            data = evaluate_embspatial_xlsx(target_file)
        elif is_cvbench:
            data = evaluate_cvbench(target_file)
        else:
            print(f"ERROR: Unsupported dataset for xlsx format: {args.data}")
            exit()
    else:
        print(f"ERROR: File not found at {target_file}")
        exit()

# 2. JSON
elif os.path.exists(json_path):
    print(f"Reading file from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    for key in list(data.keys()):
        if isinstance(data[key], (dict, list)): del data[key]

# 3. CSV
elif os.path.exists(acc_csv_path):
    print(f"Reading file from: {acc_csv_path}")
    df_acc = pd.read_csv(acc_csv_path)
    if 'split' in df_acc.columns:
        for col in df_acc.columns:
            if col != 'split': data[col] = df_acc[col].iloc[0]
    else:
        data = df_acc.iloc[0].to_dict()
    
    if os.path.exists(acc_by_relation_csv_path):
        df_acc_rel = pd.read_csv(acc_by_relation_csv_path)
        for col in df_acc_rel.columns:
            if col != 'split' and col not in data: data[col] = df_acc_rel[col].iloc[0]

else:
    print(f"ERROR: No score file found!")
    print(f"Tried: {json_path}, {acc_csv_path}")
    print(f"Search locations: {search_dirs}")
    exit()

# --- Print Results ---
print("\nKeys in the JSON file:")
keys = list(data.keys())
# Sort keys: 'Overall' first, then alphabetical
if 'Overall' in keys:
    keys.remove('Overall')
    # keys.sort()
    keys.insert(0, 'Overall')
# else:
#     keys.sort()

for key in keys:
    print(key)

print("\nValues in the JSON file:")
for key in keys:
    print(data[key])
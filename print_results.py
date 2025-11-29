import os
import json
import argparse
import glob
import pandas as pd
import ast
import re

parser = argparse.ArgumentParser(
    description='Save score json w/o answers')
parser.add_argument('--data', type=str, help='Names of Dataset')
parser.add_argument('--model', type=str, help='Names of Model')
parser.add_argument('--date', type=str, help='Date of the experiment')
parser.add_argument('format', nargs='?', default=None, help='Format to read (xlsx or json)')
args = parser.parse_args()

base_path = os.path.join('outputs', args.model)
folder_pattern = os.path.join(base_path, f"T{args.date}_G*")

found_folders = glob.glob(folder_pattern)

if not found_folders:
    print(f"ERROR: No matching folder found for pattern: {folder_pattern}")
    exit()

folder_name = os.path.basename(found_folders[0])
try:
    commit_id = folder_name.split('_')[1]
except IndexError:
    print(f"ERROR: Could not extract Commit ID from folder name: {folder_name}")
    exit()

print(f"Found target folder: {folder_name}")
print(f"Extracted Commit ID: {commit_id}")

# Try to find score file in different formats
base_filename = os.path.join(base_path, folder_name, args.model + '_' + args.data)
json_path = base_filename + '_score.json'
acc_csv_path = base_filename + '_acc.csv'
acc_by_relation_csv_path = base_filename + '_acc_by_relation.csv'
xlsx_path = base_filename + '.xlsx'

data = {}

# Function to extract answer from prediction
def extract_answer(pred):
    """Extract answer from prediction string or dict"""
    if pd.isna(pred):
        return None
    
    # If it's already a string (single character answer)
    if isinstance(pred, str):
        # Try to parse as dict if it looks like one
        if pred.strip().startswith('{'):
            try:
                pred_dict = ast.literal_eval(pred)
                if isinstance(pred_dict, dict) and 'Answer' in pred_dict:
                    return pred_dict['Answer']
            except:
                pass
        # Return as is if it's a single character
        if len(pred.strip()) == 1:
            return pred.strip()
    
    # If it's a dict
    if isinstance(pred, dict):
        if 'Answer' in pred:
            return pred['Answer']
    
    return None

# Function to extract answer from prediction for xlsx (with additional pattern matching)
def extract_answer_xlsx(pred):
    """Extract answer from prediction string or dict for xlsx files"""
    if pd.isna(pred):
        return None
    
    # If it's a string
    if isinstance(pred, str):
        # Try to parse as dict if it looks like one
        if pred.strip().startswith('{'):
            try:
                pred_dict = ast.literal_eval(pred)
                if isinstance(pred_dict, dict) and 'Answer' in pred_dict:
                    return pred_dict['Answer']
            except:
                pass
        
        # Check for pattern "A. ", "B. ", "C. ", "D. " (capital letter + dot + space)
        pattern = r'^([A-D])\.\s'
        match = re.match(pattern, pred.strip())
        if match:
            return match.group(1)
        
        # Return as is if it's a single character
        if len(pred.strip()) == 1:
            return pred.strip()
    
    # If it's a dict
    if isinstance(pred, dict):
        if 'Answer' in pred:
            return pred['Answer']
    
    return None

# Function to evaluate ERQA from xlsx
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
    """Evaluate EmbSpatial benchmark from xlsx file"""
    print(f"Reading file from: {xlsx_path}")
    
    # Read xlsx file
    df = pd.read_excel(xlsx_path)
    
    # Print columns for debugging
    print(f"Available columns: {list(df.columns)}")
    
    # Find prediction column
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
    
    # Extract answers from predictions
    predictions = []
    for pred in df[pred_col]:
        answer = extract_answer_xlsx(pred)
        predictions.append(answer)
    
    df['extracted_answer'] = predictions
    
    # Calculate overall accuracy
    correct = 0
    total = 0
    for pred, gt in zip(df['extracted_answer'], df[answer_col]):
        if not pd.isna(gt):
            total += 1
            if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                correct += 1
    
    overall_acc = (correct / total) if total > 0 else 0
    
    # Initialize results with overall accuracy
    results = {
        'Overall': overall_acc
    }
    
    # Find relation column - check multiple possible names
    relation_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'relation' in col_lower or 'spatial' in col_lower or col == 'category':
            relation_col = col
            print(f"Found relation column: {relation_col}")
            break
    
    # Find source column - check multiple possible names
    source_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'source' in col_lower or 'dataset' in col_lower or 'origin' in col_lower:
            source_col = col
            print(f"Found source column: {source_col}")
            break
    
    # Calculate accuracy by relation
    relation_stats = {}
    if relation_col:
        relations = df[relation_col].unique()
        print(f"Relations found: {relations}")
        for rel in relations:
            if pd.isna(rel):
                continue
            rel_df = df[df[relation_col] == rel]
            rel_correct = 0
            rel_total = 0
            for pred, gt in zip(rel_df['extracted_answer'], rel_df[answer_col]):
                if not pd.isna(gt):
                    rel_total += 1
                    if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                        rel_correct += 1
            
            rel_acc = (rel_correct / rel_total) if rel_total > 0 else 0
            rel_name = str(rel).strip().lower()  # Convert to lowercase for consistency
            relation_stats[rel_name] = rel_acc
    else:
        print("WARNING: Could not find relation column")
    
    # Calculate accuracy by source
    source_stats = {}
    if source_col:
        sources = df[source_col].unique()
        print(f"Sources found: {sources}")
        for src in sources:
            if pd.isna(src):
                continue
            src_df = df[df[source_col] == src]
            src_correct = 0
            src_total = 0
            for pred, gt in zip(src_df['extracted_answer'], src_df[answer_col]):
                if not pd.isna(gt):
                    src_total += 1
                    if pred is not None and str(pred).strip().upper() == str(gt).strip().upper():
                        src_correct += 1
            
            src_acc = (src_correct / src_total) if src_total > 0 else 0
            src_name = str(src).strip().lower()  # Convert to lowercase
            # Add 'source_' prefix if not present
            if not src_name.startswith('source_'):
                src_name = f'source_{src_name}'
            source_stats[src_name] = src_acc
    else:
        print("WARNING: Could not find source column")
    
    # Define order for relations and sources
    relation_order = ['above', 'close', 'far', 'left', 'right', 'under']
    source_order = ['source_ai2thor', 'source_mp3d', 'source_scannet']
    
    # Add relations in order
    for rel in relation_order:
        if rel in relation_stats:
            results[rel] = relation_stats[rel]
        else:
            print(f"WARNING: Relation '{rel}' not found in data")
    
    # Add sources in order
    for src in source_order:
        if src in source_stats:
            results[src] = source_stats[src]
        else:
            print(f"WARNING: Source '{src}' not found in data")
    
    return results

# Check if format is explicitly set to xlsx
if args.format == 'xlsx':
    if os.path.exists(xlsx_path):
        # Determine which dataset we're evaluating
        if 'ERQA' in args.data.upper():
            data = evaluate_erqa_xlsx(xlsx_path)
        elif 'EMBSPATIAL' in args.data.upper() or 'EMB_SPATIAL' in args.data.upper():
            data = evaluate_embspatial_xlsx(xlsx_path)
        else:
            print(f"ERROR: Unsupported dataset for xlsx format: {args.data}")
            exit()
    else:
        print(f"ERROR: XLSX file not found at {xlsx_path}")
        exit()

# Otherwise use default logic (json first)
elif os.path.exists(json_path):
    print(f"Reading file from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Remove dict and list entries (keep only scalar values)
    for key in list(data.keys()):
        if type(data[key]) == dict or type(data[key]) == list:
            del data[key]

# Try CSV format (for EmbSpatialBench and similar datasets)
elif os.path.exists(acc_csv_path):
    print(f"Reading file from: {acc_csv_path}")
    
    # Read main accuracy file
    df_acc = pd.read_csv(acc_csv_path)
    
    # Convert DataFrame to dict
    # Assuming format: columns are metrics, first row contains values
    if 'split' in df_acc.columns:
        # Format: split | Overall | metric1 | metric2 | ...
        for col in df_acc.columns:
            if col != 'split':
                data[col] = df_acc[col].iloc[0]
    else:
        # Alternative format: just convert first row to dict
        data = df_acc.iloc[0].to_dict()
    
    # Also read per-relation accuracy if available
    if os.path.exists(acc_by_relation_csv_path):
        df_acc_rel = pd.read_csv(acc_by_relation_csv_path)
        
        # Add relation-specific accuracies
        for col in df_acc_rel.columns:
            if col != 'split' and col not in data:
                data[col] = df_acc_rel[col].iloc[0]

else:
    print(f"ERROR: No score file found!")
    print(f"Tried:")
    print(f"  - {json_path}")
    print(f"  - {acc_csv_path}")
    exit()

# Print results
print("\nKeys in the JSON file:")
for key in list(data.keys()):
    print(key)

print("\nValues in the JSON file:")
for key in list(data.keys()):
    print(data[key])
import os
import json
import argparse
# from vlmeval.smp import githash
import glob

parser = argparse.ArgumentParser(
    description='Save score json w/o answers')
parser.add_argument('--data', type=str, help='Names of Dataset')
parser.add_argument('--model', type=str, help='Names of Model')
parser.add_argument('--date', type=str, help='Date of the experiment')
# parser.add_argument('--folder', type=str, help='Path to the folder containing result files')
args = parser.parse_args()

# commit_id = githash(digits=8)
# commit_id = 'G' + commit_id
# print(f"Commit ID: {commit_id}")

# with open(os.path.join('outputs', args.model, 'T' + args.date + '_' + commit_id, args.model + '_' + args.data + '_score.json'), 'r') as f:
#     data = json.load(f)


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

file_path = os.path.join(base_path, folder_name, args.model + '_' + args.data + '_score.json')
print(f"Reading file from: {file_path}")

with open(file_path, 'r') as f:
    data = json.load(f)

for key in list(data.keys()):
    if type(data[key]) == dict or type(data[key]) == list:
        del data[key]

print("\nKeys in the JSON file:")
for key in list(data.keys()):
    print(key)

print("\nValues in the JSON file:")
for key in list(data.keys()):
    print(data[key])
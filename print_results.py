import os
import json
import argparse
from vlmeval.smp import githash

parser = argparse.ArgumentParser(
    description='Save score json w/o answers')
parser.add_argument('--data', type=str, help='Names of Dataset')
parser.add_argument('--model', type=str, help='Names of Model')
parser.add_argument('--date', type=str, help='Date of the experiment')
# parser.add_argument('--folder', type=str, help='Path to the folder containing result files')
args = parser.parse_args()

commit_id = githash(digits=8)
commit_id = 'G' + commit_id
print(f"Commit ID: {commit_id}")

with open(os.path.join('outputs', args.model, 'T' + args.date + '_' + commit_id, args.model + '_' + args.data + '_score.json'), 'r') as f:
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
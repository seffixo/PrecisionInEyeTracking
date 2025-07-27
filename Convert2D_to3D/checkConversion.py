import os
import argparse
from pathlib import Path
import json

'''
find both files needed: 
- ground truth raw eye tracking data in 2D
- converted 2D data from raw eye tracking data in 3D using camera parameters

check how different the ground truth 2D data and the converted 2D data is, calculate average difference
 and save result in json. 

'''

def findConversionFolders(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d == "conversion":
                yield os.path.join(root, d)


def checkDiff(base_path):
    # Compute total signed difference
    total_diff = 0.0
    count = 0
    #find and open needed files
    gt_path = os.path.join(base_path, "2D3DGazeLists.py")
    conv_path = os.path.join(base_path, "normalized_coords.py")

    gt_data = {}
    conv_data = {}
    with open(gt_path) as gt:
        exec(gt.read(), gt_data)
    
    with open(conv_path) as conv:
        exec(conv.read(), conv_data)

    ground_truth = gt_data["normalized_image_points_data"]
    converted = conv_data["normalized_coords"]

    for (gt1, gt2), (conv1, conv2) in zip(ground_truth, converted):
        total_diff += (conv1 - gt1) + (conv2 - gt2)
        count += 2  # two values per tuple

    # Average signed difference
    average_diff = total_diff / count

    # Get parent folder name (one level above "conversion")
    base_path = Path(base_path)
    parent_folder_name = base_path.parent.name
    entry_id = parent_folder_name.split('_')[0] 

    # Round and convert to float to avoid scientific notation
    #formatted_diff = f"{average_diff:.6f}"

    return average_diff, entry_id
    #print(f"On average, predicted values are {average_diff:+.6f} different from ground truth.")
    


def saveResults(average_diff, entry_id, json_path):
    #check existence and open json file
    json_path = Path(json_path)
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Add or update
    data.setdefault(entry_id, []).append(average_diff)

    # Save it back
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    #print(f"Saved result to: {json_path}")

def globalDiff(json_path):
    #calculate global difference for ground truth and converted data
    data = {}
    with open(json_path, "r") as results:
        data = json.load(results)

    # Extract all average differences (excluding existing "globalDiff" if present)
    all_diffs = []

    for key, value in data.items():
        if key != "globalDiff" and isinstance(value, list):
            all_diffs.extend(value)

    # Compute global average
    if all_diffs:
        global_diff = sum(all_diffs) / len(all_diffs)
    else:
        global_diff = 0.0

    # Insert at the top (Python 3.7+ keeps dicts ordered)
    # Round and convert to float to avoid scientific notation
    formatted_gobal_diff = f"{global_diff:.6f}"
    new_data = {"globalDiff": formatted_gobal_diff}
    new_data.update(data)

    # Save it back
    with open(json_path, 'w') as f:
        json.dump(new_data, f, indent=2) 

    print(f"global diff was saved to: {json_path}")   


def main():
    parser = argparse.ArgumentParser(description="Check matching of normalized coordinates.")
    parser.add_argument("--root_dir", help="Root directory to scan.")
    parser.add_argument("--result_name", help="name/path of the result json file")
    args = parser.parse_args()

    json_path = os.path.join(args.root_dir, args.result_name)


    for conversion_folder in findConversionFolders(args.root_dir):
        #print(f"→ Processing: {conversion_folder}")
        average_diff, entry_id = checkDiff(conversion_folder)
        saveResults(average_diff, entry_id, json_path)

    globalDiff(json_path)


if __name__ == "__main__":
    main()

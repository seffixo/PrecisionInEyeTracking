
import os
import re
import pandas as pd
import numpy as np
from numpy.linalg import norm
from math import degrees
from pathlib import Path
from collections import defaultdict

# Regex to match the main folder pattern
PREFIX_RE = re.compile(r"^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$", re.IGNORECASE)

# Regex to match gaze filenames
GAZE_FILE_RE = re.compile(r"([A-Z]{2})_(P0\d{2}_\d+_[^_]+)_filtered_gaze\.csv", re.IGNORECASE)

def calculate_mae(gaze_vecs, gt_vecs):
    # Ensure both arrays are normalized (row-wise)
    gaze_vecs_normalized = gaze_vecs / norm(gaze_vecs, axis=1)[:, None]
    gt_vecs_normalized = gt_vecs / norm(gt_vecs, axis=1)[:, None]

    # Compute dot product between corresponding vectors
    dot_products = np.sum(gaze_vecs_normalized * gt_vecs_normalized, axis=1)

    # Clip for numerical stability to avoid 1.00000000001
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Angular error (in radians) -> then degrees
    angular_errors_rad = np.arccos(dot_products)
    angular_errors_deg = degrees(1) * angular_errors_rad

    # Return MAE (mean angular error) and count
    return np.mean(angular_errors_deg), len(angular_errors_deg)

def extract_distance_lighting(base):
    # Example base: P002_120_3L → extract "120" and "3L"
    parts = base.split("_")
    if len(parts) >= 3:
        distance = parts[1]
        lighting = parts[2]
        return f"{distance}_{lighting}"
    else:
        return "unknown"

def process_folder(root_dir):
    #store vectors grouped by label
    grouped_vectors = defaultdict(lambda: {'gaze': [], 'gt': []})

    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        if PREFIX_RE.match(folder_name):
            print(f"working with {Path(folder_name).name}")
            normalized_gt_path = os.path.join(root, "interpolation", "gt_data", "normalized_gt")
            normalized_gaze_path = os.path.join(root, "filtered_gazedata", "normalized_gazedata")

            if not (os.path.isdir(normalized_gt_path) and os.path.isdir(normalized_gaze_path)):
                continue

            for gaze_file in os.listdir(normalized_gaze_path):
                match = GAZE_FILE_RE.match(gaze_file)
                if not match:
                    continue

                label, base = match.groups()
                group_key = f"{label}_{extract_distance_lighting(base)}"

                gt_file = f"{base}_{label}.csv"

                gaze_path = os.path.join(normalized_gaze_path, gaze_file)
                gt_path = os.path.join(normalized_gt_path, gt_file)

                if not os.path.exists(gt_path):
                    continue

                df_gaze = pd.read_csv(gaze_path)
                df_gt = pd.read_csv(gt_path)

                # Merge on timestamp
                df_merged = pd.merge(df_gaze, df_gt, on="timestamp", suffixes=('_gaze', '_gt'))
                if df_merged.empty:
                    continue

                # Extract direction vectors
                gaze_vecs = df_merged[['dir_cam_x_gaze', 'dir_cam_y_gaze', 'dir_cam_z_gaze']].to_numpy()
                gt_vecs = df_merged[['dir_cam_x_gt', 'dir_cam_y_gt', 'dir_cam_z_gt']].to_numpy()

                #print(f"grouping vectors for group {group_key} {base}")
                grouped_vectors[group_key]['gaze'].append(gaze_vecs)
                grouped_vectors[group_key]['gt'].append(gt_vecs)

    # Output to file
    output_base = os.path.join(root_dir, "grouped_accuracy")
    os.makedirs(output_base, exist_ok=True)

    for group_key, data in grouped_vectors.items():
        all_gaze = np.vstack(data['gaze'])  #stack all gaze vectors
        all_gt = np.vstack(data['gt'])      #stack all ground truth, image vectors

        mae, count = calculate_mae(all_gaze, all_gt)

        group_folder = os.path.join(output_base, group_key)
        os.makedirs(group_folder, exist_ok=True)

        output_path = os.path.join(group_folder, f"{group_key}_group_acc.txt")
        with open(output_path, "w") as f:
            f.write(f"Group: {group_key}\n")
            f.write(f"Mean Angular Error (degrees): {mae:.4f}\n")
            f.write(f"Matched timestamps: {count}\n")

        print(f"[{group_key}] MAE: {mae:.2f}° over {count} samples.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python calculate_accuracy.py <root_directory>")
    else:
        root_directory = sys.argv[1]
        process_folder(root_directory)

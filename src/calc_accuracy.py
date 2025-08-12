
import os
import re
import pandas as pd
import numpy as np
from numpy.linalg import norm
from math import degrees
from pathlib import Path

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

    # Clip for numerical stability
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Angular error (in radians) -> then degrees
    angular_errors_rad = np.arccos(dot_products)
    angular_errors_deg = degrees(1) * angular_errors_rad

    # Return MAE (mean angular error) and count
    return np.mean(angular_errors_deg), len(angular_errors_deg)

def process_folder(root_dir):
    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        if PREFIX_RE.match(folder_name):
            normalized_gt_path = os.path.join(root, "interpolation", "gt_data", "normalized_gt")
            normalized_gaze_path = os.path.join(root, "filtered_gazedata", "normalized_gazedata")

            if not (os.path.isdir(normalized_gt_path) and os.path.isdir(normalized_gaze_path)):
                continue

            for gaze_file in os.listdir(normalized_gaze_path):
                match = GAZE_FILE_RE.match(gaze_file)
                if not match:
                    continue

                label, base = match.groups()
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

                mae, count = calculate_mae(gaze_vecs, gt_vecs)
                if count < 100: 
                    print(f"warning: very little matched timestamps in {Path(root).name}")

                # Output to file
                output_filename = gaze_file.replace("_filtered_gaze.csv", "_accuracy.txt")
                output_folder = os.path.join(root, "accuracy")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w") as f:
                    f.write(f"Mean Angular Error (degrees): {mae:.4f}\n")
                    f.write(f"Matched timestamps: {count}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python calculate_accuracy.py <root_directory>")
    else:
        root_directory = sys.argv[1]
        process_folder(root_directory)

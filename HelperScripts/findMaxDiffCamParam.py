import os
import json
import numpy as np
from itertools import combinations

def find_camera_parameter_files(root_dir):
    camera_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "conversion" and "camera_parameters.json" in filenames:
            camera_files.append(os.path.join(dirpath, "camera_parameters.json"))
    return camera_files

def load_camera_params(filepath, root_dir):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rel_path = os.path.relpath(filepath, root_dir)
    return {
        "file": rel_path,
        "camera_matrix": data["camera_matrix"],
        "distortion_coefficients": data["distortion_coefficients"],
        "rvec": data["rvec"],
        "tvec": data["tvec"]
    }

def find_max_differences(params_list):
    max_diffs = {
        "camera_matrix": {"diff": 0},
        "distortion_coefficients": {"diff": 0},
        "rvec": {"diff": 0},
        "tvec": {"diff": 0}
    }

    for a, b in combinations(params_list, 2):
        # Convert lists to numpy arrays for diff calculations
        diff_camera_matrix = float(np.max(np.abs(np.array(a["camera_matrix"]) - np.array(b["camera_matrix"]))))
        diff_dist_coeffs = float(np.max(np.abs(np.array(a["distortion_coefficients"]) - np.array(b["distortion_coefficients"]))))
        diff_rvec = float(np.max(np.abs(np.array(a["rvec"]) - np.array(b["rvec"]))))
        diff_tvec = float(np.max(np.abs(np.array(a["tvec"]) - np.array(b["tvec"]))))

        if diff_camera_matrix > max_diffs["camera_matrix"]["diff"]:
            max_diffs["camera_matrix"] = {
                "diff": diff_camera_matrix,
                "value_a": a["camera_matrix"],
                "value_b": b["camera_matrix"],
                "file_a": a["file"],
                "file_b": b["file"]
            }

        if diff_dist_coeffs > max_diffs["distortion_coefficients"]["diff"]:
            max_diffs["distortion_coefficients"] = {
                "diff": diff_dist_coeffs,
                "value_a": a["distortion_coefficients"],
                "value_b": b["distortion_coefficients"],
                "file_a": a["file"],
                "file_b": b["file"]
            }

        if diff_rvec > max_diffs["rvec"]["diff"]:
            max_diffs["rvec"] = {
                "diff": diff_rvec,
                "value_a": a["rvec"],
                "value_b": b["rvec"],
                "file_a": a["file"],
                "file_b": b["file"]
            }

        if diff_tvec > max_diffs["tvec"]["diff"]:
            max_diffs["tvec"] = {
                "diff": diff_tvec,
                "value_a": a["tvec"],
                "value_b": b["tvec"],
                "file_a": a["file"],
                "file_b": b["file"]
            }

    # Remove the "diff" values before saving to match your desired output
    #for key in max_diffs:
     #   max_diffs[key].pop("diff", None)

    return max_diffs

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def main():
    root_dir = os.path.abspath("./Recordings_static/581_stat")  # Adjust as needed
    output_file = os.path.join(root_dir, "max_camera_parameter_differences_581.json")

    print("🔍 Searching for camera parameter files...")
    files = find_camera_parameter_files(root_dir)
    print(f"✅ Found {len(files)} files.")

    print("📥 Loading camera parameters...")
    params_list = [load_camera_params(f, root_dir) for f in files]

    print("📊 Finding max differences across all pairs...")
    max_differences = find_max_differences(params_list)

    print(f"💾 Saving results to {output_file}...")
    save_results(max_differences, output_file)
    print("✅ Done.")

if __name__ == "__main__":
    main()

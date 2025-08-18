#!/usr/bin/env python3
import os
import json

def load_gaze_points(json_path):
    img_pts = []
    obj_pts = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = evt.get("data", {})
            g2 = data.get("gaze2d")
            g3 = data.get("gaze3d")
            if g2 is not None and g3 is not None:
                img_pts.append((g2[0], g2[1]))
                obj_pts.append((g3[0], g3[1], g3[2]))
    return img_pts, obj_pts

def dump_python_arrays(img_pts, obj_pts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from filtered gazedata\n\n")
        f.write("normalized_image_points_data = [\n")
        for x, y in img_pts:
            f.write(f"    ({x:.8f}, {y:.8f}),\n")
        f.write("]\n\n")
        f.write("object_points_data = [\n")
        for x, y, z in obj_pts:
            f.write(f"    ({x:.4f}, {y:.4f}, {z:.4f}),\n")
        f.write("]\n")

def process_all_folders(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        has_event_file = any(f.lower() == "event_time_ranges.txt" for f in filenames)
        #filtered_files = [f for f in filenames if f.startswith("filtered_gazedata") and f.endswith(".jsonl")]
        filtered_files = [f for f in filenames if f.startswith("gazedata_P")]

        if not has_event_file or not filtered_files:
            continue  # Skip folders missing needed files

        for fname in filtered_files:
            input_path = os.path.join(dirpath, fname)
            img_pts, obj_pts = load_gaze_points(input_path)

            if not img_pts:
                print(f"Skipping {fname}: no valid gaze2d/gaze3d pairs found.")
                continue

            # Create the 'conversion' subfolder in the same folder
            conversion_folder = os.path.join(dirpath, "conversion")
            os.makedirs(conversion_folder, exist_ok=True)

            output_path = os.path.join(conversion_folder, "2D3DGazeLists.py")
            dump_python_arrays(img_pts, obj_pts, output_path)
            print(f"{fname}: wrote {len(img_pts)} points to {output_path}")

if __name__ == "__main__":
    ROOT_DIR = "D:\WorkingFolder_PythonD\special_dynam"
    process_all_folders(ROOT_DIR)

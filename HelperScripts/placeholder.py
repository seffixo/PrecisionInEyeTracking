import os
import json
import argparse

def find_opencv_folders(root_dir):
    opencv_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == "OpenCV":
            opencv_dirs.append(dirpath)
    return opencv_dirs

def check_json_files(opencv_dir):
    for subdir in os.listdir(opencv_dir):
        subdir_path = os.path.join(opencv_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    json_path = os.path.join(subdir_path, file)
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        if not isinstance(data, dict) or len(data) != 9:
                            print(f"Invalid! Found {len(data)} in {json_path}, instead of 9 entries.")
                    except Exception as e:
                        print(f"Failed to read JSON {json_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check JSON files in OpenCV folders for exactly 9 entries.")
    parser.add_argument("root_dir", help="Root directory to search in")
    args = parser.parse_args()

    opencv_dirs = find_opencv_folders(args.root_dir)
    if not opencv_dirs:
        print("No 'OpenCV' folders found.")
        return

    for opencv_dir in opencv_dirs:
        check_json_files(opencv_dir)

if __name__ == "__main__":
    main()

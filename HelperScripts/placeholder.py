import os
import json
import argparse

# Define the lists
list_521 = {"P002", "P018", "P019", "P020", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P034", "P035", "P036"}
list_581 = {"P004", "P005", "P006", "P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016", "P017", "P033"}

def check_rms_errors(root_dir):
    min_521_error = float('inf')
    min_521_folder = None

    min_581_error = float('inf')
    min_581_folder = None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'camera_parameters.json' in filenames and os.path.basename(dirpath) == 'conversion':
            json_path = os.path.join(dirpath, 'camera_parameters.json')
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    rms_error = data.get('rms_error', 0)

                    if rms_error < 1:
                        parent_path = os.path.dirname(dirpath)  # e.g., P004_120_3lights
                        parent_folder = os.path.basename(parent_path)
                        prefix = parent_folder.split('_')[0]     # e.g., P004

                        if prefix in list_521 and rms_error < min_521_error:
                            min_521_error = rms_error
                            min_521_folder = parent_folder
                        elif prefix in list_581 and rms_error < min_581_error:
                            min_581_error = rms_error
                            min_581_folder = parent_folder
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {json_path}: {e}")

    if min_521_folder:
        print(f"list 521: smallest rms_error in {min_521_folder} being {min_521_error}")
    else:
        print("list 521: No rms_error < 1 found")

    if min_581_folder:
        print(f"list 581: smallest rms_error in {min_581_folder} being {min_581_error}")
    else:
        print("list 581: No rms_error < 1 found")

def main():
    parser = argparse.ArgumentParser(description="Find the smallest rms_error < 1 for each list.")
    parser.add_argument("--root_dir", help="Path to the root directory (e.g., Recordings_static)")
    args = parser.parse_args()

    check_rms_errors(args.root_dir)

if __name__ == "__main__":
    main()

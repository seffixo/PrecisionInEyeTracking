import os
import json

def load_time_ranges(path):
    with open(path, 'r') as f:
        ranges = []
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    start, end = float(parts[0]), float(parts[1])
                    ranges.append((start, end))
                except ValueError:
                    continue
        return ranges

def timestamp_in_ranges(timestamp, ranges):
    return any(start <= timestamp <= end for start, end in ranges)

def process_folder(folder_path):
    txt_path = os.path.join(folder_path, "Event_time_ranges.txt")
    gaze_file = None
    gaze_filename = None

    if not os.path.isfile(txt_path):
        return  # Skip if no Event_time_ranges.txt

    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if f.startswith("gazedata_P0") and os.path.isfile(full_path):
            gaze_file = full_path
            gaze_filename = f
            break

    if not gaze_file:
        print(f"no gaze file found for {folder_path}")
        return  # Skip if no gaze file found

    # Load time ranges
    time_ranges = load_time_ranges(txt_path)

    # Output filename
    output_filename = f"filtered_{gaze_filename}.jsonl"
    output_path = os.path.join(folder_path, output_filename)

    with open(gaze_file, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                if entry.get("type") == "gaze":
                    timestamp = entry.get("timestamp")
                    if timestamp_in_ranges(timestamp, time_ranges):
                        outfile.write(json.dumps(entry) + '\n')
            except json.JSONDecodeError:
                continue

# Traverse subdirectories
root_dir = './Recordings_static'  # Set your root folder here

for dirpath, dirnames, filenames in os.walk(root_dir):
    process_folder(dirpath)

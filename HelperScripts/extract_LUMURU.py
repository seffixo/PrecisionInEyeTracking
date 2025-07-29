import os
import json

# Folder to scan
folder_path = "..\\..\\WorkingFolder_Python\\Conv2D_to3D\\581_stat_conv"

def findRecordings(folder_path):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        event_file = None
        gaze_file = None
        
        # Check for the presence of the desired files
        for filename in filenames:
            if filename == "Event_time_ranges.txt":
                event_file = os.path.join(dirpath, filename)
            elif filename.startswith("filtered_gazedata_"):
                gaze_file = os.path.join(dirpath, filename)

        # Yield only if both are found
        if event_file and gaze_file:
            yield event_file, gaze_file

def parseTimeRanges(time_range_path):
    time_ranges = []
    with open(time_range_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                label = parts[0].strip()
                start = float(parts[1])
                end = float(parts[2])
                time_ranges.append((label, start, end))
    return time_ranges

def extractLUMU(time_range_path, filtered_gazedata):
    time_ranges = parseTimeRanges(time_range_path)
    with open(filtered_gazedata, 'r') as f:
        lines = [json.loads(line) for line in f]

    # Remove entries with empty "data"
    lines = [entry for entry in lines if entry.get("type") == "gaze" and entry.get("data")]

    base_filename = os.path.basename(filtered_gazedata)
    suffix = base_filename.replace("filtered_gazedata_", "").replace(".jsonl", "")


    for label, start, end in time_ranges:
        filtered = [entry for entry in lines if start <= entry["timestamp"] <= end]
        if filtered:
            output_filename = f"{label}_{suffix}.jsonl"
            output_folder = os.path.join(os.path.dirname(filtered_gazedata), "conversion", "separated_time_gazedata")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'w') as out_file:
                for entry in filtered:
                    out_file.write(json.dumps(entry) + "\n")
    
    print(f"output saved to: {output_path}")

def main(): 
    # Folder to scan
    folder_path = "..\\..\\WorkingFolder_Python\\Conv2D_to3D\\521_stat_conv"

    for event_file, gaze_file in findRecordings(folder_path):
        extractLUMU(event_file, gaze_file)


if __name__ == "__main__":
    main()
import json
import numpy as np
import argparse
import os

def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def load_time_ranges(txt_file):
    ranges = []
    with open(txt_file, "r") as f:
        for line in f:
            start, end = map(float, line.strip().split(","))
            ranges.append((start, end))
    return ranges

def compute_stats(filtered_entries):
    #x_values = [entry["gaze3d"][0] for entry in filtered_entries]
    #y_values = [entry["gaze3d"][1] for entry in filtered_entries]
    #z_values = [entry["gaze3d"][2] for entry in filtered_entries]
    x_values = [entry["gaze2d"][0] for entry in filtered_entries]
    y_values = [entry["gaze2d"][1] for entry in filtered_entries]
    #z_values = [entry["schnittpunkt"][2] for entry in filtered_entries]

    #min_x, min_y, min_z = round(min(x_values), 3), round(min(y_values), 3), round(min(z_values), 3)
    #max_x, max_y, max_z = round(max(x_values), 3), round(max(y_values), 3), round(max(z_values), 3)
    #mean_x, mean_y, mean_z = round(np.mean(x_values), 3), round(np.mean(y_values), 3), round(np.mean(z_values), 3)
    
    min_x, min_y = round(min(x_values), 3), round(min(y_values), 3)
    max_x, max_y = round(max(x_values), 3), round(max(y_values), 3)
    mean_x, mean_y = round(np.mean(x_values), 3), round(np.mean(y_values), 3)

        # Compute standard deviation (Standardabweichung)
    std_x, std_y = round(np.std(x_values), 3), round(np.std(y_values), 3)

    #return min_x, min_y, min_z, max_x, max_y, max_z, mean_x, mean_y, mean_z, std_x, std_y
    return min_x, min_y, max_x, max_y, mean_x, mean_y, std_x, std_y

def filter_gaze_data(gaze_data, time_ranges):
    filtered_results = []
    for start, end in time_ranges:
        range_data = [entry for entry in gaze_data if start <= entry["timestamp"] <= end]
        if range_data:
#            min_x, min_y, min_z, max_x, max_y, max_z, mean_x, mean_y, mean_z, std_x, std_y = compute_stats(range_data)
#            filtered_results.append({
#                "time_range": (start, end),
#                "min_x": min_x, "min_y": min_y, "min_z": min_z,
#                "max_x": max_x, "max_y": max_y, "max_z": max_z,
#                "mean_x": mean_x, "mean_y": mean_y, "mean_z": mean_z,
#                "std_x": std_x, "std_y": std_y
#            })
            min_x, min_y, max_x, max_y, mean_x, mean_y, std_x, std_y = compute_stats(range_data)
            filtered_results.append({
                            "time_range": (start, end),
                            "min_x": min_x, "min_y": min_y,
                            "max_x": max_x, "max_y": max_y,
                            "mean_x": mean_x, "mean_y": mean_y,
                            "std_x": std_x, "std_y": std_y
                        })
    return filtered_results

def save_output(filtered_data, file_name):
    output_path = "Output/dynamisch/" + file_name
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

def main(json_file, txt_file, output_file):
    gaze_data = load_json(json_file)
    time_ranges = load_time_ranges(txt_file)
    filtered_data = filter_gaze_data(gaze_data, time_ranges)
    file_name = os.path.splitext(os.path.basename(json_file))[0] + output_file
    save_output(filtered_data, file_name)
    print(f"Filtered data saved to {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze filtered gaze data for min, max, and mean gaze3d values.")
    parser.add_argument("--filtered_json", "-fj", required=True, help="Path to the filtered JSON file.")
    parser.add_argument("--timestamps", "-ts", required=True, help="Path to the timestamp ranges file.")
    parser.add_argument("--output", "-op", default="_analysis.json", help="Path to the output analysis file.")
    
    args = parser.parse_args()
    main(args.filtered_json, args.timestamps, args.output)

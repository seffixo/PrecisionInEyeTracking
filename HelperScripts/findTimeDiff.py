import os
import json
import re
from collections import defaultdict

def shorten_folder_name(folder_name):
    index = folder_name.find("P0")
    return folder_name[index:] if index != -1 else folder_name

def extract_p0_code(folder_name):
    match = re.search(r"(P0\d+)", folder_name)
    return match.group(1) if match else folder_name

def find_meta_folders(base_path):
    results = []
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "meta":
            parent_folder = os.path.basename(os.path.dirname(root))
            shortened_name = shorten_folder_name(parent_folder)
            timestamps = []

            for file_name in files:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                            if isinstance(data, dict):
                                data = [data]
                            for item in data:
                                if isinstance(item, dict) and item.get("label") == "MouseClick":
                                    timestamp = item.get("timestamp")
                                    if timestamp is not None:
                                        timestamps.append(timestamp)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            if timestamps:
                if len(timestamps) != 9:
                    print(f"Not 9 timestamps at {parent_folder}")
                    continue

                timestamps.sort()
                differences = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                if differences:
                    smallest_diff = round(min(differences),3)
                    results.append({shortened_name: smallest_diff})

    return results

def save_raw_results_to_txt(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in results:
            for folder_name, smallest_diff in entry.items():
                f.write(f"{folder_name}: {smallest_diff}\n")

def summarize_results(results):
    summarized = defaultdict(list)
    for entry in results:
        for folder_name, value in entry.items():
            p0_code = extract_p0_code(folder_name)
            summarized[p0_code].append(value)
    return {p0_code: min(values) for p0_code, values in summarized.items()}

def save_summary_to_txt(summary, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for p0_code, smallest_diff in summary.items():
            f.write(f"{p0_code}: {smallest_diff}\n")

# Example usage
if __name__ == "__main__":
    base_directory = "./Recordings_static"  # starting path

    raw_output_txt = "raw_results_static.txt"
    summarized_output_txt = "summarized_results_static.txt"

    raw_results = find_meta_folders(base_directory)

    # Save raw results
    #save_raw_results_to_txt(raw_results, raw_output_txt)

    # Summarize and save summary
    summarized_output = summarize_results(raw_results)
    save_summary_to_txt(summarized_output, summarized_output_txt)


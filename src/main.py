import json
import os
import argparse
from datetime import datetime

def read_json_file(file_path):
    """
    Reads and returns the content of a JSON or JSONL file.

    Parameters:
    file_path (str): Path to the JSON or JSONL file.

    Returns:
    list: Parsed JSON content as a list of dictionaries.
    """
    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.jsonl'):
                return [json.loads(line) for line in file]
            else:
                return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def read_timestamp_ranges_file(file_path):
    """
    Reads timestamp ranges from a file and returns them as a list of tuples.

    Parameters:
    file_path (str): Path to the file containing timestamp ranges.

    Returns:
    list: List of tuples (lower_bound, upper_bound) as floats.
    """
    timestamp_ranges = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                try:
                    lower, upper = map(float, line.split(','))
                    timestamp_ranges.append((lower, upper))
                except ValueError:
                    print(f"Invalid timestamp range format ignored: {line}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return timestamp_ranges

def shorten_timestamp(timestamp):
    """
    Shortens a timestamp to two decimal places.

    Parameters:
    timestamp (float): The original timestamp.

    Returns:
    float: Shortened timestamp.
    """
    return round(timestamp, 3)

def is_timestamp_in_ranges(timestamp, ranges):
    """
    Checks if a timestamp falls within any of the provided ranges.

    Parameters:
    timestamp (float): The timestamp to check.
    ranges (list): List of tuples (lower_bound, upper_bound).

    Returns:
    bool: True if the timestamp is in any range, False otherwise.
    """
    for lower, upper in ranges:
        if lower <= timestamp <= upper:
            return True
    return False

def filter_json_by_timestamps(json_data, timestamp_ranges):
    """
    Filters JSON data to include only the shortened timestamp, gaze2d, and gaze3d.

    Parameters:
    json_data (list): JSON data to filter.
    timestamp_ranges (list): List of timestamp ranges.

    Returns:
    list: Filtered JSON data.
    """
    filtered_entries = []

    for entry in json_data:
        timestamp = entry.get("timestamp")
        if timestamp is not None:
            shortened_timestamp = shorten_timestamp(timestamp)
            if is_timestamp_in_ranges(shortened_timestamp, timestamp_ranges):
                filtered_entry = {
                    "timestamp": shortened_timestamp,
                    "gaze3d": entry.get("data", {}).get("gaze3d"),
                    "eyeleft": entry.get("data", {}).get("eyeleft"),
                    "eyeright": entry.get("data", {}).get("eyeright")
                }
                filtered_entries.append(filtered_entry)

    return filtered_entries

def save_filtered_data_to_file(filtered_data, output_file_path):
    """
    Saves filtered JSON data to a new JSON file.

    Parameters:
    filtered_data (list): The filtered JSON data to be saved.
    output_file_path (str): Path to the output JSON file.
    """
    try:
        with open(output_file_path, 'w') as file:
            json.dump(filtered_data, file, indent=4)
        print(f"Filtered data successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to save filtered data: {e}")

def main(json_files, timestamps, output):
    timestamp_ranges = read_timestamp_ranges_file(timestamps)
    if not timestamp_ranges:
        print("No valid timestamp ranges found. Exiting.")
        return

    all_filtered_data = []

    for file_path in json_files:
        if os.path.exists(file_path) and (file_path.endswith('.json') or file_path.endswith('.jsonl')):
            data = read_json_file(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            print("file name: ", file_name)
            if data is not None:
                filtered_data = filter_json_by_timestamps(data, timestamp_ranges)
                #print(f"Filtered contents of {file_path}:\n{json.dumps(filtered_data, indent=4)}\n")
                all_filtered_data.extend(filtered_data)
        else:
            print(f"Invalid file path or not a JSON/JSONL file: {file_path}")

    if all_filtered_data:
        output_path = "OutputFiles/" + file_name + "_" + output
        save_filtered_data_to_file(all_filtered_data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSON/JSONL files by timestamp ranges and save the results in a separate JSON file.")
    parser.add_argument("--json_files", "-jf", nargs='+', required=True, help="Path(s) to the input JSON or JSONL file(s).")
    parser.add_argument("--timestamps", "-ts", required=True, help="Path to the file containing timestamp ranges.")
    parser.add_argument("--output", "-op", default="filtered_eyes.json", help="Path to the output JSON file.")

    args = parser.parse_args()
    main(args.json_files, args.timestamps, args.output)

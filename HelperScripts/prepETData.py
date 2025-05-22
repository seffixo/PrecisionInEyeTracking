import os
import json
import argparse

def extract_and_generate_mouseclick_ranges(meta_path, output_path):
    """
    Processes all JSON files in the given meta_path and writes start,end timestamps to output_path.
    """
    json_files = [f for f in os.listdir(meta_path) if f.endswith('.json')]
    timestamps = []

    for json_file in json_files:
        file_path = os.path.join(meta_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("label") == "MouseClick":
                    timestamp = data.get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        timestamps.append(timestamp)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    if not timestamps:
        print(f"No MouseClick events found in {meta_path}")
        return

    timestamps.sort()

    output_lines = []
    for ts in timestamps:
        start = round(ts, 3)
        end = round(ts + 2.0, 3)
        output_lines.append(f"{start:.3f}, {end:.3f}")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in output_lines:
            out_f.write(line + "\n")

    print(f"{len(output_lines)} MouseClick ranges written to '{output_path}'")

def process_all_participants(root_dir):
    """
    Search recursively for 'meta' folders under root_dir and process each one.
    """
    if not os.path.exists(root_dir):
        print(f"Root folder '{root_dir}' does not exist.")
        return

    meta_found = False

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == "meta":
                meta_found = True
                meta_path = os.path.join(root, dir_name)
                participant_path = os.path.dirname(meta_path)
                output_path = os.path.join(participant_path, "Event_time_ranges.txt")
                extract_and_generate_mouseclick_ranges(meta_path, output_path)

    if not meta_found:
        print("No 'meta' folders found in the specified root directory.")

def main():
    parser = argparse.ArgumentParser(description="Extract and process MouseClick timestamps.")
    parser.add_argument("--root_dir", "-rd", type=str, required=True, help="Path to the root directory containing participant folders.")

    args = parser.parse_args()
    process_all_participants(args.root_dir)

if __name__ == "__main__":
    main()

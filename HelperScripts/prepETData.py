import os
import json
import argparse
import gzip
import shutil
from pathlib import Path


def rename_participant_folders(root_dir):
    """
    Renames any folder containing 'P0' by removing everything before 'P0'.
    """
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if "P0" in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_name = dirname[dirname.index("P0"):]  # Trim before "P0"
                new_path = os.path.join(dirpath, new_name)
                if old_path != new_path:
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"Renamed folder: {old_path} → {new_path}")
                    else:
                        print(f"Skipped renaming '{old_path}': '{new_path}' already exists.")


def unpack_and_rename_gazedata_gz_files(root_dir):
    """
    Recursively looks for 'gazedata.gz' files and unpacks them into 'gazedata_<foldername>'.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "gazedata.gz":
                folder_name = os.path.basename(dirpath)
                new_filename = f"gazedata_{folder_name}"
                unpacked_path = os.path.join(dirpath, new_filename)
                gz_path = os.path.join(dirpath, filename)

                if os.path.exists(unpacked_path):
                    print(f"Skipped unpacking: '{unpacked_path}' already exists.")
                    continue

                try:
                    with gzip.open(gz_path, 'rb') as f_in, open(unpacked_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    print(f"Unpacked: {gz_path} → {unpacked_path}")
                except Exception as e:
                    print(f"Failed to unpack '{gz_path}': {e}")


def extract_and_generate_mouseclick_ranges(meta_path, output_path):
    """
    Processes all JSON files in the given meta_path and writes start,end timestamps to output_path.
    """
    json_files = [f for f in os.listdir(meta_path) if f.endswith('.json')]
    timestamps = []
    parent_path = Path(meta_path).parent
    
    if not json_files: 
        print(f"there were no json files found inside {Path(meta_path).parent.name}")
        for f in os.listdir(parent_path):
            if f.endswith('.txt') and "time_" in f:
                new_name = os.path.join(parent_path, "Event_time_ranges.txt")
                old_name = os.path.join(parent_path, f)
                os.rename(old_name, new_name)
                print(f"found txt file in {parent_path} and renamed it.")
        return

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
    Renames folders, unpacks gazedata.gz files, and processes all meta folders.
    """
    if not os.path.exists(root_dir):
        print(f"Root folder '{root_dir}' does not exist.")
        return

    #rename_participant_folders(root_dir)
    unpack_and_rename_gazedata_gz_files(root_dir)

#    meta_found = False
#    for root, dirs, _ in os.walk(root_dir):
#        for dir_name in dirs:
#            dir_path = os.path.join(root, dir_name)
#            if dir_name == "meta":
#                meta_found = True
#                meta_path = os.path.join(root, dir_name)
#                participant_path = os.path.dirname(meta_path)
#                output_path = os.path.join(participant_path, "Event_time_ranges.txt")
#                extract_and_generate_mouseclick_ranges(meta_path, output_path)

#    if not meta_found:
#        print(f"No 'meta' folders found in {Path(dir_path).parent.name}.")


def main():
    parser = argparse.ArgumentParser(description="Rename folders, unpack gazedata.gz, and extract MouseClick ranges.")
    parser.add_argument("--root_dir", "-rd", type=str, required=True, help="Path to the root directory containing participant folders.")
    args = parser.parse_args()
    process_all_participants(args.root_dir)


if __name__ == "__main__":
    main()

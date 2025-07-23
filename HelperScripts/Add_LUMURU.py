import os
import argparse

# The fixed labels in order
'''
LU == Left Up
MU == Middle Up
RU == Right Up
LM == Left Middle
MM == Middle Middle
RM == Right Middle
LD == Left Down
MD == Middle Down
RD == Right Down
'''
REGION_LABELS = ['LU', 'MU', 'RU', 'LM', 'MM', 'RM', 'LD', 'MD', 'RD']

def label_event_time_ranges(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) != len(REGION_LABELS):
            print(f"Warning: {file_path} has {len(lines)} lines, expected {len(REGION_LABELS)}. Skipping.")
            return

        new_lines = []
        for label, line in zip(REGION_LABELS, lines):
            line = line.strip()
            if line:  # Only label non-empty lines
                new_lines.append(f"{label},{line}")

        with open(file_path, 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')

        print(f"Labeled: {file_path}")

    except Exception as e:
        print(f"Failed to process '{file_path}': {e}")

def process_all_event_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "Event_time_ranges.txt":
                file_path = os.path.join(dirpath, filename)
                label_event_time_ranges(file_path)

def main():
    parser = argparse.ArgumentParser(description="Label Event_time_ranges.txt files with LU–RD tags.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing Event_time_ranges.txt files.")
    args = parser.parse_args()
    process_all_event_files(args.root_dir)

if __name__ == "__main__":
    main()

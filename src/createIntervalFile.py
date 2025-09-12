import os
from pathlib import Path
import argparse
import re
import pandas as pd


PREFIX_RE = re.compile(r"(P0\d{2}(?:_[^_]+)?_(?:80|120|180)(?:cm)?(?:_[^_]+)?_(?:basicL|bL|3lights?|3L))",re.IGNORECASE)
participant = re.compile(
    r'(P0\d{2})_'
    r'(statisch|statisch_teils|dynamisch)' #participant folder
    )

def find_event_files(root: Path):
    for folder in os.listdir(root):
        #if folder_filter and folder_filter not in dirpath:
            #continue
        if participant.match(folder):
            folder_path = os.path.join(root, folder)
            for subfolder in os.listdir(folder_path):
                if PREFIX_RE.match(subfolder):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    candidate = Path(subfolder_path) / "Event_time_ranges.txt"
                    if candidate.exists():
                        yield candidate
                        continue

def create_tsv_file(og_file, vali_path, fps):
    frame_path = os.path.join(vali_path, "Frame_ranges.tsv")
    df = pd.read_csv(og_file, header=None, names=["label", "start", "end"])
    
    df["start_frame"] = (df["start"] * fps).round().astype(int)
    df["end_frame"] = (df["end"] * fps).round().astype(int)

    frame_ranges = df[["start_frame", "end_frame"]]

    frame_ranges.to_csv(frame_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect point detections from *_coords folders and export to an interpolation folder.")
    parser.add_argument("--root", help="Path to scan (e.g. the directory that contains the 'OpenCV' folder or any ancestor).")
    parser.add_argument("--glassVali", help="path to where glassesValidator recordings are.")
    args =parser.parse_args()

    fps = 25
    root_path = args.root
    vali_path = args.glassVali

    for entry in find_event_files(root_path):
        participant = Path(entry).parent.parent.name
        participant_path = os.path.join(vali_path, participant)
        entry_parent = Path(entry).parent.name
        for subfolder in os.listdir(participant_path):
            if entry_parent in subfolder:
                recording_path = os.path.join(participant_path, subfolder)
                create_tsv_file(entry, recording_path, fps)

if __name__ == "__main__":
    main()

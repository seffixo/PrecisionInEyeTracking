#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import namedtuple
import pandas as pd
from dataclasses import asdict, is_dataclass
'''
Extract information for every participant to enable box plot creation. 
- camera_id
- participant_id
- label
- distance
- lighting
- angular_error
- matchings

'''

# --------- regex helpers ---------
pattern = re.compile(
    r'^(?:LM|RM|MM|LU|RU|MU|LD|RD|MD)_'   # label
    r'(P0\d{2})_'                         # participant
    r'(80|120|180)_'                      # distance
    r'(bL|3L)_'                           # lighting
    r'.*\.txt$'                           # trailing extra text + .txt
)
camera = re.compile(r'(521|581)')



def append_csv(row, out_csv: Path) -> None:
    #namedtuple -> dict
    data = row._asdict()
    fieldnames = ['camera_id','participant_id','label','distance','lighting','angular_error','matchings']

    file_exists = out_csv.exists()
    with out_csv.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
       
        if not file_exists or out_csv.stat().st_size == 0:
            w.writerow(fieldnames)
        
        w.writerow([data[field] for field in fieldnames])

    print(f"Wrote rows to {out_csv}")

def find_median_folders(root):
    #look through all subfolders and find median folder
    root_path = Path(root)
    return [p for p in root_path.rglob("median_accuracy") if p.is_dir()]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scan for 'median_accuracy' folders and extract metadata + angular_error.")
    ap.add_argument("--root", "-r", help="Root directory to scan (e.g., D:\\WorkingFolder_PythonD\\2Dto3D_Conversion)")
    ap.add_argument("-o", "--out", default="median_accuracy_summary.csv", help="Output CSV filename (default: %(default)s)")
    args = ap.parse_args()

    out_csv = Path(args.out)
    median_folders = find_median_folders(args.root)
    for folder in median_folders: 
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                #file should look like: label_participant_distance_lighting_median_acc.txt
                file_name = Path(file).name
                MatchResult = namedtuple("MatchResult", "camera_id participant_id label distance lighting angular_error matchings")

                if pattern.match(file_name):
                    label = file_name.split("_")[0]
                    participant_id = file_name.split("_")[1]
                    distance = file_name.split("_")[2]
                    lighting = file_name.split("_")[3]
                    
                    camera_folder = Path(folder).parent.parent.parent.name
                    camera_id = camera_folder.split("_")[0]
                    if camera.match(camera_id):
                        file_path = os.path.join(folder, file)
                        file_path = Path(file_path)
                        with file_path.open("r", encoding="utf-8") as f:
                            for line in f: 
                                line = line.strip()
                                if line.startswith("Median Angular Error"):
                                    angular_error = float(line.split(":")[1].strip())
                                elif line.startswith("Matched timestamps"):
                                    matchings = int(line.split(":")[1].strip())
                                else:
                                    print(f"something went wrong. no MAE or timestamp in {file}. Skipping!")
                                    break
                    else:
                        print(f"camera ids did not match in {file}. skipping!")
                        break
                else: 
                    print(f"something went wrong. pattern did not match {file}. Skipping!")
                    break

                row = MatchResult(camera_id, participant_id, label, distance, lighting, angular_error, matchings)
                append_csv(row, out_csv)

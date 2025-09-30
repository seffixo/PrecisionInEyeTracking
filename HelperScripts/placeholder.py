#!/usr/bin/env python3
# build_summary.py

import os
import re
import csv
import argparse
from pathlib import Path
import logging

# --- config / mappings --------------------------------------------------------
TARGET_TO_LABEL = {
    "1": "LU", "2": "MU", "3": "RU",
    "4": "LM", "5": "MM", "6": "RM",
    "7": "LD", "8": "MD", "9": "RD",
}

participant = re.compile(
    r'(P0\d{2})_'
    r'(statisch|statisch_teils|dynamisch)' #participant folder
    )

def normalize_lighting(s: str) -> str | None:
    s = s.lower()
    if "basicl" in s:
        return "bL"
    if "3light" in s or "3lights" in s or s == "3l":
        return "3L"
    return None

VALID_CAMERAS = {"521", "581"}
VALID_DISTANCES = {"80", "120", "180"}

# --- helpers ------------------------------------------------------------------
def extract_metadata_from_path(fpath: Path):
    path_str = str(fpath).replace("\\", "/")
    parts = path_str.split("/")

    camera_id = None
    for seg in parts:
        m = re.match(r"^(521|581)", seg)
        if m:
            camera_id = m.group(1)
            break

    participant_id = None
    m = re.search(r"\b(P0\d{2})_(statisch|statisch_teils|dynamisch)", path_str, flags=re.IGNORECASE)
    if m:
        participant_id = m.group(1).upper()

    distance = None
    for m in re.finditer(r"(80|120|180)\s*cm?", path_str, flags=re.IGNORECASE):
        cand = m.group(1)
        if cand in VALID_DISTANCES:
            distance = cand
            break
    if distance is None:
        for m in re.finditer(r"(80|120|180)", path_str):
            cand = m.group(1)
            if cand in VALID_DISTANCES:
                distance = cand
                break

    lighting = None
    for seg in parts:
        lit = normalize_lighting(seg)
        if lit:
            lighting = lit
            break

    return camera_id, participant_id, distance, lighting


def parse_dataquality_tsv(tsv_path: Path):
    rows = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not ("target" in reader.fieldnames and "acc" in reader.fieldnames and "rms" in reader.fieldnames):
            raise ValueError(f"{tsv_path}: missing 'target' or 'acc' columns")
        for r in reader:
            target_raw = str(r.get("target", "")).strip()
            acc_raw = str(r.get("acc", "")).strip()
            rms_raw = str(r.get("rms", "")).strip()
            if target_raw == "" or acc_raw == "" or rms_raw == "":
                continue
            label = TARGET_TO_LABEL.get(target_raw)
            if not label:
                try:
                    label = TARGET_TO_LABEL.get(str(int(float(target_raw))))
                except Exception:
                    label = None
            if not label:
                continue
            try:
                angular_error = float(acc_raw)
                rms_value = float(rms_raw)
            except ValueError:
                continue
            rows.append((label, angular_error, rms_value))
    return rows


def find_dataquality_files(root: Path, folder_filter: str | None):
    for folder in os.listdir(root):
        #if folder_filter and folder_filter not in dirpath:
            #continue
        if participant.match(folder):
            folder_path = os.path.join(root, folder)
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                candidate = Path(subfolder_path) / "dataQuality.tsv"
                if candidate.exists():
                    yield candidate
                    continue


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate Tobii 'dataQuality.tsv' files to summary_MedAEs_for_boxPlot.csv")
    ap.add_argument("--root", type=str, help="Root folder to crawl")
    ap.add_argument("-o", "--output", type=str, default="summary_MedAEs_for_boxPlot.csv", help="Output CSV path")
    ap.add_argument("--folder-filter", type=str, default=None, help="Only process subfolders whose path contains this string")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

        # --- setup logging ---
    log_file = root / "summary_boxplots.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Starting summary build")
    logging.info(f"Root folder: {root}")
    logging.info(f"Output file: {out_path}")

    with out_path.open("w", encoding="utf-8", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["camera_id", "participant_id", "label", "distance", "lighting", "angular_error", "rms"])

        n_files = 0
        n_rows = 0

        for tsv in find_dataquality_files(root, args.folder_filter):
            camera_id, participant_id, distance, lighting = extract_metadata_from_path(tsv)
            if camera_id not in VALID_CAMERAS or not participant_id or not distance or not lighting:
                logging.warning(f"Skipping {tsv} due to missing metadata: "
                                f"camera_id={camera_id}, participant={participant_id}, distance={distance}, lighting={lighting}")
                continue
            try:
                for label, angular_error, rms_value in parse_dataquality_tsv(tsv):
                    writer.writerow([camera_id, participant_id, label, distance, lighting, f"{angular_error:.6f}", f"{rms_value:.6f}"])
                    n_rows += 1
                n_files += 1
                logging.info(f"Processed {tsv} with {n_rows} total rows so far.")
            except Exception:
                continue

    print(f"Wrote {n_rows} rows from {n_files} files to: {out_path}")


if __name__ == "__main__":
    main()

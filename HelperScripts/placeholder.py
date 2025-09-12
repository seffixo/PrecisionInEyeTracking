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
LABELS = {"LU", "MU", "RU", "LM", "MM", "RM", "LD", "MD", "RD"}

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


def parse_median_accuracy_file(fpath: Path):
    """
    Reads only:
      - Median Angular Error (degrees)
      - Matched timestamps
    """
    label = Path(fpath).name.split('_')[0]
    median_acc = None
    matchings = None
    if label not in LABELS:
        pass

    try:
        with fpath.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                # Median Angular Error (degrees): 0.9910
                m_acc = re.match(
                    r"(?i)^median\s+angular\s+error\s*\(degrees\)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$",
                    line
                )
                if m_acc:
                    median_acc = float(m_acc.group(1))
                    continue

                # Matched timestamps: 897
                m_mtch = re.match(
                    r"(?i)^matched\s+timestamps\s*:\s*(\d+)\s*$",
                    line
                )
                if m_mtch:
                    matchings = int(m_mtch.group(1))
                    continue

                # RMS Angular Error (degrees): 0.6452
                m_rms = re.match(
                    r"(?i)^rms\s+angular\s+error\s*\(degrees\)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$",
                    line
                )
                if m_rms:
                    rms_acc = float(m_rms.group(1))
                    continue
    except Exception:
        return None, None

    return label, median_acc, matchings, rms_acc


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

def find_median_files(root: Path, folder_filter: str | None):
    for folder in os.listdir(root):
        if participant.match(folder):
            folder_path = os.path.join(root, folder)
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                median_folder = os.path.join(subfolder_path, "median_accuracy")
                for file in os.listdir(median_folder):
                    if file.endswith(".txt"):
                        candidate = Path(median_folder) / file 
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
        writer.writerow(["camera_id", "participant_id", "label", "distance", "lighting", "angular_error", "matchings", "rms"])

        n_files = 0
        n_rows = 0

        for tsv in find_median_files(root, args.folder_filter):
            camera_id, participant_id, distance, lighting = extract_metadata_from_path(tsv)
            if camera_id not in VALID_CAMERAS or not participant_id or not distance or not lighting:
                logging.warning(f"Skipping {tsv} due to missing metadata: "
                                f"camera_id={camera_id}, participant={participant_id}, distance={distance}, lighting={lighting}")
                continue
            try:
                label, median_acc, matchings, rms_acc = parse_median_accuracy_file(tsv)

                if label not in LABELS:
                    logging.warning(f"Skipping {tsv}: label '{label}' not in expected set {sorted(LABELS)}")
                    continue

                if median_acc is None or matchings is None:
                    logging.warning(
                        f"Skipping {tsv}: missing required values "
                        f"(median={median_acc}, matchings={matchings})"
                    )
                    continue

                writer.writerow([
                    camera_id,
                    participant_id,
                    label,
                    distance,
                    lighting,
                    f"{median_acc:.6f}",
                    matchings,
                    "" if rms_acc is None else f"{rms_acc:.6f}",
                ])
                n_rows += 1
                n_files += 1
                logging.info(f"Processed {tsv} with {n_rows} total rows so far.")
            except Exception as e:
                logging.exception(f"Failed processing {tsv}: {e}")
                continue

    print(f"Wrote {n_rows} rows from {n_files} files to: {out_path}")


if __name__ == "__main__":
    main()

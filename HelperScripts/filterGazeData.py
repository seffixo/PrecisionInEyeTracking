#!/usr/bin/env python3
"""
Recursively find gazedata files, filter by event time ranges, and export per-label CSVs.

- Scans a root directory for files named like: gazedata_P0XX_* (case-insensitive).
- For each file, walks upward to find a parent directory whose name matches:
    ^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$
  and uses a normalized prefix "P0XX_DIST_LIGHT" as the specific_folder_name in CSV outputs.
    - LIGHT normalized to "bL" or "3L"
    - DIST is "80", "120", or "180" (no "cm")
- Filters entries within each label’s [start, end] window (inclusive by default).
- Only writes rows with non-empty gaze2d -> CSV columns: timestamp,gaze2d_x,gaze2d_y
- Output CSVs go to a 'filtered_gazedata' folder next to each gaze file (or use --output-dir).
- File naming: {LABEL}_{specific_folder_name}_filtered_gaze.csv

Examples:
  python filter_gaze_batch.py --root /data/study --events-file /data/Event_time_ranges.txt
  python filter_gaze_batch.py --root . --events-file Event_time_ranges.txt --exclusive
"""

import argparse
import csv
import json
import math
import os
import re
from typing import Iterable, List, Optional, Tuple
from pathlib import Path
import logging

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\581_dynam\filter_gaze.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------- Prefix discovery (as requested) ----------
PREFIX_RE = re.compile(
    r"^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$",
    re.IGNORECASE,
)

def set_prefix(current_path: str) -> Optional[str]:
    """
    Walk upward from start_dir until a directory matches PREFIX_RE.
    Returns normalized "P0XX_DIST_LIGHT" (e.g., P012_120_3L) or None if not found.
    """
    base = os.path.basename(current_path)
    m = PREFIX_RE.match(base)
    if m:
        pnum, dist, _cm, light = m.groups()
        # Normalize
        dist_norm = dist  # drop 'cm' automatically
        l = light.lower()
        if l in ["basicl", "bl"]:
            light_norm = "bL"
        elif l in ["3lights", "3l"]:
            light_norm = "3L"
        else:
            light_norm = light
        return f"{pnum}_{dist_norm}_{light_norm}"
    else: 
        print("something went wrong. no prefix folder found!")

# ---------- Core filtering logic ----------
EventRange = Tuple[str, float, float]

def parse_event_ranges(path: str) -> List[EventRange]:
    ranges: List[EventRange] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                # also allow mixed whitespace
                parts = [p for p in re.split(r"[,\s]+", line) if p]
            if len(parts) != 3:
                raise ValueError(f"Bad event line (need LABEL,START,END): {raw!r}")
            label, start_s, end_s = parts[0], float(parts[1]), float(parts[2])
            if end_s < start_s:
                logging.error(f"End before start for label {label}: {start_s} > {end_s}")
                raise ValueError(f"End before start for label {label}: {start_s} > {end_s}")
            ranges.append((label, start_s, end_s))
    return ranges

def iter_gaze_entries(path: str) -> Iterable[Tuple[float, Optional[Tuple[float, float]]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # NEW: skip if data is missing/empty/non-dict
            data = obj.get("data")
            if not isinstance(data, dict) or not data:
                continue

            ts = obj.get("timestamp", None)
            if ts is None:
                continue
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue

            g2d = data.get("gaze2d", None)

            xy: Optional[Tuple[float, float]] = None
            if isinstance(g2d, (list, tuple)) and len(g2d) >= 2:
                x, y = g2d[0], g2d[1]
                if x is not None and y is not None:
                    try:
                        x = float(x)
                        y = float(y)
                        if not (math.isnan(x) or math.isnan(y)):
                            xy = (x, y)
                    except (TypeError, ValueError):
                        pass

            yield ts, xy

def ensure_output_dir_for_gaze(gaze_path: str) -> str:
    """
    Always create 'filtered_gazedata' in the SAME directory as the gaze file.
    """
    parent_dir = os.path.dirname(os.path.abspath(gaze_path))
    out_dir = os.path.join(parent_dir, "filtered_gazedata")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def filter_one_gaze_file(
    gaze_path: str,
    event_path: str,
    prefix_folder_name: str,
    inclusive: bool,
):
    """
    Filter a single gaze file into per-label CSVs.
    Returns list of file paths created.
    """
    ranges = parse_event_ranges(event_path)
    out_dir = ensure_output_dir_for_gaze(gaze_path)

    # Read once; keep minimal fields
    entries = [(ts, xy) for ts, xy in iter_gaze_entries(gaze_path)]
    # Pre-round all timestamps once
    entries_2dp = [(round(ts, 2), xy) for ts, xy in entries]

    for label, start, end in ranges:
        start_2dp = round(start, 2)
        end_2dp   = round(end, 2)
        
        def in_window(ts2: float) -> bool:
            return (start_2dp <= ts2 <= end_2dp) if inclusive else (start_2dp < ts2 < end_2dp)

        rows = [(
            f"{ts2:.2f}",  # timestamp with exactly 2 decimal places
            f"{round(xy[0], 4):.8f}",  # 4 digits after rounding + 4 trailing zeros → total 8 decimals
            f"{round(xy[1], 4):.8f}") 
            for ts2, xy in entries_2dp if xy is not None and in_window(ts2)]
        if not rows:
            continue

        out_name = f"{label}_{prefix_folder_name}_filtered_gaze.csv"
        out_path = os.path.join(out_dir, out_name)

        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "gaze2d_x", "gaze2d_y"])
            writer.writerows(rows)
        logging.info(f"saved filtered gazedata to: {Path(out_path).parent.parent.name} as {out_name}")
        #print(f"saved filtered gazedata to: {Path(out_path).parent.parent.name} as {out_name}")

# ---------- Discovery & CLI ----------
GAZEFILE_RE = re.compile(r"^gazedata_P0\d{2}_.+", re.IGNORECASE)

def discover_gaze_file(folder_path: str) -> str:
    """Find files whose basename matches gazedata_P0XX_* (any extension)."""
    match: str
    for dirpath, _dirnames, filenames in os.walk(folder_path):
        for fn in filenames:
            if GAZEFILE_RE.match(fn):
                match = fn
                return os.path.join(folder_path, match)

def find_event_for_gaze(folder_path: str) -> Optional[str]:
    """
    Return the path to Event_time_ranges.txt (case-insensitive) in the same folder as gaze_path.
    """
    for dirpath, _dirnames, filenames in os.walk(folder_path):
        for fn in filenames:
        # case-insensitive exact name match
            if fn.lower() == "event_time_ranges.txt":
                return os.path.join(folder_path, fn)

def main():
    parser = argparse.ArgumentParser(description="Batch-filter gaze data by event time ranges.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", help="Root directory to search for gazedata_P0XX_* files.")
    parser.add_argument("--exclusive", action="store_true",
                        help="Use exclusive bounds (start < ts < end). Default inclusive (start <= ts <= end).")
    args = parser.parse_args()

    inclusive = not args.exclusive

    root_dir = args.root
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            base = os.path.basename(subfolder_path)
            m = PREFIX_RE.match(base)
            if m: 
                gaze_path = discover_gaze_file(subfolder_path)
                if not gaze_path:
                    logging.error(f"No gazedata_P0XX_* files found under: {Path(subfolder_path).parent.name}")
                    return
                event_path = find_event_for_gaze(subfolder_path)
                if not event_path:
                    logging.error(f"[skip] No Event_time_ranges.txt next to: {Path(gaze_path).parent.name}")
                    continue

            else:
                print(f"subfolder_path: {subfolder_path} has no gazedata file.")
                return

            # Find normalized specific_folder_name by walking up from the gaze file's directory
            prefix = set_prefix(subfolder_path)
            if prefix is None:
                # Fallback: derive from file name if prefix folder is not found
                base = os.path.basename(gaze_path)
                root_no_ext, _ext = os.path.splitext(base)
                if root_no_ext.startswith("gazedata_"):
                    prefix = root_no_ext[len("gazedata_"):]
                else:
                    prefix = root_no_ext  # last resort

            filter_one_gaze_file(
                gaze_path=gaze_path,
                event_path=event_path,
                prefix_folder_name=prefix,
                inclusive=inclusive,
            )

if __name__ == "__main__":
    main()

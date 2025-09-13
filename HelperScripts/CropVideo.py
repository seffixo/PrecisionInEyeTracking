#!/usr/bin/env python3
"""
Batch-crop MP4 videos by time ranges per folder.

- Recursively walks a base directory (default: current working directory).
- Each folder that contains `event_time_ranges.txt` and at least one `.mp4`:
  - Parses time ranges from the text file.
  - Crops every `.mp4` in that folder to only those ranges.
  - Saves `<name>_cropped.mp4` next to the original.

Time range file formats supported per non-empty, non-# line:
  1) CSV with label:        LABEL, start, end          (e.g., LU,3.780, 5.780)
  2) CSV without label:     start, end                 (e.g., 3.780, 5.780)
  3) Dash/comma range:      HH:MM:SS[.ms]-HH:MM:SS[.ms]  or  start,end

Examples (mixed allowed):
  LU,3.780, 5.780
  MU,10.160, 12.160
  00:00:17.457-00:00:19.457
  24.657,26.657

Requires: ffmpeg + ffprobe on PATH.
"""

import argparse
import csv
import os
import re
import shlex
import subprocess
from typing import List, Tuple

# For "a-b" style ranges
RANGE_LINE = re.compile(r"^\s*([^,#]+?)\s*[-,]\s*([^,#]+?)\s*$")

def parse_timecode(t: str) -> float:
    """
    Parse a time string into seconds (float).
    Accepts:
      - SS
      - SS.mmm
      - MM:SS[.mmm]
      - HH:MM:SS[.mmm]
    """
    t = t.strip()
    if not t:
        raise ValueError("Empty timecode")

    if re.fullmatch(r"\d+(?:\.\d+)?", t):  # seconds (possibly fractional)
        return float(t)

    parts = t.split(":")
    if len(parts) == 2:  # MM:SS
        mm, ss = parts
        return int(mm) * 60 + float(ss)
    if len(parts) == 3:  # HH:MM:SS
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    raise ValueError(f"Unsupported time format: {t!r}")

def merge_ranges(ranges: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    """Merge overlapping/touching ranges and return sorted list."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le + eps:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def _read_csv_line(line: str) -> Tuple[float, float] | None:
    """
    Try to parse a CSV line with 2 or 3 columns:
      - start, end
      - label, start, end
    Returns (start_sec, end_sec) or None if not CSV-like.
    """
    # Fast check: if it doesn't contain a comma, it's not CSV for our purpose
    if "," not in line:
        return None

    # Use csv.reader to handle spaces properly (skipinitialspace=True)
    row = next(csv.reader([line], skipinitialspace=True))
    # Allow comments anywhere: drop trailing column if it starts with '#'
    if row and row[-1].lstrip().startswith("#"):
        row = row[:-1]

    if len(row) == 2:
        start_s, end_s = row
    elif len(row) == 3:
        # label, start, end
        _, start_s, end_s = row
    else:
        return None

    start = parse_timecode(start_s)
    end = parse_timecode(end_s)
    if end <= start:
        raise ValueError(f"end <= start in CSV line: {line.strip()!r}")
    return (start, end)

def read_ranges(txt_path: str) -> List[Tuple[float, float]]:
    """
    Read and parse time ranges from a text file in any of the supported formats.
    Ignores blank lines and lines starting with '#'.
    Returns merged, sorted (start_sec, end_sec) tuples.
    """
    ranges: List[Tuple[float, float]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # 1) Try CSV formats (with or without label)
            try:
                csv_tuple = _read_csv_line(line)
            except Exception as e:
                raise ValueError(f"{txt_path}:{line_no}: {e}") from e
            if csv_tuple is not None:
                ranges.append(csv_tuple)
                continue

            # 2) Fallback: "start-end" or "start,end" free-form
            m = RANGE_LINE.match(line)
            if not m:
                raise ValueError(f"{txt_path}:{line_no}: Cannot parse line: {line!r}")
            start = parse_timecode(m.group(1))
            end = parse_timecode(m.group(2))
            if end <= start:
                raise ValueError(f"{txt_path}:{line_no}: end <= start: {line!r}")
            ranges.append((start, end))

    return merge_ranges(ranges)

def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}):\n{shlex.join(cmd)}\n\nOutput:\n{proc.stdout}")

def has_audio(input_path: str) -> bool:
    """Check if at least one audio stream exists using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        input_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool(proc.stdout.strip())

def build_select_expr(ranges: List[Tuple[float, float]]) -> str:
    """ffmpeg select expression: between(t,s,e)+between(t,...)..."""
    return "+".join([f"between(t,{s:.6f},{e:.6f})" for s, e in ranges])

def crop_with_ffmpeg(input_path: str, output_path: str, ranges: List[Tuple[float, float]]) -> None:
    """
    Use ffmpeg select/aselect to keep only the given time ranges.
    Re-encodes (H.264 + AAC) for clean, keyframe-independent cuts.
    """
    expr = build_select_expr(ranges)
    filter_complex = f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[v]"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", output_path,
    ]

    run(cmd)

def find_targets(base_dir: str, ranges_filename: str = "Event_time_ranges.txt") -> List[Tuple[str, List[str]]]:
    """
    Return a list of tuples: (folder_path, [mp4_paths...]) for folders that
    contain the ranges file and at least one .mp4 file.
    """
    targets = []
    for root, _, files in os.walk(base_dir):
        names_lower = {f.lower() for f in files}
        if ranges_filename.lower() in names_lower:
            mp4s = [os.path.join(root, f) for f in files if f.lower().endswith(".mp4")]
            if mp4s:
                targets.append((root, mp4s))
    return targets

def main():
    ap = argparse.ArgumentParser(description="Crop MP4s by time ranges per folder.")
    ap.add_argument("--base", default=".", help="Base directory to scan (default: current dir).")
    ap.add_argument("--ranges-name", default="Event_time_ranges.txt",
                    help="Name of the time range file (default: event_time_ranges.txt).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip creating output if <name>_cropped.mp4 already exists.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base)
    print(f"Scanning: {base_dir}")

    targets = find_targets(base_dir, args.ranges_name)
    if not targets:
        print("No folders with time ranges and mp4 files found.")
        return

    for folder, mp4s in targets:
        ranges_path = os.path.join(folder, args.ranges_name)
        try:
            ranges = read_ranges(ranges_path)
            start, end = ranges[-1]             # unpack last (start, end)
            ranges[-1] = (start, end - 0.8)     # replace tuple with modified one
        except Exception as e:
            print(f"[ERROR] {ranges_path}: {e}")
            continue

        if not ranges:
            print(f"[WARN] No valid ranges in {ranges_path}; skipping folder.")
            continue

        print(f"\nFolder: {folder}")
        print(f"  Ranges ({len(ranges)} merged): " + ", ".join([f"{s:.3f}-{e:.3f}s" for s, e in ranges]))
        for mp4 in sorted(mp4s):
            base, ext = os.path.splitext(mp4)
            out = base + "_cropped" + ext
            if args.skip_existing and os.path.exists(out):
                print(f"  Skip (exists): {os.path.basename(out)}")
                continue
            print(f"  Cropping: {os.path.basename(mp4)} -> {os.path.basename(out)}")
            try:
                crop_with_ffmpeg(mp4, out, ranges)
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Map region code -> point label
REGION_TO_POINT = {
    "LU": "P1", "MU": "P2", "RU": "P3",
    "LM": "P4", "MM": "P5", "RM": "P6",
    "LD": "P7", "MD": "P8", "RD": "P9",
}

# Match *_coords folders
COORDS_DIR_RE = re.compile(r"^(RU|MU|LU|LD|MD|RD|MM|LM|RM)_coords$", re.IGNORECASE)
# Example filename: coords_LD_timestamp52.5.json
FILENAME_TS_RE = re.compile(r"_timestamp([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
# Accept prefixes like P0XX_80_bL / 120_bL / 180_bL / 80_3L / 120_3L / 180_3L
PREFIX_RE = re.compile(r"^(P0\d{2})_(80|120|180)(cm)?_(basicL|bL|3lights|3light|3L)$", re.IGNORECASE)


def find_prefix_folder(start_dir: str) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    while True:
        base = os.path.basename(cur)
        m = PREFIX_RE.match(base)
        if m:
            pnum, dist, _, light = m.groups()
            # Normalize
            dist_norm = dist  # drop cm automatically
            if light.lower() in ["basicl", "bl"]:
                light_norm = "bL"
            elif light.lower() in ["3lights", "3l"]:
                light_norm = "3L"
            else:
                light_norm = light
            return f"{pnum}_{dist_norm}_{light_norm}"
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def locate_interpolation_dir(coords_dir: str) -> str:
    abs_coords = os.path.abspath(coords_dir)
    opencv_dir = os.path.dirname(abs_coords)            # .../OpenCV
    parent_of_opencv = os.path.dirname(opencv_dir)      # .../P0XX_###_bL or ..._3L parent (e.g., P002_120cm_stat_3lights)
    interpolation_dir = os.path.join(parent_of_opencv, "interpolation")
    os.makedirs(interpolation_dir, exist_ok=True)
    return interpolation_dir


def parse_timestamp_from_filename(filename: str, fps: Optional[float]) -> Optional[float]:
    m = FILENAME_TS_RE.search(filename)
    if not m:
        return None
    raw = m.group(1)
    try:
        val = float(raw)
    except ValueError:
        return None

    if fps:
        # If fps provided, treat integer as frame index; fractional stays as seconds.
        if raw.isdigit():
            return val / fps
        return val
    return val


def extract_detection_for_point(data: Any, point_label: str) -> Optional[List[float]]:
    if isinstance(data, dict) and point_label in data and isinstance(data[point_label], (list, tuple)):
        xy = data[point_label]
        if len(xy) >= 2:
            return [float(xy[0]), float(xy[1])]

    if isinstance(data, dict) and "detections" in data:
        d = data["detections"]
        if isinstance(d, dict) and point_label in d and isinstance(d[point_label], (list, tuple)):
            xy = d[point_label]
            if len(xy) >= 2:
                return [float(xy[0]), float(xy[1])]

    candidates = None
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict) and "points" in data and isinstance(data["points"], list):
        candidates = data["points"]

    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            pid = item.get("pointNr") or item.get("id") or item.get("point") or item.get("label")
            if pid == point_label:
                if "detection" in item and isinstance(item["detection"], (list, tuple)) and len(item["detection"]) >= 2:
                    return [float(item["detection"][0]), float(item["detection"][1])]
                if "coords" in item and isinstance(item["coords"], (list, tuple)) and len(item["coords"]) >= 2:
                    return [float(item["coords"][0]), float(item["coords"][1])]
                if "x" in item and "y" in item:
                    return [float(item["x"]), float(item["y"])]
    return None


def process_coords_folder(coords_dir: str, fps: Optional[float]) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    base = os.path.basename(coords_dir)
    m = COORDS_DIR_RE.match(base)
    if not m:
        return ("P0XX_80_bL", {})

    region = m.group(1).upper()
    point_label = REGION_TO_POINT[region]

    prefix = find_prefix_folder(coords_dir) or "P0XX_80_bL"

    results: Dict[str, List[Dict[str, Any]]] = {region: []}

    for entry in os.listdir(coords_dir):
        if not entry.lower().endswith(".json"):
            continue
        fullpath = os.path.join(coords_dir, entry)
        if not os.path.isfile(fullpath):
            continue

        ts = parse_timestamp_from_filename(entry, fps)
        if ts is None:
            print(f"[WARN] Skipping (no timestamp): {fullpath}")
            continue

        try:
            with open(fullpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping (bad json): {fullpath} ({e})")
            continue

        det = extract_detection_for_point(data, point_label)
        if det is None or len(det) < 2:
            print(f"[WARN] Skipping (no {point_label} detection): {fullpath}")
            continue

        results[region].append({
            "timestamp": float(ts),
            "detection": [float(det[0]), float(det[1])],
            "pointNr": point_label,
        })

    for r in results:
        results[r].sort(key=lambda x: x["timestamp"])

    return (prefix, results)


def write_results(interpolation_dir: str, prefix: str, results_by_region: Dict[str, List[Dict[str, Any]]]) -> None:
    for region, records in results_by_region.items():
        if not records:
            continue
        out_name = f"{prefix}_{region}.json"
        out_path = os.path.join(interpolation_dir, out_name)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"[OK] Wrote {len(records)} records -> {out_path}")
        except Exception as e:
            print(f"[ERROR] Could not write {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect point detections from *_coords folders and export to an interpolation folder.")
    parser.add_argument(
        "root",
        help="Path to scan (e.g. the directory that contains the 'OpenCV' folder or any ancestor).")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="If provided, treat '_timestampN' in filenames as a FRAME index and convert seconds = N / fps. "
             "If omitted, '_timestampN' is treated as seconds already.")
    parser.add_argument(
        "--regions", nargs="*", default=None, choices=list(REGION_TO_POINT.keys()),
        help="Optional: restrict processing to these region codes (e.g. RU MU LD). Defaults to all.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    restrict_regions = set([r.upper() for r in (args.regions or REGION_TO_POINT.keys())])

    found_any = False
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        m = COORDS_DIR_RE.match(base)
        if not m:
            continue

        region = m.group(1).upper()
        if region not in restrict_regions:
            continue

        found_any = True
        prefix, results_by_region = process_coords_folder(dirpath, args.fps)
        if not results_by_region or not results_by_region.get(region):
            print(f"[INFO] No usable detections in: {dirpath}")
            continue

        interpolation_dir = locate_interpolation_dir(dirpath)
        write_results(interpolation_dir, prefix, results_by_region)

    if not found_any:
        print("[INFO] No '*_coords' folders found under:", root)


if __name__ == "__main__":
    main()

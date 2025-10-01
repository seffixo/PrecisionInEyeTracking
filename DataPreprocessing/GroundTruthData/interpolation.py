import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging
import shutil

"""
Interpolate detections at missing centisecond timestamps and save to gt_data.
Usage:
  python interpolate_detections.py --root "D:\\WorkingFolder_PythonD\\2Dto3D_Conversion\\521_stat_conv\\P002_statisch"
"""

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\581_dynam\interpolation.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)

DECIMALS = 8  # number formatting for detection values


def ts_to_centiseconds(ts_str: str) -> int:
    """
    Convert a timestamp string like '52.17' to total centiseconds (e.g., 5217).
    Assumes exactly two digits after the decimal point.
    """
    if "." not in ts_str:
        # If someone passed '52' without decimals, treat as '.00'
        sec = int(ts_str)
        cs = 0
    else:
        s_sec, s_cs = ts_str.split(".", 1)
        if len(s_cs) != 2 or not s_cs.isdigit():
            logging.error(f"Timestamp '{ts_str}' must have exactly two digits after the decimal.")
            raise ValueError(f"Timestamp '{ts_str}' must have exactly two digits after the decimal.")
        sec = int(s_sec)
        cs = int(s_cs)
    return sec * 100 + cs


def centiseconds_to_ts(cs_total: int) -> str:
    """Convert total centiseconds (e.g., 5217) back to 'SS.CC' string with 2 digits after decimal."""
    sec = cs_total // 100
    cs = cs_total % 100
    return f"{sec}.{cs:02d}"


def format_float_str(x: float) -> str:
    """Always return 8 decimal places, with last 4 digits as zeros."""
    return f"{round(x, 4):.4f}" + "0000"


def interpolate_segment(in_path: Path, p0: Dict, p1: Dict) -> List[Dict]:
    """
    Given two detection entries (consecutive in time), fill missing centisecond timestamps linearly.
    Returns list of ONLY the new, interpolated entries between p0 and p1 (excluding endpoints).
    """
    t0 = ts_to_centiseconds(p0["timestamp"])
    t1 = ts_to_centiseconds(p1["timestamp"])
    gap = (t1 - t0)

    if gap < 1:
        logging.error(f"Timestamps out of order or duplicate: {p0['timestamp']} -> {p1['timestamp']}")
        raise ValueError(f"Timestamps out of order or duplicate: {p0['timestamp']} -> {p1['timestamp']}")

    if gap > 5:
        logging.error(f"Timestamp differences: {gap}! {in_path.parent.parent.name}\{in_path.name}, t0 {t0}, t1 {t1}")

    # Parse detection floats
    x0, y0 = map(float, p0["detection"])
    x1, y1 = map(float, p1["detection"])

    new_points: List[Dict] = []
    total_steps = gap  # number of 0.01-second steps from t0 to t1
    # For each missing centisecond (exclude endpoints)
    for step in range(1, gap):
        t = t0 + step
        # fraction across the segment
        frac = step / total_steps
        x = x0 + frac * (x1 - x0)
        y = y0 + frac * (y1 - y0)

        new_points.append({
            "timestamp": centiseconds_to_ts(t),
            "detection": [format_float_str(x), format_float_str(y)],
            "pointNr": p0.get("pointNr", p1.get("pointNr"))  # keep same point id
        })

    return new_points


def process_file(in_path: Path, out_path: Path) -> None:
    """Read one JSON file, interpolate, and write result to out_path."""
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logging.error(f"{in_path} does not contain a list of detections.")
        raise ValueError(f"{in_path} does not contain a list of detections.")

    # Sort by timestamp (just in case)
    try:
        data_sorted = sorted(data, key=lambda d: ts_to_centiseconds(str(d["timestamp"])))
    except Exception as e:
        logging.error(f"Error sorting {in_path}: {e}")
        raise ValueError(f"Error sorting {in_path}: {e}")

    # Build new list including original + interpolated
    upscaled: List[Dict] = []
    for i in range(len(data_sorted) - 1):
        curr = data_sorted[i]
        nxt = data_sorted[i + 1]
        upscaled.append(curr)  # always include the current point
        # add in-between points
        upscaled.extend(interpolate_segment(in_path, curr, nxt))
    # include last original point
    if data_sorted:
        upscaled.append(data_sorted[-1])

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(upscaled, f, ensure_ascii=False, indent=2)


def find_interpolation_dirs(root: Path) -> List[Path]:
    """Recursively find all directories named exactly 'interpolation' under root."""
    return [p for p in root.rglob("interpolation") if p.is_dir()]


def main():
    parser = argparse.ArgumentParser(description="Upscale detection JSON by centisecond linear interpolation.")
    parser.add_argument("--root", required=True, help="Root directory to search (recursively) for 'interpolation' folders.")
    parser.add_argument("--ext", default=".json", help="File extension to process (default: .json)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    interp_dirs = find_interpolation_dirs(root)
    if not interp_dirs:
        print(f"No 'interpolation' folders found under: {root}")
        return

    for interp_dir in interp_dirs:
        out_dir = interp_dir / "gt_data"
        # Remove gt_data if it exists, so we always start clean
        if out_dir.exists():
            shutil.rmtree(out_dir)
            logging.info(f"removed previous existing gt_data folder in {out_dir.parent.parent.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Now process every file in the interpolation folder
        for in_file in interp_dir.glob(f"*{args.ext}"):
            out_file = out_dir / in_file.name
            try:
                process_file(in_file, out_file)
                in_file_name = in_file.name
                out_file_name = out_file.name
                logging.info(f"finished {in_file_name}")
            except Exception as e:
                logging.error(f"Failed: {in_file}  ({e})")


if __name__ == "__main__":
    main()

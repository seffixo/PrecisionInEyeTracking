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
PREFIX_RE = re.compile(r"^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$",re.IGNORECASE)



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


def parse_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    Extract the number after '_timestamp' from filename.
    The timestamp value is already in seconds.milliseconds, so no conversion is done.
    """
    m = FILENAME_TS_RE.search(filename)
    if not m:
        return None
    raw = m.group(1)
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return None


def extract_detection_for_point(data: Any, point_label: str) -> Optional[List[str]]:
    if isinstance(data, dict) and point_label in data and isinstance(data[point_label], (list, tuple)):
        xy = data[point_label]
        if len(xy) >= 2:
            return [f"{float(xy[0]):.8f}", f"{float(xy[1]):.8f}"]

    # other options where json file could be structured differently
    if isinstance(data, dict) and "detections" in data:
        d = data["detections"]
        if isinstance(d, dict) and point_label in d and isinstance(d[point_label], (list, tuple)):
            xy = d[point_label]
            if len(xy) >= 2:
                return [f"{float(xy[0]):.8f}", f"{float(xy[1]):.8f}"]

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
                    return [f"{float(item['detection'][0]):.8f}", f"{float(item['detection'][1]):.8f}"]
                if "coords" in item and isinstance(item["coords"], (list, tuple)) and len(item["coords"]) >= 2:
                    return [f"{float(item['coords'][0]):.8f}", f"{float(item['coords'][1]):.8f}"]
                if "x" in item and "y" in item:
                    return [f"{float(item['x']):.8f}", f"{float(item['y']):.8f}"]
    return None


def process_coords_folder(coords_dir: str) -> Tuple[Optional[str], Dict[str, List[Dict[str, Any]]]]:
    base = os.path.basename(coords_dir)
    m = COORDS_DIR_RE.match(base)
    if not m:
        print(f"[INFO] Skipping: {coords_dir} (not a *_coords folder)")
        return (None, {})

    region = m.group(1).upper()
    point_label = REGION_TO_POINT[region]

    prefix = find_prefix_folder(coords_dir)
    if not prefix:
        print(f"[WARN] No valid P0XX_###_(bL|3L) prefix found for {coords_dir}")
        return (None, {})

    results: Dict[str, List[Dict[str, Any]]] = {region: []}

    for entry in os.listdir(coords_dir):
        if not entry.lower().endswith(".json"):
            continue
        fullpath = os.path.join(coords_dir, entry)
        if not os.path.isfile(fullpath):
            continue

        ts = parse_timestamp_from_filename(entry)
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
            "timestamp": ts,
            "detection": det,
            "pointNr": point_label,
        })

    for r in results:
        results[r].sort(key=lambda x: x["timestamp"])

    return (prefix, results)

def check_outliers(results_by_region: Dict[str, List[Dict[str, Any]]], prefix: str,
                   factor: float = 1.5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Check for outliers using IQR and ask the user whether to remove each one.
    Returns a possibly modified results_by_region (in-place edit).
    """
    for region, records in results_by_region.items():
        if not records:
            continue

        # Parse detection strings -> floats for checking
        try:
            xs = [float(r["detection"][0]) for r in records]
            ys = [float(r["detection"][1]) for r in records]
        except Exception as e:
            print(f"[WARN] {region}: Cannot parse detection values ({e}). Skipping outlier check.")
            continue

        if len(records) < 5:
            print(f"[INFO] {region}: Too few records ({len(records)}) for robust outlier check.")
            continue

        # Calculate IQR bounds
        def percentile(vals, p):
            vals_sorted = sorted(vals)
            k = (len(vals_sorted) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(vals_sorted) - 1)
            if f == c:
                return vals_sorted[f]
            return vals_sorted[f] * (c - k) + vals_sorted[c] * (k - f)

        def iqr_bounds(vals, factor):
            q1 = percentile(vals, 25)
            q3 = percentile(vals, 75)
            iqr = q3 - q1
            return q1 - factor * iqr, q3 + factor * iqr

        x_lo, x_hi = iqr_bounds(xs, factor)
        y_lo, y_hi = iqr_bounds(ys, factor)

        # Check each record
        to_remove = []
        for idx, rec in enumerate(records):
            x = xs[idx]
            y = ys[idx]
            if x < x_lo or x > x_hi or y < y_lo or y > y_hi:
                print(f"\n[OUTLIER] Region: {region}, Folder: {prefix}")
                print(f" Timestamp: {rec['timestamp']}")
                print(f" Detection: x: {x}, y: {y}")
                print(f" Bounds: x_lo: {x_lo}, x_hi: {x_hi}")
                print(f" Bounds: y_lo: {y_lo}, y_hi: {y_hi}")
                choice = input(" Remove this outlier? (y/n): ").strip().lower()
                if choice == "y":
                    to_remove.append(idx)

        # Remove chosen outliers (in reverse order so indexing is safe)
        for idx in reversed(to_remove):
            removed = records.pop(idx)
            print(f"[INFO] Removed outlier: Timestamp {removed['timestamp']} Detection {removed['detection']}")

    return results_by_region

def check_outliers_hampel(results_by_region: Dict[str, List[Dict[str, Any]]],
                          prefix: str,
                          window: int = 21,      # odd window size is best
                          k: float = 3.0,        # sensitivity (higher = fewer flags)
                          deadband: float = 0.0005,  # ignore tiny diffs
                          require_both: bool = False  # flag only if BOTH x and y are out
                          ) -> Dict[str, List[Dict[str, Any]]]:
    """
    Rolling Hampel outlier check (per region), interactive removal.
    Uses local median ± k * (1.4826 * MAD) with a deadband.
    Coordinates are parsed to floats for checking; records stay in their original string format.
    """
    def median(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return float("nan")
        mid = n // 2
        if n % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def mad(vals, med):
        return median([abs(v - med) for v in vals])

    # ensure odd window
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2

    for region, records in results_by_region.items():
        if not records:
            continue
        n = len(records)
        if n < 5:
            print(f"[INFO] {region}: Too few records ({n}) for robust local check.")
            continue

        # parse floats
        xs = []
        ys = []
        for r in records:
            try:
                xs.append(float(r["detection"][0]))
                ys.append(float(r["detection"][1]))
            except Exception:
                xs.append(float("nan"))
                ys.append(float("nan"))

        to_remove = []
        for i in range(n):
            # window bounds
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            # exclude point itself for robustness
            win_x = xs[lo:i] + xs[i+1:hi]
            win_y = ys[lo:i] + ys[i+1:hi]
            # need enough neighbors
            if len(win_x) < max(4, window // 2):
                continue
            # remove NaNs
            wx = [v for v in win_x if v == v]
            wy = [v for v in win_y if v == v]
            if len(wx) < 4 or len(wy) < 4:
                continue

            med_x = median(wx)
            med_y = median(wy)
            mad_x = mad(wx, med_x)
            mad_y = mad(wy, med_y)
            # scaled MAD ~ robust std
            sigma_x = 1.4826 * mad_x if mad_x > 0 else 0.0
            sigma_y = 1.4826 * mad_y if mad_y > 0 else 0.0

            # if local variation is basically zero, skip flagging unless very large diff
            thr_x = k * sigma_x + deadband
            thr_y = k * sigma_y + deadband

            dx = abs(xs[i] - med_x)
            dy = abs(ys[i] - med_y)

            flag_x = sigma_x == 0.0 and dx > deadband or (sigma_x > 0.0 and dx > thr_x)
            flag_y = sigma_y == 0.0 and dy > deadband or (sigma_y > 0.0 and dy > thr_y)

            flag = (flag_x and flag_y) if require_both else (flag_x or flag_y)
            if not flag:
                continue

            # interactive prompt
            rec = records[i]
            print(f"\n[OUTLIER] Region: {region}, Folder: {prefix}")
            print(f" Timestamp: {rec['timestamp']}")
            print(f" Detection: x: {rec['detection'][0]}, y: {rec['detection'][1]}")
            # show local stats (rounded for readability)
            print(f" Local med ± thr  (x): {med_x:.6f} ± {thr_x:.6f}  (dx={dx:.6f})")
            print(f" Local med ± thr  (y): {med_y:.6f} ± {thr_y:.6f}  (dy={dy:.6f})")
            choice = input(" Remove this outlier? (y/n): ").strip().lower()
            if choice == "y":
                to_remove.append(i)

        for idx in reversed(to_remove):
            removed = records.pop(idx)
            print(f"[INFO] Removed outlier: ts={removed['timestamp']} det={removed['detection']}")

    return results_by_region


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
        "--root",
        help="Path to scan (e.g. the directory that contains the 'OpenCV' folder or any ancestor).")
    parser.add_argument(
        "--regions", nargs="*", default=None, choices=list(REGION_TO_POINT.keys()),
        help="Optional: restrict processing to these region codes (e.g. RU MU LD). Defaults to all.")
    args =parser.parse_args()

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
        prefix, results_by_region = process_coords_folder(dirpath)
        if not results_by_region or not results_by_region.get(region):
            print(f"[INFO] No usable detections in: {dirpath}")
            continue
        
        #results_by_region = check_outliers(results_by_region, prefix, factor=1.5)
        results_by_region = check_outliers_hampel(
            results_by_region,
            prefix=prefix,
            window=31,       # try 21, 31, or 41 depending on your sampling rate
            k=3.0,           # increase to 3.5–4.5 if it still feels too sensitive
            deadband=0.01, # widen if you see many near-threshold flags (e.g., 0.001)
            require_both=False  # set True to only flag when BOTH x and y deviate
        )
        interpolation_dir = locate_interpolation_dir(dirpath)
        write_results(interpolation_dir, prefix, results_by_region)

    if not found_any:
        print("[INFO] No '*_coords' folders found under:", root)


if __name__ == "__main__":
    main()

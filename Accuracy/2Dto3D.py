import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import csv
from decimal import Decimal
from typing import Iterable, Optional, Tuple
import logging

'''
Always check for correct camera parameters!
'''

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\521_dynam\normalized_gazedata.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)
# -------- camera parameters (581) --------
#K = np.array([
#       [912.7711395670913, 0.0, 958.5026077976856],
#        [0.0, 912.172924797333, 515.7067710093856],
#        [0.0, 0.0, 1.0]
#], dtype=np.float32)
#
#dist = np.array(
#        [-0.043006050416685725, 0.05772694058191421, 2.5235099153145128e-05,
#        0.0004967988726047721, -0.03825636453680942],
#    dtype=np.float32
#)

# -------- camera parameters (521) --------
K = np.array([
        [911.7661807262908, 0.0, 953.1539301425507],
        [0.0, 911.2665673922147, 515.5778687592352],
        [0.0, 0.0, 1.0]
], dtype=np.float32)

dist = np.array(
        [-0.04277252073338572, 0.061178714113208424, -2.0962433109889865e-06,
        0.0002723710640655781, -0.040889510305845714],
    dtype=np.float32
)

def load_norm01_from_json(json_path: Path):
    """Load detections from JSON with format:
       [{ "timestamp": str, "detection": [x01, y01], "pointNr": str }, ...]
       Returns list of dicts: {timestamp, x01, y01, pointNr}
    """
    data = json.loads(json_path.read_text())
    out = []
    for row in data:
        x = float(row["detection"][0])  # 0..1
        y = float(row["detection"][1])  # 0..1
        ts = str(row.get("timestamp", ""))  # keep original string
        point_nr = str(row.get("pointNr", ""))
        out.append({"timestamp": ts, "x01": x, "y01": y, "pointNr": point_nr})
    return out


def _parse_float_safe(d, key, default=None):
    if key in d and d[key] not in (None, ""):
        try:
            return float(d[key])
        except Exception:
            pass
    return default

def load_norm01_from_csv(csv_path: Path, width: int, height: int):
    """Load detections from CSV. Supports:
       - normalized: x01, y01 OR gaze2d_x, gaze2d_y
       - pixel: u_px, v_px (scaled by width/height)
       Optional: timestamp, pointNr
    """
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]

        # mapping for common variants
        colmap = {
            "x01": "x01",
            "y01": "y01",
            "gaze2d_x": "x01",
            "gaze2d_y": "y01",
            "u_px": "u_px",
            "v_px": "v_px",
        }

        has_norm = any(c in fieldnames for c in ["x01", "gaze2d_x"]) and \
                   any(c in fieldnames for c in ["y01", "gaze2d_y"])
        has_pixels = ("u_px" in fieldnames and "v_px" in fieldnames)

        if not (has_norm or has_pixels):
            raise ValueError(
                f"{csv_path}: CSV must contain x01,y01 OR gaze2d_x,gaze2d_y OR u_px,v_px columns."
            )

        for row in reader:
            ts = str(row.get("timestamp", "")) if "timestamp" in row else ""

            if has_norm:
                xcol = "x01" if "x01" in row else "gaze2d_x"
                ycol = "y01" if "y01" in row else "gaze2d_y"
                x01 = _parse_float_safe(row, xcol)
                y01 = _parse_float_safe(row, ycol)
            else:
                u = _parse_float_safe(row, "u_px")
                v = _parse_float_safe(row, "v_px")
                x01 = u / float(width)
                y01 = v / float(height)

            out.append({
                "timestamp": ts,
                "x01": float(x01),
                "y01": float(y01)
            })
    return out

def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def convert_2d_list_to_3d_rows(dets, width: int, height: int, clip01: bool):
    """Common 2D->3D path shared by JSON and CSV loaders."""
    x01 = np.array([d["x01"] for d in dets], dtype=np.float32)
    y01 = np.array([d["y01"] for d in dets], dtype=np.float32)
    if clip01:
        x01 = np.clip(x01, 0.0, 1.0)
        y01 = np.clip(y01, 0.0, 1.0)

    u = x01 * width
    v = y01 * height

    # Prepare for OpenCV undistortPoints: shape (N,1,2)
    pts_pix = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.float32)

    # Undistort + remove intrinsics -> normalized pinhole coords (x', y'), z=1
    normed = cv2.undistortPoints(pts_pix, K, dist, P=None).reshape(-1, 2)  # (N,2)

    # Build camera-frame rays and normalize to unit length (x^2 + y^2 + z^2 = 1)
    rays_cam = np.hstack([normed, np.ones((normed.shape[0], 1), dtype=np.float32)])  # [x', y', 1]
    rays_cam = normalize_rows(rays_cam)

    # Stitch output rows
    rows = []
    for i, d in enumerate(dets):
        if "pointNr" in d:
            rows.append({
                "timestamp": d["timestamp"],
                "pointNr": d["pointNr"],
                "x01": float(x01[i]),
                "y01": float(y01[i]),
                "u_px": float(u[i]),
                "v_px": float(v[i]),
                "dir_cam_x": float(rays_cam[i, 0]),
                "dir_cam_y": float(rays_cam[i, 1]),
                "dir_cam_z": float(rays_cam[i, 2]),
            })
        else: 
            rows.append({
                "timestamp": d["timestamp"],
                "x01": float(x01[i]),
                "y01": float(y01[i]),
                "u_px": float(u[i]),
                "v_px": float(v[i]),
                "dir_cam_x": float(rays_cam[i, 0]),
                "dir_cam_y": float(rays_cam[i, 1]),
                "dir_cam_z": float(rays_cam[i, 2]),
            })

    return rows

def write_output_csv(out_csv_path: Path, rows):
    has_pointnr = any("pointNr" in r and r["pointNr"] != "" for r in rows)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        if has_pointnr:
            f.write("timestamp,pointNr,x_norm,y_norm,u_px,v_px,dir_cam_x,dir_cam_y,dir_cam_z\n")
        else: 
            f.write("timestamp,x_norm,y_norm,u_px,v_px,dir_cam_x,dir_cam_y,dir_cam_z\n")
        for r in rows:
            x_norm_str = f"{round(r['x01'], 4):.8f}"
            y_norm_str = f"{round(r['y01'], 4):.8f}"

            if has_pointnr: 
                if "pointNr" in r:
                    f.write(
                    f"{r['timestamp']},{r['pointNr']},"
                    f"{x_norm_str},{y_norm_str},"
                    f"{r['u_px']:.3f},{r['v_px']:.3f},"
                    f"{r['dir_cam_x']:.9f},{r['dir_cam_y']:.9f},{r['dir_cam_z']:.9f}\n"
                )
            else:
                f.write(
                f"{r['timestamp']},"
                f"{x_norm_str},{y_norm_str},"
                f"{r['u_px']:.3f},{r['v_px']:.3f},"
                f"{r['dir_cam_x']:.9f},{r['dir_cam_y']:.9f},{r['dir_cam_z']:.9f}\n"
            )

def find_data_folder(root: Path, input_type: str):
    """Yield files under any 'str_to_find' dir matching the desired input_type."""
    exts = []
    if input_type == "json":
        exts = [".json"]
        str_to_find = "gt_data"
    elif input_type == "csv":
        exts = [".csv"]
        str_to_find = "filtered_gazedata"
    elif input_type == "auto":
        exts = [".json", ".csv"]
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    for data_dir in root.rglob(str_to_find):
        if not data_dir.is_dir():
            continue
        for ext in exts:
            yield from sorted(data_dir.glob(f"*{ext}"))

def process_one_file_generic(path: Path, out_csv_path: Path, width: int, height: int, clip01: bool):
    """Dispatch loader based on extension and run the common converter."""
    if path.suffix.lower() == ".json":
        dets = load_norm01_from_json(path)
    elif path.suffix.lower() == ".csv":
        dets = load_norm01_from_csv(path, width=width, height=height)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    rows = convert_2d_list_to_3d_rows(dets, width=width, height=height, clip01=clip01)
    write_output_csv(out_csv_path, rows)
    print(f"finished ")

def main():
    ap = argparse.ArgumentParser(description="Convert 0–1 or pixel image coords to camera-frame unit rays.")
    ap.add_argument("--root", required=True,
                    help="Root directory to search for gt_data folders (kept name for backward compatibility).")
    ap.add_argument("--input_type", choices=["json", "csv", "auto"], default="auto",
                    help="Type of input files to process. 'auto' looks for both.")
    ap.add_argument("--width", type=int, default=1920, help="Image width in pixels. 1920")
    ap.add_argument("--height", type=int, default=1080, help="Image height in pixels. 1080")
    ap.add_argument("--clip01", action="store_true",
                    help="Clip normalized inputs to [0,1] before converting to pixels.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root}")

    files = list(find_data_folder(root, args.input_type))
    if not files:
        logging.error(f"No data/*.{args.input_type if args.input_type!='auto' else '{json,csv}'} files found under: {root}")
        print(f"No data/* files found under: {root}")
        return

    if "json" in args.input_type:
        new_folder_name = "normalized_gt"
    elif "csv" in args.input_type: 
        new_folder_name = "normalized_gazedata"

    total = len(files)
    for idx, p in enumerate(files, start=1):
        gt_dir = p.parent
        out_dir = gt_dir / new_folder_name
        out_csv = out_dir / (p.stem + ".csv")
        try:
            process_one_file_generic(p, out_csv, args.width, args.height, args.clip01)
            logging.info(f"Saved file {p.name} in {gt_dir.parent.parent.name}")
        except Exception as e:
            logging.error(f"[{idx}/{total}] failed to process file: {p}, {e}")

if __name__ == "__main__":
    main()

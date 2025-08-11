import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import logging

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\581_stat_conv\normalized_gt.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)

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

def load_norm01(json_path: Path):
    data = json.loads(json_path.read_text())
    out = []
    for row in data:
        x = float(row["detection"][0])  # 0..1
        y = float(row["detection"][1])  # 0..1
        ts = str(row.get("timestamp", ""))  # keep original string
        point_nr = str(row.get("pointNr", ""))
        out.append({"timestamp": ts, "x01": x, "y01": y, "pointNr": point_nr})
    return out

def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def process_one_file(json_path: Path, out_csv_path: Path, width: int, height: int, clip01: bool):
    dets = load_norm01(json_path)

    # (x01,y01) -> pixel (u,v)
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

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        f.write("timestamp,pointNr,x_norm,y_norm,u_px,v_px,dir_cam_x,dir_cam_y,dir_cam_z\n")
        for i, d in enumerate(dets):
            f.write(
                f"{d['timestamp']},{d['pointNr']},"
                f"{x01[i]:.8f},{y01[i]:.8f},"
                f"{u[i]:.3f},{v[i]:.3f},"
                f"{rays_cam[i,0]:.9f},{rays_cam[i,1]:.9f},{rays_cam[i,2]:.9f}\n"
            )

def find_jsons_under_gt_data(root: Path):
    # Find every folder named "gt_data" under root, then all *.json directly inside it
    for gt_dir in root.rglob("gt_data"):
        if gt_dir.is_dir():
            yield from sorted(gt_dir.glob("*.json"))

def main():
    ap = argparse.ArgumentParser(description="Convert 0–1 image coords to camera-frame unit rays.")
    ap.add_argument("--json", required=True, help="Path to detections JSON (list of {timestamp, detection:[x,y], pointNr}).")
    ap.add_argument("--width", type=int, default=1920, help="Image width in pixels. 1920")
    ap.add_argument("--height", type=int, default=1080, help="Image height in pixels. 1080")
    ap.add_argument("--clip01", action="store_true", help="Clip inputs to [0,1] before converting to pixels.")
    args = ap.parse_args()

    root = Path(args.json)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root}")

    json_files = list(find_jsons_under_gt_data(root))
    if not json_files:
        logging.error(f"No gt_data/*.json files found under: {root}")
        print(f"No gt_data/*.json files found under: {root}")
        return

    total = len(json_files)
    #print(f"Found {total} JSON files under gt_data folders.")
    for idx, jf in enumerate(json_files, start=1):
        gt_dir = jf.parent
        out_dir = gt_dir / "normalized_gt"
        out_csv = out_dir / (jf.stem + ".csv")
        #print(f"[{idx}/{total}] {jf}")
        try:
            process_one_file(jf, out_csv, args.width, args.height, args.clip01)
            logging.info(f"saved file {jf.name} in {gt_dir.parent.parent.name}")
        except Exception as e:
            logging.error(f"failed to process file: {jf}, {e}")

if __name__ == "__main__":
    main()

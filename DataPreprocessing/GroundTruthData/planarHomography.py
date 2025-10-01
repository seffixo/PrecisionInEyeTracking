import cv2
import numpy as np
import os
import argparse
import sys
import json
from pathlib import Path
import logging
from math import sqrt
from collections import namedtuple

'''
dot finder for dynamic recordings with planar homography rectification:

- Keeps your folder handling, participant selection (to_do_list), drawing, and JSON outputs.
- NEW: per-frame homography (frame->reference), rectify, run matchTemplate in reference space,
  then back-project match center into the original frame coordinates for saving/drawing.

Requirements per subfolder:
  - poster_reference.(png|jpg|jpeg)  # fronto-parallel poster image
  - template1..template4.(png|jpg|jpeg)  # crops in reference space (fronto-parallel)
  - image_processing/1_interval...4_interval # frames live here (unchanged)

Done: 581_dynam: P002, 

'''

# -------------- Setup logging --------------
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\521_dynam\dotfinder_dynam.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------- Config --------------
to_do_list = ["P004", "P005", "P006", "P007", "P008", "P009", "P011", "P012", "P013", "P014", "P015", "P016", "P017", "P033"]
image_endings = (".png", ".jpg", ".jpeg")

# -------------- Utilities --------------
def save_image_and_json(selected_detected, frame, output_path, image_filename, img_int_path):
    for key, ((x,y), (normx, normy)) in selected_detected.items():
        cv2.line(frame, (x - 5, y), (x + 5, y), (0, 255, 0), 1)
        cv2.line(frame, (x, y - 5), (x, y + 5), (0, 255, 0), 1)
        coords = f"{key} {round(normx,4)},{round(normy,4)}"
        cv2.putText(frame, text=coords, org=(x-50, y-15), fontScale=1, fontFace=2, color=(0,255,0), thickness=2)

    # Save image
    output_images = os.path.join(output_path, "images")
    os.makedirs(output_images, exist_ok=True)
    interval_name = Path(img_int_path).name
    interval_path = os.path.join(output_images, interval_name)
    os.makedirs(interval_path, exist_ok=True)
    output_image_path = os.path.join(interval_path, image_filename)
    cv2.imwrite(output_image_path, frame)

    # Save coords
    subfolder_output = os.path.join(output_path, "MM_coords")
    os.makedirs(subfolder_output, exist_ok=True)
    img_name = os.path.splitext(image_filename)[0]
    output_text_path = os.path.join(subfolder_output, f"coords_{img_name}.json")

    formatted_detections = {}
    for key, ((x,y), (normx, normy)) in selected_detected.items():
        x_str = f"{normx:.4f}0000"
        y_str = f"{normy:.4f}0000"
        formatted_detections[key] = [x_str, y_str]

    with open(output_text_path, "w") as f:
        json.dump(formatted_detections, f, indent=2)

def find_template_files(folder_path):
    intervals = ["1", "2", "3", "4"]
    candidates = {}
    for ext in image_endings:
        for i in intervals:
            candidate = os.path.join(folder_path, f"template{i}{ext}")
            if os.path.isfile(candidate):
                candidates[i] = candidate
    return candidates

def find_img_proc_paths(folder_path):
    intervals = ["1", "2", "3", "4"]
    img_folders = {}
    for i in intervals:
        folder = os.path.join(folder_path, "image_processing", f"{i}_interval")
        if os.path.exists(folder):
            img_folders[i] = folder
    return img_folders

def find_reference_poster(folder_path):
    for name in ["poster_reference.png", "poster_reference.jpg", "poster_reference.jpeg"]:
        candidate = os.path.join(folder_path, name)
        if os.path.isfile(candidate):
            return candidate
    return None

def delete_detections(key_list, detections):
    key_list.sort(key=lambda x: int(x[1:]), reverse=True)
    filtered_detections = [value for key, value in detections.items() if key not in key_list]
    reindexed = {f"P{idx+1}": value for idx, value in enumerate(filtered_detections)}
    return reindexed

def remove_close_points(detections, min_distance=10):
    removed_keys = set()
    det_items = list(detections.items())
    for i, (key1, ((x1, y1), _)) in enumerate(det_items):
        if key1 in removed_keys:
            continue
        for j in range(i + 1, len(det_items)):
            key2, ((x2, y2), _) = det_items[j]
            if key2 in removed_keys:
                continue
            dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            if dist < min_distance:
                removed_keys.add(key2)
                print(f"Removed duplicate point {key2} (too close to {key1}, dist={dist:.2f})")
    cleaned = {k: v for k, v in detections.items() if k not in removed_keys}
    return cleaned

def filter_detections(detections, w_frame, h_frame, img_processing_path, edgecase):
    current_path = Path(img_processing_path)
    distance = current_path.parent.parent.name
    interval = current_path.name
    filtered_detections = {}

    cleaned_detections = remove_close_points(detections, min_distance=10)
    return cleaned_detections

# -------------- Homography helpers --------------
def init_feature_model(ref_gray):
    # ORB is fast & free; bump features for robustness
    orb = cv2.ORB_create(nfeatures=3000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return orb, bf, kp_ref, des_ref

def estimate_h_fr2ref(frame_gray, ref_gray, orb, bf, kp_ref, des_ref, ratio=0.75, ransac_thresh=3.0):
    kp_fr, des_fr = orb.detectAndCompute(frame_gray, None)
    if des_fr is None or len(kp_fr) < 10 or des_ref is None or len(kp_ref) < 10:
        return None, None, None
    knn = bf.knnMatch(des_ref, des_fr, k=2)  # ref -> frame
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    if len(good) < 10:
        return None, None, None

    src_pts = np.float32([kp_ref[i].pt for i, _ in good]).reshape(-1,1,2)  # reference
    dst_pts = np.float32([kp_fr[j].pt for _, j in good]).reshape(-1,1,2)   # frame
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)  # frame->ref
    if H is None:
        return None, None, None
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers, len(good)

def perspective_points(pts, H):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1,2)
    pts_h = np.hstack([pts, np.ones((len(pts),1), dtype=np.float32)])
    proj = (H @ pts_h.T).T
    proj /= proj[:, [2]]
    return proj[:, :2]

# -------------- Matching --------------
def match_in_rectified_multi(frame_bgr, template_gray, ref_gray, feature_ctx,
                             peak_thresh, max_peaks, min_inliers):
    """
    Find multiple template matches in the rectified (reference) view and
    back-project their centers to the original frame.

    Returns:
      centers_fr: list[(x_int, y_int)]           # centers in original frame coords
      scores:     list[float]                    # TM_CCOEFF_NORMED scores
      H:          3x3 homography (frame->ref) or None
      rectified:  rectified grayscale image or None
    """
    orb, bf, kp_ref, des_ref = feature_ctx
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h_ref, w_ref = ref_gray.shape[:2]

    # --- estimate homography (frame -> reference) ---
    H, inliers, matched = estimate_h_fr2ref(frame_gray, ref_gray, orb, bf, kp_ref, des_ref)
    if H is None or (inliers is not None and inliers < min_inliers):
        return [], [], None, None

    rectified = cv2.warpPerspective(frame_gray, H, (w_ref, h_ref))

    # --- template matching in reference space ---
    h_t, w_t = template_gray.shape[:2]
    res = cv2.matchTemplate(rectified, template_gray, cv2.TM_CCOEFF_NORMED)

    # --- local maxima above threshold (fast NMS with dilation) ---
    # Use a kernel about the template size so we pick one peak per template-sized area.
    k_h = max(1, h_t)
    k_w = max(1, w_t)
    kernel = np.ones((k_h, k_w), np.uint8)

    res_dil = cv2.dilate(res, kernel)
    peaks_mask = (res >= peak_thresh) & (res == res_dil)

    ys, xs = np.where(peaks_mask)
    if len(xs) == 0:
        return [], [], H, rectified

    # sort by score desc and keep top max_peaks
    scores = res[ys, xs]
    order = np.argsort(-scores)
    xs = xs[order][:max_peaks]
    ys = ys[order][:max_peaks]
    scores = scores[order][:max_peaks]

    # centers in reference space
    centers_ref = np.stack([xs + w_t // 2, ys + h_t // 2], axis=1).astype(np.float32)

    # back-project all to frame space
    H_ref2fr = np.linalg.inv(H)
    centers_fr = perspective_points(centers_ref, H_ref2fr)  # Nx2 float
    centers_fr = [(int(round(x)), int(round(y))) for x, y in centers_fr]

    return centers_fr, scores.tolist(), H, rectified

def load_targets_ref(folder_path):
    """Load 9 target points (pixels) from poster_targets.json in the subfolder."""
    p = os.path.join(folder_path, "poster_targets.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r") as f:
        data = json.load(f)
    # dict: "P1"->(x,y) ... cast to float
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}

def project_targets_frame(targets_ref, H_ref2fr, w_frame, h_frame):
    """Project known reference targets to frame coords and clamp to image."""
    in_bounds = {}
    oob = []
    raw_proj = {}

    labels = list(targets_ref.keys())
    pts = np.array([targets_ref[k] for k in labels], dtype=np.float32)
    pts_fr = perspective_points(pts, H_ref2fr)  # Nx2

    for label, (xf, yf) in zip(labels, pts_fr):
        raw_proj[label] = (float(xf), float(yf))
        if 0 <= xf < w_frame and 0 <= yf < h_frame:
            xi = int(round(xf)); yi = int(round(yf))
            in_bounds[label] = (xi, yi)
        else:
            xi = int(round(xf)); yi = int(round(yf))
            detection = (label, xi, yi)
            oob.append(detection)

    return in_bounds, oob, raw_proj

def match_guided_project_only(image_path, frame_bgr, ref_gray, feature_ctx, targets_ref, min_inliers=10):
    """
    Estimate H (frame->ref) and directly project the known 9 targets to frame.
    Returns: centers_fr_dict (label->(x,y)), H, rectified(None)
    """
    MatchResult = namedtuple("MatchResult", "centers oob raw_proj H rectified")
    orb, bf, kp_ref, des_ref = feature_ctx
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    H, inliers, matched = estimate_h_fr2ref(frame_gray, ref_gray, orb, bf, kp_ref, des_ref)
    if H is None or (inliers is not None and inliers < min_inliers):
        return MatchResult({}, [], {}, None, None)

    H_ref2fr = np.linalg.inv(H)
    h_frame, w_frame = frame_gray.shape[:2]
    in_bounds, oob, raw_proj = project_targets_frame(targets_ref, H_ref2fr, w_frame, h_frame)  # dict label->(x,y)

    return MatchResult(in_bounds, oob, raw_proj, H, None)


# -------------- Your original detection wrapper (now rectified-first) --------------
def find_marker_positions(image_path, template_path, img_path, threshold, image_filename, edgecase,
                          feature_ctx, ref_gray, targets_ref):
    
    Result = namedtuple("Result", "selected_detected, frame, image_filename, oob_detections, raw_proj")
    base = Path(img_path).parent.parent.parent.parent
    not_used_images_path = os.path.join(base, "unused_images.txt")
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error((f"Bild nicht gefunden: {image_path}"))
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = gray_frame.shape

    # homography + direct projection (Option 1)
    result = match_guided_project_only(image_path, frame, ref_gray, feature_ctx, targets_ref, min_inliers=10)
    centers_fr_dict, oob_detections, raw_proj, H, _ = result

    if not centers_fr_dict and H is None:
        #skip image and save into separate txt file
        with open(not_used_images_path, "a", encoding="utf-8") as f:
            f.write(image_path)
            f.write("\n")
        logging.error(f"file {image_filename} skipped and saved to unused_images.txt")
        return Result({}, 0, None, None, None)

    detected = []
    if centers_fr_dict:
        # Keep order P1..P9 if present
        for lab in sorted(centers_fr_dict.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
            detected.append(centers_fr_dict[lab])
    else:
        # Homography failed -> return empty; caller may skip or log
        logger.info(f"No homography for {image_filename} (skip / retry).")
        detected = []

    # Build detections dict in FRAME coordinates (same downstream format)
    detections = {}
    for label, (x, y) in centers_fr_dict.items():
        normx = float(x / w_frame)
        normy = float(y / h_frame)
        detections[label] = ((x, y), (normx, normy))

    selected_detected = filter_detections(detections, w_frame, h_frame, img_path, edgecase)
    return Result(selected_detected, frame, image_filename, oob_detections, raw_proj)


def relabel_grid_points(detections):
    """
    Sorts and relabels 9 detections in a 3x3 grid:
    
    P1 P2 P3
    P4 P5 P6
    P7 P8 P9

    Sorting is done top-to-bottom (by y), then left-to-right (by x) within rows.
    """
    # Extract list of (original_key, ((x_px, y_px), (x_norm, y_norm)))
    points = list(detections.values())  # We only need ((x_px, y_px), (x_norm, y_norm))

    # Sort all points by normalized y (ascending = top to bottom)
    points.sort(key=lambda item: item[1][1])  # sort by y_norm

    # Cluster into 3 rows
    row1 = sorted(points[:3], key=lambda item: item[1][0])  # sort by x_norm (left to right)
    row2 = sorted(points[3:6], key=lambda item: item[1][0])
    row3 = sorted(points[6:9], key=lambda item: item[1][0])

    # Combine into final ordered list
    ordered_points = row1 + row2 + row3

    # Create new relabeled dict
    relabeled = {
        f"P{index+1}": value
        for index, value in enumerate(ordered_points)
    }

    return relabeled


# -------------- Main loop (loads reference & features once per subfolder) --------------
def main(root_dir, threshold):
    default_threshold = threshold
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for task_id in to_do_list:
            if task_id in folder:
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if not os.path.isdir(subfolder_path):
                        continue

                    img_processing_paths = find_img_proc_paths(subfolder_path)
                    #templates = find_template_files(subfolder_path)
                    ref_path = find_reference_poster(subfolder_path)

                    if len(img_processing_paths) != 4:
                        logging.info(f"Skipping {subfolder}: less than 4 img folders found.")
                        continue
                    #if len(templates) != 4:
                        #logging.info(f"Skipping {subfolder}: less than 4 templates found.")
                        #continue
                    if ref_path is None:
                        logging.info(f"Skipping {subfolder}: poster_reference.(png/jpg) not found.")
                        continue

                    # Load and prepare reference once
                    ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                    if ref_gray is None:
                        logging.info(f"Skipping {subfolder}: failed to read reference {ref_path}.")
                        continue

                    targets_ref = load_targets_ref(subfolder_path)
                    if targets_ref is None or len(targets_ref) == 0:
                        logging.info(f"Skipping {subfolder}: poster_targets.json not found or empty.")
                        continue

                    feature_ctx = init_feature_model(ref_gray)
                    output_path = os.path.join(subfolder_path, "OpenCV")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    th = default_threshold

                    for key in img_processing_paths.keys():
                        img_int_path = img_processing_paths[key]

                        subfolder = Path(subfolder).name
                        short_path = Path(img_int_path).name
                        logging.info(f"working on: {subfolder} in {short_path} (threshold {th})")

                        all_saved_timestamps = []
                        for image_filename in os.listdir(img_int_path):
                            if not image_filename.lower().endswith(image_endings):
                                current_path = Path(img_int_path).parent.name
                                logging.info(f"Skipping non-image file {image_filename}, {current_path}")
                                continue

                            image_path = os.path.join(img_int_path, image_filename)
                            edgecase = False

                            #single attempt (projection does not use threshold loops)
                            positions = find_marker_positions(image_path, None, img_int_path, th, image_filename, edgecase, feature_ctx, ref_gray, targets_ref)
                            detections, frame, image_filename, oob_detections, raw_proj = positions
                            if not detections and not image_filename:
                                #image will be skipped.
                                continue

                            sum_detections = len(oob_detections) + len(detections)
                            if len(detections) > 5:
                                if sum_detections == 9:
                                    relabeled_detections = detections
                                    #relabeled_detections = relabel_grid_points(detections)
                                if sum_detections != 9:
                                    logging.error(f"something is wrong, there are {sum_detections} detections found in total for {image_filename} in {subfolder}, {short_path}.")
                                else:
                                    relabeled_detections = detections

                                save_image_and_json(relabeled_detections, frame, output_path, image_filename, img_int_path)
                                filename = image_filename.split("timestamp")[1]
                                timestamp = filename.split(".png")[0]
                                all_saved_timestamps.append(float(timestamp))
                            else:
                                logging.info(f"{image_filename}: got {len(detections)} detections: {subfolder}, {short_path}")



    logger.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dots in extracted frame from video (homography rectified).")
    parser.add_argument("--root_dir", "-r", required=True, help="Path to the root directory containing all folders like P004_statisch, etc.")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="Matching threshold for template matching.")
    args = parser.parse_args()
    print(f"Root directory: {args.root_dir}")
    main(args.root_dir, args.threshold)

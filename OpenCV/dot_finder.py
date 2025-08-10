import cv2
import numpy as np
import os
import argparse
import sys
import json
from pathlib import Path
import logging

'''
Script to find fixation points (dots) in extracted video frames and saving their values to jsonl files. 

LD: Left Down       <-- P7
MD: Middle Down     <-- P8
RD: Right Down      <-- P9
LU: Left Up         <-- P1
MU: Middle Up       <-- P2
RU: Right Up        <-- P3
LM: Left Middle     <-- P4
MM: Middle Middle   <-- P5
RM: Right Middle    <-- P6

Parameter: 
    frame_path: path to video frames sorted in different subfolders. 
    template: extracted image from one fixation point (dot) to act as the sample of what we are trying to find (the dots).
    output_path: path to where jsonl files should be saved using a similiar folder-structure as in frame_path.

Methods: 
    check_n_del_images: preprocessing subfolders and checking if there are any files that are not images and deleting those. 
    find_marker_positions: using subfolder structure and checking every frame for fixation points and saving those to csv files.

Done: 521: P002, P018, P019, P020, P021, P022, P023, P024, P025, P027, P028, P032, P036, P035, P034, P031
Skipped: 521: P029 80_bL manually

Done: 581: P005, P006, P008, P009, P011, P012, P013, P014, P015, P021, P033, P017, P007,
Skipped: 581: P004 (manually 803L) P010 manually, P016_80_3L manually

'''

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\521_stat_conv\my_log.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

to_do_list = ["P002"]

image_endings = (".png", ".jpg", ".jpeg")

def save_image_and_json(selected_detected, frame, output_path, image_filename, subfolder_name):
    for key, ((x,y), (normx, normy)) in selected_detected.items():
        cv2.line(frame, (x - 5, y), (x + 5, y), (0, 255, 0), 1)
        cv2.line(frame, (x, y - 5), (x, y + 5), (0, 255, 0), 1)
        coords = f"{key} {round(normx,4)},{round(normy,4)}"
        cv2.putText(frame, text=coords, org=(x-50, y-15), fontScale=1, fontFace=2, color=(0,255,0), thickness=2)

    # Save image
    output_images = os.path.join(output_path, "images")
    os.makedirs(output_images, exist_ok=True)
    output_image_path = os.path.join(output_images, image_filename)
    cv2.imwrite(output_image_path, frame)

    # Save coords
    subfolder_output = os.path.join(output_path, subfolder_name + "_coords")
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

def find_template_file(folder_path):
    for ext in image_endings:
        candidate = os.path.join(folder_path, f"template{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None

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

def delete_detections(key_list, detections):
    key_list.sort(key=lambda x: int(x[1:]), reverse=True)
    filtered_detections = [
        value for key, value in detections.items()
        if key not in key_list
    ]

    # Step 2: Reindex remaining entries as P1, P2, ...
    reindexed_detections = {
        f"P{idx+1}": value
        for idx, value in enumerate(filtered_detections)
    }

    # Optional: replace original
    detections = reindexed_detections

    return detections

def filter_detections(detections, w_frame, h_frame, img_processing_path):
    current_path = Path(img_processing_path)
    distance = current_path.parent.name
    filtered_detections = {}
    #print("raw detections: ", len(raw_detections))
    #print("parent folder name: ", distance)

    del_list_80 = []
    del_list_120 = []
    del_list_180 = []
    for key, ((x,y), (normx, normy)) in detections.items():
        if "_80" in distance:
            if (normx > 0.85 
                or normx < 0.13 
                or normy > 0.89):
                #print(f"skipping invalid input _80: {normx}, {normy}")
                del_list_80.append(key)

        elif "_120" in distance:
            if (normx < 0.26 
                or normx > 0.74 
                or normy < 0.06 
                or normy > 0.78):
                #print(f"skipping invalid input _120: {normx}, {normy}")
                del_list_120.append(key)

        elif "_180" in distance:
            if (normx < 0.25 
                or normx > 0.65 
                or normy < 0.07 
                or normy > 0.75):
                #print(f"skipping invalid input _180: {normx}, {normy}")
                del_list_180.append(key)

    if del_list_80 and "_80" in distance:
        filtered_detections = delete_detections(del_list_80, detections)

    elif del_list_120 and "_120" in distance: 
        filtered_detections = delete_detections(del_list_120, detections)

    elif del_list_180 and "_180" in distance: 
        filtered_detections = delete_detections(del_list_180, detections)
        
    else:
        filtered_detections = detections

    cleaned_detections = remove_close_points(filtered_detections, min_distance=10)
    
    return cleaned_detections


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
                removed_keys.add(key2)  # remove the second one
                logger.info(f"Removed duplicate point {key2} (too close to {key1}, dist={dist:.2f})")
    # Rebuild detections without removed keys
    cleaned = {k: v for k, v in detections.items() if k not in removed_keys}

    return cleaned


def find_marker_positions(image_path, template_path, img_processing_path, threshold, image_filename):
    #print(f"Processing image: {image_path}")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error((f"Bild nicht gefunden: {image_path}"))
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = gray_frame.shape

    # Load and validate template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.error(f"Template nicht gefunden: {template_path}")
        raise FileNotFoundError(f"Template nicht gefunden: {template_path}")
    h_temp, w_temp = template.shape

    # Template Matching
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    detected = []
    matches = sorted(zip(*locations[::-1]), key=lambda pt: result[pt[1], pt[0]], reverse=True)
    for pt in matches:
        center = (pt[0] + w_temp // 2, pt[1] + h_temp // 2)
        if all(np.linalg.norm(np.array(center) - np.array(d)) >= 10 for d in detected):
            detected.append(center)
        if len(detected) == 9:
            break

    norm_coords = [(x / w_frame, y / h_frame) for (x, y) in detected]
    current_path = Path(image_path).parent.name

    detections = {}
    # save detections into dictionary
    for count, (x, y) in enumerate(detected, 1):
        normx = float((x / w_frame))
        normy = float((y / h_frame))
        detections[f"P{count}"] = ((x, y), (normx, normy))

    if len(detected) == 0:
        logger.info(f"No detections found in image {image_filename}, {current_path} with threshold {threshold}")

    selected_detected = filter_detections(detections, w_frame, h_frame, img_processing_path)

    return selected_detected, frame, image_filename


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

                    img_processing_path = os.path.join(subfolder_path, "image_processing")
                    template_path = find_template_file(subfolder_path)

                    if not os.path.isdir(img_processing_path):
                        logger.info(f"Skipping {subfolder}: no image_processing folder found.")
                        continue

                    if not os.path.isfile(template_path):
                        logger.info(f"Skipping {subfolder}: template.png not found.")
                        continue

                    output_path = os.path.join(subfolder_path, "OpenCV")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    threshold = default_threshold
                    if "180" in subfolder_path:
                        threshold = 0.85

                    short_path = Path(img_processing_path).parent.name
                    logging.info(f"working on: {short_path} with threshold {threshold}")
                    for subfolder_name in os.listdir(img_processing_path):
                        subfolder_path = os.path.join(img_processing_path, subfolder_name)

                        if not os.path.isdir(subfolder_path):
                            continue  # skip files

                        for image_filename in os.listdir(subfolder_path):
                            if not image_filename.lower().endswith(image_endings):
                                current_path = Path(img_processing_path).parent.name
                                logging.info(f"Skipping non-image file {image_filename}, {current_path}")
                                continue  # skip non-image files

                            image_path = os.path.join(subfolder_path, image_filename)

                            current_path = Path(img_processing_path).parent.name
                            gt_count = 9
                            edgecase = 8
                            current_threshold = threshold

                            if (current_threshold < 0.4):
                                print("Threshold getting dangerously low...")

                            var = 0
                            while True:
                                if var > 35:
                                    if len(detections) == edgecase or len(detections) == 7 or len(detections) == 6:
                                        logging.error(f"{image_filename} has too many iterations {var}, but edgecase - check manually! {current_path}")
                                        #relabeled_detections = relabel_grid_points(detections)
                                        #save_image_and_json(relabeled_detections, frame, output_path, image_filename, subfolder_name)
                                        #break
                                    #break

                                detections, frame, image_filename = find_marker_positions(image_path, template_path, img_processing_path, current_threshold, image_filename)
                                if len(detections) > gt_count:
                                    #print(f"len(detections): {len(detections)}, used threshold: {current_threshold} in {image_filename}")
                                    current_threshold += 0.01
                                    var += 1
                                elif len(detections) < gt_count:
                                    #print(f"len(detections): {len(detections)}, used threshold: {current_threshold} in {image_filename}")
                                    current_threshold -= 0.01
                                    var += 1
                                elif len(detections) == gt_count:
                                    relabeled_detections = relabel_grid_points(detections) 
                                    save_image_and_json(relabeled_detections, frame, output_path, image_filename, subfolder_name)
                                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dots in extracted frame from video.")
    parser.add_argument("--root_dir", "-r", required=True, help="Path to the root directory containing all folders like P004_statisch, etc.")
    #parser.add_argument("--output_path", "-o", required=True, help="Base output path where results should be saved.")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="Matching threshold for template matching.")

    args = parser.parse_args()

    print(f"Root directory: {args.root_dir}")
    #print(f"Output path: {args.output_path}")

    main(args.root_dir, args.threshold)

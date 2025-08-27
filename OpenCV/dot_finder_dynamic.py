import cv2
import numpy as np
import os
import argparse
import sys
import json
from pathlib import Path
import logging
from math import sqrt

'''
dot finder for dynamic recordings: 
    - only the center fixation point is needed for evaluation
    - more movement in recordings: divided into separate intervals: left side, middle, right side <- where participant is standing looking at the poster
    - uses different acceptance windows for where the found template is a real match

one label: MM - Middle 

Parameter: 
    frame_path: path to video frames sorted in different subfolders. 
    output_path: path to where jsonl files should be saved using a similiar folder-structure as in frame_path.

Methods: 
    check_n_del_images: preprocessing subfolders and checking if there are any files that are not images and deleting those. 
    find_marker_positions: using subfolder structure and checking every frame for fixation points and saving those to csv files.
'''

# Setup basic logging
log_path = r"D:\WorkingFolder_PythonD\test_usingStat_dynam\dot_finder_stat9Dots.log"

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, etc. as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path), 
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

to_do_list = ["P020"]

image_endings = (".png", ".jpg", ".jpeg")

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

def filter_detections(detections, w_frame, h_frame, img_processing_path, edgecase):
    current_path = Path(img_processing_path)
    distance = current_path.parent.parent.name
    interval = current_path.name
    filtered_detections = {}
    static = True
    #print("raw detections: ", len(raw_detections))
    #print("parent folder name: ", distance)

    del_list_80 = []
    del_list_120 = []
    del_list_180 = []
    for key, ((x,y), (normx, normy)) in detections.items():
        if static is True: 
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
        
        else: 
            if "_80" in distance:
                match interval:
                    case "1_interval":
                        if (normx < 0.44 
                            or normx > 0.56 
                            or normy < 0.27
                            or normy > 0.54):
                            del_list_80.append(key)

                    case "2_interval":
                        if (normx < 0.41 
                            or normx > 0.58 
                            or normy < 0.28
                            or normy > 0.53):
                            del_list_80.append(key)

                    case "3_interval":
                        if (normx < 0.42 
                            or normx > 0.60 
                            or normy < 0.28
                            or normy > 0.53):
                            del_list_80.append(key)

                    case "4_interval":
                        if (normx < 0.43
                            or normx > 0.58
                            or normy < 0.27
                            or normy > 0.52):
                            del_list_80.append(key)

            elif "_120" in distance:
                match interval:
                    case "1_interval":
                        if (normx < 0.42 
                            or normx > 0.54 
                            or normy < 0.24
                            or normy > 0.48):
                            del_list_120.append(key)

                    case "2_interval":
                        if (normx < 0.42 
                            or normx > 0.55 
                            or normy < 0.28
                            or normy > 0.48):
                            del_list_120.append(key)

                    case "3_interval":
                        if (normx < 0.45 
                            or normx > 0.57 
                            or normy < 0.28
                            or normy > 0.46):
                            del_list_120.append(key)

                    case "4_interval":
                        if (normx < 0.45
                            or normx > 0.55
                            or normy < 0.27
                            or normy > 0.46):
                            del_list_120.append(key)

            elif "_180" in distance:
                if edgecase: #try static and save all 9 points
                        if (normx > 0.59 
                            or normx < 0.50 
                            or normy > 0.42
                            or normy < 0.28):
                            #print(f"skipping invalid input _80: {normx}, {normy}")
                            del_list_180.append(key)
                else:
                    match interval:
                        case "1_interval":
                            if (normx < 0.47 
                                or normx > 0.54 
                                or normy < 0.31
                                or normy > 0.42):
                                del_list_180.append(key)

                        case "2_interval":
                            if (normx < 0.46 
                                or normx > 0.53 
                                or normy < 0.31
                                or normy > 0.42):
                                del_list_180.append(key)

                        case "3_interval":
                            if (normx < 0.47 
                                or normx > 0.56 
                                or normy < 0.31
                                or normy > 0.43):
                                del_list_180.append(key)

                        case "4_interval":
                            if (normx < 0.47
                                or normx > 0.55
                                or normy < 0.30
                                or normy > 0.43):
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


def find_marker_positions(image_path, template_path, img_path, threshold, image_filename, edgecase):
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
        if len(detected) > 4:
            break
        elif edgecase and len(detected) >= 1:
            break

    norm_coords = [(x / w_frame, y / h_frame) for (x, y) in detected]
    parent_path = Path(image_path).parent.parent.parent.name
    current_interval = Path(image_path).parent.name

    detections = {}
    # save detections into dictionary
    for count, (x, y) in enumerate(detected, 1):
        normx = float((x / w_frame))
        normy = float((y / h_frame))
        detections[f"P{count}"] = ((x, y), (normx, normy))

    if len(detected) == 0:
        logger.info(f"No detections found in image {image_filename}, {parent_path} in {current_interval} with threshold {threshold}")

    selected_detected = filter_detections(detections, w_frame, h_frame, img_path, edgecase)

    return selected_detected, frame, image_filename

def check_previous_img_to_compare(list_prev_timestamps, current_image_filename, current_detections, frame, img_int_path):
    filename = current_image_filename.split("timestamp")[1]
    cur_timestamp = filename.split(".png")[0]
    fl_timestamp = float(cur_timestamp)
    closest = min(list_prev_timestamps, key=lambda t: abs(t - fl_timestamp))
    closest_str = f"{closest:.2f}"
    previous_file = current_image_filename.replace(cur_timestamp, closest_str)
    previous_json = previous_file.replace(".png", ".json")
    json_filename = "coords_" + previous_json
    #D:\WorkingFolder_PythonD\testDynam\P018_dynamisch\P018_180cm_dnam_3lights\image_processing\1_interval
    parent_path = Path(img_int_path).parent.parent
    json_path = os.path.join(parent_path, "OpenCV", "MM_coords", json_filename)

    closest_key = None
    closest_val = None
    closest_dist = float("inf")

    if not os.path.isfile(json_path):
        alternative_filename = "coords_prev_" + previous_json
        json_path = json_path.replace(json_filename, alternative_filename)
        if not os.path.isfile(json_path):
            logging.debug(f"something went wrong. no json found with {json_filename}.") 

    with open(json_path, "r") as j:
        data = json.load(j)

        if len(data) > 1:
            logging.debug(f"something went wrong in the previous file: {previous_json} in {img_int_path}. More than one entry!")

        elif len(data) == 1:
            key, coords = next(iter(data.items()))
            prev_x, prev_y = map(float, coords)
        
        if len(current_detections) == 0:
            logging.error(f"current_detection still zero, cant proceed {current_image_filename} in {img_int_path}")
            return 1
            #h_frame, w_frame = frame.shape[:2]
            #a = int(round(prev_x * w_frame))
            #b = int(round(prev_y * h_frame))
            #clamp coordinates, to ensure they stay inside frame (safety net)
            #px_a = max(0, min(w_frame - 1, a))
            #py_b = max(0, min(h_frame - 1, b))
            #logging.debug(f"set {current_image_filename} to {previous_json}")
            #keeping_detections = {key: ((px_a, py_b), (prev_x, prev_y))}
            #return keeping_detections
        
        for key, ((a, b), (cur_x, cur_y)) in current_detections.items():
            # compute distance to previous point
            float_x = round(float(cur_x), 4)
            float_y = round(float(cur_y), 4)
            dx = float_x - prev_x
            dy = float_y - prev_y
            dist = sqrt(dx*dx + dy*dy)

            # check if this one is closer
            if dist < closest_dist:
                closest_dist = dist
                closest_key = key
                closest_val = (cur_x, cur_y)
                cur_a = a
                cur_b = b   
        
        keep_vals = ((cur_a, cur_b), closest_val)
        keeping_detections = {closest_key: keep_vals}
    return keeping_detections


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
                    templates = find_template_files(subfolder_path)

                    if len(img_processing_paths) != 4:
                        logger.info(f"Skipping {subfolder}: less than 3 img folders found.")
                        continue

                    if len(templates) != 4:
                        logger.info(f"Skipping {subfolder}: less than 3 templates found.")
                        continue

                    output_path = os.path.join(subfolder_path, "OpenCV")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    threshold = default_threshold
                    if "180" in subfolder_path:
                        threshold = 0.85

                    for key in img_processing_paths.keys() & templates.keys():
                        img_int_path = img_processing_paths[key]
                        template_path = templates[key]
                        
                        subfolder = Path(subfolder).name
                        short_path = Path(img_int_path).name
                        logging.info(f"working on: {subfolder} in {short_path} with threshold {threshold}")

                        all_saved_timestamps = []
                        for image_filename in os.listdir(img_int_path):
                            if not image_filename.lower().endswith(image_endings):
                                current_path = Path(img_int_path).parent.name
                                logging.info(f"Skipping non-image file {image_filename}, {current_path}")
                                continue  # skip non-image files
                            
                            image_path = os.path.join(img_int_path, image_filename)

                            current_threshold = threshold
                            edgecase = 0

                            if (current_threshold < 0.4):
                                print("Threshold getting dangerously low...")

                            var = 0
                            while True:
                                if var > 35:
                                    #statisch: 7 und 6 als weitere optionen
                                    if len(detections) == 8 or len(detections) == 7 or len(detections) == 6 or len(detections) == 5:
                                        logging.error(f"{image_filename} has too many iterations {var}, but edgecase - check manually! {subfolder}, {short_path}")
                                        relabeled_detections = relabel_grid_points(detections)
                                        save_image_and_json(relabeled_detections, frame, output_path, image_filename, img_int_path)
                                        break
                                    break

                                detections, frame, image_filename = find_marker_positions(image_path, template_path, img_int_path, current_threshold, image_filename, edgecase)
                                
                                if len(detections) > 9:
                                    #print(f"len(detections): {len(detections)}, used threshold: {current_threshold} in {image_filename}")
                                    current_threshold += 0.01
                                    var += 1
                                elif len(detections) < 9:
                                    #print(f"len(detections): {len(detections)}, used threshold: {current_threshold} in {image_filename}")
                                    current_threshold -= 0.01
                                    var += 1
                                elif len(detections) == 9:
                                    relabeled_detections = relabel_grid_points(detections) 
                                    save_image_and_json(relabeled_detections, frame, output_path, image_filename, img_int_path)
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
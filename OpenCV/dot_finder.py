import cv2
import numpy as np
import os
import argparse
import sys
import json
from pathlib import Path

'''
Script to find fixation points (dots) in extracted video frames and saving their values to jsonl files. 

LD: Left Down       <-- P9
MD: Middle Down     <-- P8
RD: Right Down      <-- P7
LU: Left Up         <-- P3
MU: Middle Up       <-- P2
RU: Right Up        <-- P1
LM: Left Middle     <-- P6
MM: Middle Middle   <-- P5
RM: Right Middle    <-- P4

Parameter: 
    frame_path: path to video frames sorted in different subfolders. 
    template: extracted image from one fixation point (dot) to act as the sample of what we are trying to find (the dots).
    output_path: path to where jsonl files should be saved using a similiar folder-structure as in frame_path.

Methods: 
    check_n_del_images: preprocessing subfolders and checking if there are any files that are not images and deleting those. 
    find_marker_positions: using subfolder structure and checking every frame for fixation points and saving those to csv files.

Done: 521: P002, P018, P019,     

'''

to_do_list = ["P020"]

image_endings = (".png", ".jpg", ".jpeg")

def find_template_file(folder_path):
    for ext in image_endings:
        candidate = os.path.join(folder_path, f"template{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None

def filter_detections(detected, w_frame, h_frame, img_processing_path):
    raw_detections = []
    for x, y in detected:
                normx = float(x / w_frame)
                normy = float(y / h_frame)
                raw_detections.append({"pixel": (x, y), 
                                       "norm": (normx, normy)})

    current_path = Path(img_processing_path)
    distance = current_path.parent.name
    #print("raw detections: ", len(raw_detections))
    #print("parent folder name: ", distance)

    filtered = []
    for detection in raw_detections:
        normx, normy = detection["norm"]  # unpack normalized coordinates
        if "_80" in distance:
            if normx > 0.85 or normx < 0.15 or normy > 0.85:
                print(f"skipping invalid input _80: {normx}, {normy}")
                continue  # skip invalid
        elif "_120" in distance:
            if normx < 0.28 or normx > 0.70 or normy < 0.10 or normy > 0.70:
                print(f"skipping invalid input _120: {normx}, {normy}")
                continue
        elif "_180" in distance:
            if normx < 0.35 or normx > 0.65 or normy < 0.15 or normy > 0.75:
                print(f"skipping invalid input _180: {normx}, {normy}")
                continue
        filtered.append(detection)

    detections = {}
    for count, detection in enumerate(filtered, 1):
        detections[f"P{count}"] = detection
    
    return detections



def find_marker_positions(img_processing_path, template_path, output_path, threshold):
    for subfolder_name in os.listdir(img_processing_path):
        subfolder_path = os.path.join(img_processing_path, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue  # skip files

        print(f"subfolder: {subfolder_path}")
        for image_filename in os.listdir(subfolder_path):
            if not image_filename.lower().endswith(image_endings):
                continue  # skip non-image files

            image_path = os.path.join(subfolder_path, image_filename)
            #print(f"Processing image: {image_path}")

            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_frame, w_frame = gray_frame.shape

            # Load and validate template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError(f"Template nicht gefunden: {template_path}")
            h_temp, w_temp = template.shape

            # Template Matching
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            detected = []
            for pt in zip(*locations[::-1]):
                center = (pt[0] + w_temp // 2, pt[1] + h_temp // 2)
                if not any(np.linalg.norm(np.array(center) - np.array(d)) < 10 for d in detected):
                    detected.append(center)

            norm_coords = [(x / w_frame, y / h_frame) for (x, y) in detected]

            selected_detected = filter_detections(detected, w_frame, h_frame, img_processing_path)

            # Draw detections
            for count, (key, detection) in enumerate(selected_detected.items(), 1):
                x, y = detection["pixel"]
                normx, normy = detection["norm"]
                round_normx = round(normx, 4)
                round_normy = round(normy, 4)
                coords = f"P{count} {round_normx},{round_normy}"
                cv2.line(frame, (x - 5, y), (x + 5, y), (0, 255, 0), 1)
                cv2.line(frame, (x, y - 5), (x, y + 5), (0, 255, 0), 1)
                cv2.putText(frame, text=coords, org=(x-50, y-15), fontScale=1, fontFace=2, color=(0,255,0), thickness=2)

            # Save image
            output_images = os.path.join(output_path, "images")
            os.makedirs(output_images, exist_ok=True)
            output_image_path = os.path.join(output_images, image_filename)
            cv2.imwrite(output_image_path, frame)

            gt_count = 9
            len_count = len(selected_detected)
            if len_count != gt_count:
                print(f"{len_count} Treffer gefunden und gespeichert unter: {output_image_path}, threshold: {threshold}")

            # Save coords
            subfolder_output = os.path.join(output_path, subfolder_name + "_coords")
            os.makedirs(subfolder_output, exist_ok=True)
            img_name = os.path.splitext(image_filename)[0]
            output_text_path = os.path.join(subfolder_output, f"coords_{img_name}.json")

            formatted_detections = {}

            for key, detection in selected_detected.items():
                normx, normy = detection["norm"]
                x_str = f"{normx:.4f}0000"
                y_str = f"{normy:.4f}0000"
                formatted_detections[key] = [x_str, y_str]

            with open(output_text_path, "w") as f:
                json.dump(formatted_detections, f, indent=2)

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

                    image_processing_path = os.path.join(subfolder_path, "image_processing")
                    template_path = find_template_file(subfolder_path)

                    if not os.path.isdir(image_processing_path):
                        print(f"Skipping {subfolder}: no image_processing folder found.")
                        continue

                    if not os.path.isfile(template_path):
                        print(f"Skipping {subfolder}: template.png not found.")
                        continue

                    output_path = os.path.join(subfolder_path, "OpenCV")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    threshold = default_threshold
                    if "180" in subfolder_path:
                        threshold = 0.88

                    find_marker_positions(image_processing_path, template_path, output_path, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dots in extracted frame from video.")
    parser.add_argument("--root_dir", "-r", required=True, help="Path to the root directory containing all folders like P004_statisch, etc.")
    #parser.add_argument("--output_path", "-o", required=True, help="Base output path where results should be saved.")
    parser.add_argument("--threshold", "-t", type=float, default=0.62, help="Matching threshold for template matching.")

    args = parser.parse_args()

    print(f"Root directory: {args.root_dir}")
    #print(f"Output path: {args.output_path}")

    main(args.root_dir, args.threshold)

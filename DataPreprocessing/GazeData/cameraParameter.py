#!/usr/bin/env python3
import numpy as np
import cv2
import importlib.util
import os
import sys
import json

def calibrate_camera(obj_points_data, norm_img_points_data, img_size):
    if len(obj_points_data) < 6:
        raise ValueError("Fehler: Es werden mindestens 6 Punkte für die Kalibrierung benötigt.")

    width, height = img_size
    image_points_pixel = np.array(
        [[p[0] * width, p[1] * height] for p in norm_img_points_data],
        dtype=np.float32,
    )

    object_points = [np.array(obj_points_data, dtype=np.float32)]
    image_points = [image_points_pixel]

    initial_camera_matrix = np.array([
        [float(width), 0, width / 2],
        [0, float(width), height / 2],
        [0, 0, 1],
    ], dtype=np.float32)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (width, height),
        initial_camera_matrix, None, flags=flags
    )

    if not ret:
        raise RuntimeError("Kalibrierung fehlgeschlagen. Überprüfe deine Punkte.")

    return {
        "rms_error": ret,
        "data_count":len(obj_points_data),
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist()[0],  # Flatten the array
        "rvec": rvecs[0].flatten().tolist(),
        "tvec": tvecs[0].flatten().tolist()
    }

def load_data_from_py(py_file):
    spec = importlib.util.spec_from_file_location("gaze_data", py_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gaze_data"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "object_points_data") or not hasattr(module, "normalized_image_points_data"):
        raise AttributeError("Fehlende Daten in Datei.")

    return module.object_points_data, module.normalized_image_points_data

def process_all_conversions(root_dir, image_size=(1920, 1080)):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) != "conversion":
            continue

        py_file = os.path.join(dirpath, "2D3DGazeLists.py")
        if not os.path.isfile(py_file):
            print(f"Übersprungen: {dirpath} (keine 2D3DGazeLists.py)")
            continue

        try:
            obj_pts, img_pts = load_data_from_py(py_file)

            if len(obj_pts) != len(img_pts):
                print(f"Ungleiche Anzahl an 2D- und 3D-Punkten, übersprungen: {py_file}")
                continue

            result = calibrate_camera(obj_pts, img_pts, image_size)

            # Remove old TXT file if it exists
            old_js_path = os.path.join(dirpath, "camera_parameters.json")
            if os.path.isfile(old_js_path):
                os.remove(old_js_path)

            # Save to JSON
            out_json_path = os.path.join(dirpath, "camera_parameters.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            #print(f"Gespeichert: {out_json_path}")
        except Exception as e:
            print(f"Fehler in {dirpath}: {e}")

if __name__ == "__main__":
    ROOT_DIR = "D:\\WorkingFolder_PythonD\\2Dto3D_Conversion\\581_dynam"
    process_all_conversions(ROOT_DIR)

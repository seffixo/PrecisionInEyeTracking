import os
import json
import argparse
import numpy as np
import cv2

def project_3d_to_2d(
    points_3d,
    camera_matrix,
    distortion_coeffs,
    rotation_vector,
    translation_vector,
    normalize_coordinates=False,
    image_size=None
):
    points_3d = np.array(points_3d, dtype=np.float32)
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    distortion_coeffs = np.array(distortion_coeffs, dtype=np.float32)
    rotation_vector = np.array(rotation_vector, dtype=np.float32)
    translation_vector = np.array(translation_vector, dtype=np.float32)

    single_point = False
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)
        single_point = True

    if points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3) or (3,)")

    image_points, _ = cv2.projectPoints(
        points_3d,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coeffs
    )

    image_points = image_points.reshape(-1, 2)

    if normalize_coordinates:
        if image_size is None:
            raise ValueError("image_size must be provided when normalize_coordinates=True")
        width, height = image_size
        image_points[:, 0] /= width
        image_points[:, 1] /= height

    return image_points[0] if single_point else image_points


def load_camera_parameters(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return (
        data["camera_matrix"],
        data["distortion_coefficients"],
        data["rvec"],
        data["tvec"]
    )


def load_object_points_from_py(py_path):
    # Execute file safely and extract only 'object_points_data'
    namespace = {}
    with open(py_path, "r") as f:
        code = f.read()
    exec(code, {}, namespace)
    if "object_points_data" not in namespace:
        raise ValueError(f"'object_points_data' not found in {py_path}")
    return namespace["object_points_data"]


def process_conversion_folder(conversion_path, cam_path):
    json_path = os.path.join(cam_path, "conversion/camera_parameters.json")
    #print(f"json path is {json_path}")
    gaze_py_path = os.path.join(conversion_path, "2D3DGazeLists.py")
    output_path = os.path.join(conversion_path, "normalized_coords.py")

    if not (os.path.isfile(json_path) and os.path.isfile(gaze_py_path)):
        print(f"Skipping (missing files): {conversion_path}")
        return

    try:
        camera_matrix, distortion_coeffs, rvec, tvec = load_camera_parameters(json_path)
        object_points = load_object_points_from_py(gaze_py_path)
        image_size = (1920, 1080)

        normalized_coords = project_3d_to_2d(
            object_points,
            camera_matrix,
            distortion_coeffs,
            rvec,
            tvec,
            normalize_coordinates=True,
            image_size=image_size
        )

        with open(output_path, "w") as out_file:
            out_file.write("normalized_coords = [\n")
            for point in normalized_coords:
                out_file.write(f"    ({point[0]:.4f}0000, {point[1]:.4f}0000),\n")
            out_file.write("]\n")

        #print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {conversion_path}: {e}")


def find_conversion_folders(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
             if d == "conversion":
                yield os.path.join(root, d)


def main():
    parser = argparse.ArgumentParser(description="Process calibration and gaze data.")
    parser.add_argument("--root_dir", help="Root directory containing recordings.")
    parser.add_argument("--camParaFolder", help="folder for used camera parameters")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)

    # Join everything together
    cam_path = os.path.join(root_dir, args.camParaFolder)

    print(f"\nSearching in: {args.root_dir}\n")
    for conversion_folder in find_conversion_folders(args.root_dir):
        #print(f"→ Processing: {conversion_folder}")
        process_conversion_folder(conversion_folder, cam_path)


if __name__ == "__main__":
    main()

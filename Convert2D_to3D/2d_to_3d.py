import numpy as np
import cv2
import os
import json

#camera parameters for 521
camera_matrix = np.array([
        [911.7661807262908, 0.0, 953.1539301425507],
        [0.0, 911.2665673922147, 515.5778687592352],
        [0.0, 0.0, 1.0]
], dtype=np.float32)

distortion_coeffs = np.array(
    [-0.04277252073338572, 0.061178714113208424,
        -2.0962433109889865e-06, 0.0002723710640655781,
        -0.040889510305845714],
    dtype=np.float32
)

rotation_vector = np.array([0.003823618491883325, -0.006546093505797456, 3.141501770525814], dtype=np.float32)
translation_vector = np.array([0.005087319435724582, 0.0018889011476757851, -0.01485958162960992], dtype=np.float32)

# === FUNCTIONS ===

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return v / norm

def get_obj_points(obj_path):
    f_data = {}
    with open(obj_path) as f: 
        exec(f.read(), f_data)

    return f_data["normalized_image_points_data"]

def compute_normalized_3d_gaze_vectors(obj_path, K, dist_coeffs, rvec, tvec):
    object_points = get_obj_points(obj_path)

    object_points_np = np.array(object_points, dtype=np.float32).reshape(-1, 1, 2)

    # Step 1: Undistort 2D points
    undistorted_points = cv2.undistortPoints(object_points_np, K, dist_coeffs, P=K)

    # Step 2: Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Step 3: Compute direction vectors in world coordinates
    rays_world = []
    for pt in undistorted_points:
        # Point in normalized camera coordinates
        x, y = pt[0]
        cam_vec = np.array([x, y, 1.0])  # Direction in camera space

        # Transform direction into world space (rotate only, no translation for direction)
        world_vec = R.T @ cam_vec
        world_vec_normalized = normalize_vector(world_vec)

        rays_world.append(world_vec_normalized)

    return rays_world

def save_results(out_path, gaze_rays):
    with open(out_path, "w") as f:
        f.write("gaze_rays_world = [\n")
        for vec in gaze_rays:
            f.write(f"    {vec.tolist()},\n")
        f.write("]\n")

    print(f"output was saved to: {out_path}")



def main():
    obj_path = "..\\..\\WorkingFolder_Python\\Conv2D_to3D\\521_stat_conv\\P018_statisch\\P018_80cm_stat_3lights\\conversion\\2D3DGazeLists.py"
    gaze_rays_world = compute_normalized_3d_gaze_vectors(
        obj_path,
        camera_matrix,
        distortion_coeffs,
        rotation_vector,
        translation_vector
    )
    folder_path = os.path.dirname(obj_path)
    out_path = os.path.join(folder_path, "2d_to_3d.py")
    save_results(out_path, gaze_rays_world)


if __name__ == "__main__":
    main()

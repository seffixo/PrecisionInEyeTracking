import os
import argparse

def extract_variable_from_py(file_path, variable_name):
    """Executes a Python file and extracts a specific variable."""
    namespace = {}
    with open(file_path, "r") as f:
        exec(f.read(), {}, namespace)
    if variable_name not in namespace:
        raise ValueError(f"{variable_name} not found in {file_path}")
    return namespace[variable_name]


def lists_are_equal(list1, list2, tol=1e-6):
    """Compares two lists of tuples with float tolerance."""
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if abs(x - y) > tol:
                return False
    return True


def check_conversion_folder(conversion_path):
    gaze_path = os.path.join(conversion_path, "2D3DGazeLists.py")
    coords_path = os.path.join(conversion_path, "normalized_coords.py")

    if not (os.path.isfile(gaze_path) and os.path.isfile(coords_path)):
        return  # Skip if either file is missing

    try:
        gaze_data = extract_variable_from_py(gaze_path, "normalized_image_points_data")
        coords_data = extract_variable_from_py(coords_path, "normalized_coords")

        if not lists_are_equal(gaze_data, coords_data):
            # Extract P0XX folder name from the path
            parts = conversion_path.split(os.sep)
            p0_folder = next((p for p in parts if p.startswith("P0")), "UNKNOWN")
            print(f"{p0_folder} are not identical")

    except Exception as e:
        print(f"Error in {conversion_path}: {e}")


def find_conversion_folders(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d == "conversion":
                yield os.path.join(root, d)


def main():
    parser = argparse.ArgumentParser(description="Check matching of normalized coordinates.")
    parser.add_argument("--root_dir", help="Root directory to scan.")
    args = parser.parse_args()

    for conversion_folder in find_conversion_folders(args.root_dir):
        check_conversion_folder(conversion_folder)


if __name__ == "__main__":
    main()

import cv2
import os

def extract_frames(video_path, txt_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    time_windows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                label = parts[0]
                try:
                    start = float(parts[1])
                    end = float(parts[2])
                    time_windows.append((label, start, end))
                except ValueError:
                    print(f"Invalid time range skipped: {line}")
            else:
                print(f"Invalid format in: {line}")

    for label, start_sec, end_sec in time_windows:
        label_dir = os.path.join(output_path, label)
        os.makedirs(label_dir, exist_ok=True)

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                timestamp = round(frame_num / fps, 2)
                filename = f"{label}_timestamp{timestamp}.png"
                cv2.imwrite(os.path.join(label_dir, filename), frame)
            else:
                print(f"Could not read frame {frame_num} from {video_path}")

    cap.release()


def process_all_recordings(base_folder):
    to_do = {"P018", "P019", "P020", "P021", "P022"}
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path) and any(item.startswith(p) for p in to_do):
            # Now recursively search for subfolders containing both required files
            for root, dirs, files in os.walk(item_path):
                if "scenevideo.mp4" in files and "Event_time_ranges.txt" in files:
                    video_path = os.path.join(root, "scenevideo.mp4")
                    txt_path = os.path.join(root, "Event_time_ranges.txt")
                    output_path = os.path.join(root, "image_processing")

                    os.makedirs(output_path, exist_ok=True)
                    #print(f"Processing: {root}")
                    extract_frames(video_path, txt_path, output_path)
                    #print(f"Done with: {root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract labeled frames from scenevideo.mp4 in all P0 subfolders.")
    parser.add_argument("--base", "-b", required=True, help="Path to the base 'Recordings_static' folder.")
    args = parser.parse_args()

    process_all_recordings(args.base)

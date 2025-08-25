import cv2
import os
from pathlib import Path
import shutil

def extract_frames(video_path, txt_path, output_path, type):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if type == "static":
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
                    filename = f"{label}_timestamp{timestamp:.2f}.png"
                    cv2.imwrite(os.path.join(label_dir, filename), frame)
                else:
                    print(f"Could not read frame {frame_num} from {video_path}")
        
    elif type == "dynamic":
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    label = parts[0]
                    start = float(parts[1])
                    end = float(parts[2])
                else:
                    print(f"invalid format in: {line}, in {Path(txt_path).name}")

        new_end = round((end - 1.00), 2)
        #divide into 4 separate intervals to be grouped 
        time_range = round(((new_end - start) / 4), 2)
        first_end = round((start + time_range), 2)
        second_end = round((first_end + time_range), 2)
        third_end = round((second_end + time_range), 2)
        intervals = [(start, first_end), (first_end, second_end), (second_end, third_end), (third_end, new_end)]
        count = 0
        for entry in intervals:
            count += 1
            start_sec, end_sec = entry
            label_dir = os.path.join(output_path, f"{count}_interval")
            os.makedirs(label_dir, exist_ok=True)

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            for frame_num in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    timestamp = round(frame_num / fps, 2)
                    filename = f"{label}_timestamp{timestamp:.2f}.png"
                    cv2.imwrite(os.path.join(label_dir, filename), frame)
                else:
                    print(f"Could not read frame {frame_num} from {video_path}")

    cap.release()

def setTemplates(output_path):
    for subfolder in os.listdir(output_path):
        if "_interval" in subfolder:
            save_all_timestamps = []
            interval_path = os.path.join(output_path, subfolder)
            for image in os.listdir(interval_path):
                if image.endswith(".png"):
                    image_name = Path(image).name
                    timestamp = image_name.split("_timestamp")[1]
                    timestamp_number = timestamp.split(".png")[0]
                    save_all_timestamps.append(float(timestamp_number))
            #find image in the middle of the time range for that interval
            min_ts = min(save_all_timestamps)
            max_ts = max(save_all_timestamps)
            range = max_ts - min_ts
            template_ts = min_ts + (range / 2)
            closest = min(save_all_timestamps, key=lambda x: abs(x - template_ts))
            
            #copy template file in parent folder with name "template(number)"
            image_name = f"MM_timestamp{closest:.2f}.png"
            parent_path = Path(output_path).parent
            og_path = os.path.join(output_path, subfolder, image_name)
            interval = Path(subfolder).name
            interval_number = interval.split("_interval")[0]
            template_name = f"template{interval_number}.png"
            new_path = os.path.join(parent_path, template_name)

            #copy file into parent folder with new name
            shutil.copy(og_path, new_path)
            print(f"{og_path} was copied as {template_name}.")

def process_all_recordings(base_folder, type):
    to_do = {"P002", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P034", "P035", "P036"}
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
                    print(f"Processing: {root}")
                    extract_frames(video_path, txt_path, output_path, type)
                    #setTemplates(output_path)
                    print(f"Done with: {root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract labeled frames from scenevideo.mp4 in all P0 subfolders.")
    parser.add_argument("--base", "-b", required=True, help="Path to the base 'Recordings_static' folder.")
    parser.add_argument("--type", "-t", type=str, required=True, help="study type: static or dynamic")
    args = parser.parse_args()

    process_all_recordings(args.base, args.type)

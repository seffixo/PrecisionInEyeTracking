import cv2
import os
import argparse

'''
Extract frames from videos using txt-files. 
Parameters: 
    video_location: path where videos are located. 
    basic_path: basic output path for extracted frames. 
    time_path: location to all txt-files with corresponding file names for videos.

    content of txt-file: 
    LO,0.10,3.10
    MO,4.00,5.50
    RO,6.50,8.00
    ....

    LO representing the position of fixation point: links oben
    MO = mitte oben
    RO = rechts oben
'''

def extract_frames(video_path, txt_path, output_path):
    #open video and extract fps and total frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(f"Video-FPS: {fps} | Gesamtanzahl Frames: {total_frames}")

    #extract time_windows and format to extract frames
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
                    print(f"invalid time range skipped: {line}")
            else:
                print(f"invalid format in: {line}")

    #extract frames for every time_window
    for label, start_sec, end_sec in time_windows:
        #create specific output folder for label
        label_dir = os.path.join(output_path, label)
        os.makedirs(label_dir, exist_ok=True)

        #calculate frame-range and extract frames
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                exact_time = float(round((frame_num / fps),2))
                filename = f"{label}_timestamp{exact_time}.png"
                full_path = os.path.join(label_dir, filename)
                cv2.imwrite(full_path, frame)
            else:
                print(f"Frame {frame_num} could not be read.")

    cap.release()

def main(video_location, basic_path, time_path):
    #extract all videos with ".mp4" as file type
    videos = [f for f in os.listdir(video_location) if f.endswith(".mp4")]

    #loop through all videos in video_location folder
    for video_file in videos:
        #basename of video
        v_basename = os.path.splitext(video_file)[0]
        
        #create txt-path
        txt_file = v_basename + ".txt"
        txt_path = os.path.join(time_path, txt_file)
        
        if os.path.exists(txt_path):
            #both files exist
            video_path = os.path.join(video_location, video_file)
            print(f"Working on {video_file} with {txt_file}")

            #create output folder
            file_name = os.path.splitext(os.path.basename(video_file))[0]
            output_path = os.path.join(basic_path, file_name)
            os.makedirs(output_path, exist_ok=True)

            #extract time windows and corresponding frames
            extract_frames(video_path, txt_path, output_path)
            print(f"Video {video_file} has been extracted using {txt_file}.")
        else:
            print(f"Video {video_file} has been skipped.")

    print("Alle Frames wurden extrahiert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames using time ranges where participant was fixating certain points.")
    parser.add_argument("--video_path", "-vp", required=True, type=str, help="Path(s) to video file(s).")
    parser.add_argument("--time_ranges", "-t",required=True, help="TXT file, where all time_ranges and corresponding fixation point are saved per video.")
    parser.add_argument("--basic_path", "-bp", default="frames", help="Path to the output frame file(s).")

    args = parser.parse_args()
    main(args.video_path, args.basic_path, args.time_ranges)

# === Parameter ===
# video_path = r"SeffiLin1.mp4"             # Pfad zum Video
# time_ranges_file = r"zeitfenster.txt"     # txt-Datei mit Kürzel und Zeitfenstern
# output_root = "frames/" + str(video_path) # Hauptzielordner
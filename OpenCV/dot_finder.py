import cv2
import numpy as np
import os
import argparse
import sys

'''
Script to find fixation points (dots) in extracted video frames and saving their values to csv files. 
Parameter: 
    frame_path: path to video frames sorted in different subfolders. 
    template: extracted image from one fixation point (dot) to act as the sample of what we are trying to find (the dots).
    output_path: path to where csv files should be saved using a similiar folder-structure as in frame_path.

Methods: 
    check_n_del_images: preprocessing subfolders and checking if there are any files that are not images and deleting those. 
    find_marker_positions: using subfolder structure and checking every frame for fixation points and saving those to csv files.
'''

def check_n_del_images(video_title, template):
    #check if template is a .png file
    image_endings = ".png", ".jpeg", ".jpg"
    if not template.endswith(image_endings):
        print("Template image is not in a valid format! Script will be stopped.")
        sys.exit(1)

    #check if subfolders (location of fixated point) contain anything else than jpg, jpeg or png
    for subfolder in os.listdir(video_title):
        subfolder_path = os.path.join(video_title, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Durchlaufe alle Dateien im Unterordner
            for image in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image)
                
                if os.path.isfile(image_path):
                    _, ending = os.path.splitext(image)
                    ending = ending.lower()
                    
                    if ending not in image_endings:
                        print(f"Deleting: {image_path} - invalid format.")
                        os.remove(image_path)


def find_marker_positions(title_path, template_path, output_path, threshold=0.8):
    #working through subfolder structure and extracting frames and their fixation points
    for subfolder in title_path:
        for image in subfolder:
            # Lade Poster-Frame und Referenz-Template
            frame = cv2.imread(image)
            if frame is None:
                raise FileNotFoundError(f"Bild nicht gefunden: {image}")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_frame, w_frame = gray_frame.shape

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template nicht gefunden: {template_path}")
    h_temp, w_temp = template.shape

    # Template Matching durchführen
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    #print("locations:", locations)

    detected = []
    for pt in zip(*locations[::-1]):
        center = (pt[0] + w_temp // 2, pt[1] + h_temp // 2)

        # Doppelte Punkte vermeiden
        if not any(np.linalg.norm(np.array(center) - np.array(d)) < 10 for d in detected):
            detected.append(center)

    # Normierte Koordinaten berechnen
    norm_coords = [(x / w_frame, y / h_frame) for (x, y) in detected]
    rounded_coords = [(float(round(x, 4)), float(round(y, 4))) for x, y in norm_coords]

    # Treffer im Bild einzeichnen
    count = 1
    for (x, y) in detected:
        cv2.line(frame, (x - 5, y), (x + 5, y), (0, 255, 0), 1)  # horizontale Linie
        cv2.line(frame, (x, y - 5), (x, y + 5), (0, 255, 0), 1)  # vertikale Linie
        normx = float(round((x / w_frame), 4))
        normy = float(round((y / h_frame), 4))
        coords = "P" + str(count) + " " + str(normx) + "," + str(normy)
        cv2.putText(frame, text=coords, org=(x-50, y-15), fontScale=1, fontFace=2, color=(0,255,0), thickness=2)
        count = count+1

    # Bild speichern
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, frame)

    print(f"{len(norm_coords)} Treffer gefunden und gespeichert unter: {output_path}")
    print(rounded_coords)
    # Zielpfad zur Textdatei
    output = "koordinatenText.txt"

    # Schreiben in Datei
    with open(output, "w") as f:
        for x, y in rounded_coords:
            f.write(f"{x},{y}\n")

    return rounded_coords

def main(frame_path, template, output_path, threshold):
    video_titles = [n for n in os.listdir(frame_path)]

    for title in video_titles:
        #check if chosen video title has invalid images
        check_n_del_images(title, template)

        #find marker positions and save them according to the folder name
        find_marker_positions(title, template, output_path, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dots in extracted frame from video.")
    parser.add_argument("--frame_path", "-fp", required=True, type=str, help="Path(s) to frame file(s).")
    parser.add_argument("--template", "-t", default="template1.png", help="Template image for search of template in frame PNG.")
    parser.add_argument("--output_path", "-op", required=True, help="Path to the output PNG file.")

    args = parser.parse_args()
    threshold=0.8
    find_marker_positions(args.frame_path, args.template, args.output_path, threshold)



#frame_path = r"Lin1_MM_1675Fr.png"
#output_path = "Lin1MM1675Fr_dots.png"
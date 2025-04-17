import cv2
import numpy as np
import os
import argparse

def find_marker_positions(frame_path, template_path, output_path, threshold=0.8):
    """
    Findet alle Vorkommen eines Templates in einem Bild und speichert die Treffer visualisiert im Bild.

    :param frame_path: Pfad zum zu durchsuchenden Bild
    :param template_path: Pfad zum Referenzbild (Marker)
    :param output_path: Pfad zur gespeicherten Ausgabe mit Markierungen
    :param threshold: Matching-Schwelle (zwischen 0 und 1)
    :return: Liste der normierten Koordinaten [(x1, y1), (x2, y2), ...]
    """

    # Lade Poster-Frame und Referenz-Template
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {frame_path}")
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
    rounded_coords = [(float(round(x, 3)), float(round(y, 3))) for x, y in norm_coords]

    # Treffer im Bild einzeichnen
    for (x, y) in detected:
        cv2.line(frame, (x - 5, y), (x + 5, y), (0, 255, 0), 1)  # horizontale Linie
        cv2.line(frame, (x, y - 5), (x, y + 5), (0, 255, 0), 1)  # vertikale Linie

    # Bild speichern
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, frame)

    print(f"{len(norm_coords)} Treffer gefunden und gespeichert unter: {output_path}")
    print(rounded_coords)
    # Zielpfad zur Textdatei
    output = "koordinaten.txt"

    # Schreiben in Datei
    with open(output, "w") as f:
        for x, y in rounded_coords:
            f.write(f"{x},{y}\n")

    return rounded_coords

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
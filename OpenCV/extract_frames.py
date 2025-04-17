import cv2
import os

# === Parameter ===
video_path = r"SeffiLin1.mp4"     # Pfad zu deinem Video
frame_number = 1675                      # Nummer des Frames, den du extrahieren willst
output_path = r"Lin1_MM_1675Fr.png"      # Name der gespeicherten Bilddatei

# === Video öffnen ===
cap = cv2.VideoCapture(video_path)

# Zum gewünschten Frame springen
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Frame lesen
ret, frame = cap.read()

if ret:
    # Bild speichern
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_number} gespeichert als '{output_path}'")
else:
    print("Frame konnte nicht gelesen werden.")

cap.release()
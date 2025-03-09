import numpy as np
import argparse
import json
import os

def finde_naehesten_punkt(a, b, c, d):
    # Richtungsvektoren normalisieren
    b = b / np.linalg.norm(b)
    d = d / np.linalg.norm(d)
    
    # Vektor zwischen den Ursprüngen
    w0 = a - c
    
    # Koeffizienten berechnen
    a_dot_b = np.dot(b, b)
    d_dot_d = np.dot(d, d)
    b_dot_d = np.dot(b, d)
    w0_dot_b = np.dot(w0, b)
    w0_dot_d = np.dot(w0, d)
    
    # Determinante berechnen
    denom = a_dot_b * d_dot_d - b_dot_d ** 2
    
    # Falls denom ≈ 0, sind die Blickstrahlen fast parallel
    if abs(denom) < 1e-6:
        print("Die Blickstrahlen sind nahezu parallel.")
        return None
    
    # Parameter für die kürzesten Punkte berechnen
    lambda_wert = (b_dot_d * w0_dot_d - d_dot_d * w0_dot_b) / denom
    mu_wert = (a_dot_b * w0_dot_d - b_dot_d * w0_dot_b) / denom
    
    # Punkte auf den Blickstrahlen berechnen
    p1 = a + lambda_wert * b
    p2 = c + mu_wert * d
    
    # Mittelpunkt der kürzesten Verbindung als Schnittpunkt
    schnittpunkt = (p1 + p2) / 2
    #print(f"Geschätzter Schnittpunkt: {schnittpunkt}")
    return schnittpunkt

def load_json_data(file_path):
    """Lädt die JSON-Daten aus einer Datei und gibt die ersten Werte zurück."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def add_schnittpunkt_to_json(data):
    """Berechnet den Schnittpunkt für jedes Datenelement und fügt ihn dem JSON hinzu."""
    for entry in data:
        timestamp = entry['timestamp']
        # Extrahiere die Ursprungspunkte und Richtungsvektoren
        eye_left_origin = np.array(entry['eyeleft']['gazeorigin'])
        eye_left_direction = np.array(entry['eyeleft']['gazedirection'])
        eye_right_origin = np.array(entry['eyeright']['gazeorigin'])
        eye_right_direction = np.array(entry['eyeright']['gazedirection'])

        # Berechne den Schnittpunkt
        schnittpunkt = finde_naehesten_punkt(eye_right_origin, eye_right_direction, eye_left_origin, eye_left_direction)

        # Wenn ein Schnittpunkt berechnet wurde, füge ihn unter 'gaze3d' hinzu
        if schnittpunkt is not None:
            entry['gaze3d'] = list(entry['gaze3d'])  # Sicherstellen, dass gaze3d eine Liste ist
            entry['schnittpunkt'] = schnittpunkt.tolist()  # Hinzufügen des Schnittpunkts
        #print(f"entry at {timestamp} gave this schnittpunkt {schnittpunkt}")

        remove_eye_entries(entry)
        
    return data

def remove_eye_entries(entry):
    """Removes 'eyeleft' and 'eyeright' entries from all data entries."""
    if 'eyeleft' in entry:
        del entry['eyeleft']
    if 'eyeright' in entry:
        del entry['eyeright']

def save_json_data(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Filtered data successfully saved to {output_path}")

def main():
    # Argumente parsen
    parser = argparse.ArgumentParser(description="Berechne den Schnittpunkt der Blickstrahlen und speichere das Ergebnis.")
    parser.add_argument("--filtered_json", "-fj", required=True, help="Path to the filtered JSON file.")
    parser.add_argument("--output", "-op", default="schnittpunkt_ohne_eyes.json", help="Path to the output analysis file.")
    args = parser.parse_args()
    
    # JSON-Daten laden
    data = load_json_data(args.filtered_json)
    
    # Schnittpunkt berechnen und zu den Daten hinzufügen
    updated_data = add_schnittpunkt_to_json(data)

    # Die modifizierten Daten in einer neuen Datei speichern
    file_name = os.path.splitext(os.path.basename(args.filtered_json))[0]
    output_path = "OutputFiles/" + file_name + "_" + args.output
    save_json_data(updated_data, output_path)

if __name__ == "__main__":
    main()

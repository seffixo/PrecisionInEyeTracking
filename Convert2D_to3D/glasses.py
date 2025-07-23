import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --- Konfiguration: Ihre optimalen Parameter ---
DEFAULT_CENTER_X = 0.4976
DEFAULT_CENTER_Y = 0.4790
DEFAULT_SCALE_X = 16
DEFAULT_SCALE_Y = 9
DEFAULT_DIST_DIVISOR = 18.357
DEFAULT_POLY_COEFFS_CORRECTION = np.array([-0.6,-1.1, 2.9, -0.012, 1.0]) # p[0]d^3 .. p[3]d^0

SLOPE_X_FROM_RANSAC = -2.1053
INTERCEPT_X_FROM_RANSAC = 1.0477
SLOPE_Y_FROM_RANSAC = -1.1851
INTERCEPT_Y_FROM_RANSAC = 0.5616

GAZEDATA_FILE = 'gazedata' # Name Ihrer Datendatei

# --- Hilfsfunktionen (aus dem vorherigen Skript) ---
def normalize_d3_raw(d3_raw_x, d3_raw_y, d3_raw_z):
    d3_raw_x_arr, d3_raw_y_arr, d3_raw_z_arr = np.atleast_1d(d3_raw_x, d3_raw_y, d3_raw_z)
    magnitude = np.sqrt(d3_raw_x_arr**2 + d3_raw_y_arr**2 + d3_raw_z_arr**2)
    x3d_norm, y3d_norm, z3d_norm = (np.full_like(arr, np.nan, dtype=np.float64) for arr in (d3_raw_x_arr, d3_raw_y_arr, d3_raw_z_arr))
    non_zero_mag_mask = ~np.isclose(magnitude, 0)
    if np.any(non_zero_mag_mask): # Nur dividieren, wenn es nicht-null Magnituden gibt
        x3d_norm[non_zero_mag_mask] = d3_raw_x_arr[non_zero_mag_mask] / magnitude[non_zero_mag_mask]
        y3d_norm[non_zero_mag_mask] = d3_raw_y_arr[non_zero_mag_mask] / magnitude[non_zero_mag_mask]
        z3d_norm[non_zero_mag_mask] = d3_raw_z_arr[non_zero_mag_mask] / magnitude[non_zero_mag_mask]
    if np.isscalar(d3_raw_x): return x3d_norm[0], y3d_norm[0], z3d_norm[0]
    return x3d_norm, y3d_norm, z3d_norm

def calculate_distance_and_correction_factor(
    d2_values_x, d2_values_y, center_x, center_y, scale_x, scale_y, dist_divisor, poly_coeffs
):
    dist_x_scaled = (center_x - d2_values_x) * scale_x
    dist_y_scaled = (center_y - d2_values_y) * scale_y
    distance_from_center = np.sqrt(dist_x_scaled**2 + dist_y_scaled**2) / dist_divisor
    correction_factor = np.polyval(poly_coeffs, distance_from_center)
    return distance_from_center, correction_factor

def convert_d2_and_raw_d3_to_corrected_d3s(
    d2_x, d2_y, d3_raw_x, d3_raw_y, d3_raw_z,
    center_x=DEFAULT_CENTER_X, center_y=DEFAULT_CENTER_Y,
    scale_x=DEFAULT_SCALE_X, scale_y=DEFAULT_SCALE_Y,
    dist_divisor=DEFAULT_DIST_DIVISOR, poly_coeffs_correction=DEFAULT_POLY_COEFFS_CORRECTION
):
    x3d_norm, y3d_norm, _ = normalize_d3_raw(d3_raw_x, d3_raw_y, d3_raw_z)
    _, correction_factor = calculate_distance_and_correction_factor(
        d2_x, d2_y, center_x, center_y, scale_x, scale_y, dist_divisor, poly_coeffs_correction
    )
    x3d_corrected = x3d_norm * correction_factor
    y3d_corrected = y3d_norm * correction_factor
    return x3d_corrected, y3d_corrected

def corrected_d3_to_d2(d3_corrected, slope, intercept):
    if np.isclose(slope, 0):
        return np.full_like(d3_corrected, np.nan, dtype=np.float64) if isinstance(d3_corrected, np.ndarray) else np.nan
    return (d3_corrected - intercept) / slope

# --- Hauptskript ---
def main():
    # 1. Daten einlesen und parsen
    print(f"Lese Daten aus '{GAZEDATA_FILE}'...")
    extracted_data = []
    try:
        with open(GAZEDATA_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    timestamp = record.get("timestamp")
                    gaze_data_field = record.get("data", {}) # "data" ist das Feld mit gaze2d, gaze3d etc.
                    
                    gaze2d_actual = gaze_data_field.get("gaze2d")
                    gaze3d_raw_vec = gaze_data_field.get("gaze3d") # Dies sollte ein 3-Element-Array sein

                    entry = {
                        "timestamp": timestamp,
                        "x2d_actual": None, "y2d_actual": None,
                        "d3_raw_x": None, "d3_raw_y": None, "d3_raw_z": None,
                    }

                    if gaze2d_actual and len(gaze2d_actual) >= 2:
                        entry["x2d_actual"] = gaze2d_actual[0]
                        entry["y2d_actual"] = gaze2d_actual[1]
                    
                    if gaze3d_raw_vec and len(gaze3d_raw_vec) >= 3:
                        entry["d3_raw_x"] = gaze3d_raw_vec[0]
                        entry["d3_raw_y"] = gaze3d_raw_vec[1]
                        entry["d3_raw_z"] = gaze3d_raw_vec[2]
                    
                    # Nur Einträge mit vollständigen Daten hinzufügen
                    if all(entry[k] is not None for k in ["x2d_actual", "y2d_actual", "d3_raw_x", "d3_raw_y", "d3_raw_z"]):
                        extracted_data.append(entry)

                except json.JSONDecodeError:
                    print(f"Warnung: Konnte JSON nicht dekodieren: {line.strip()}")
                except (TypeError, IndexError) as e:
                    print(f"Warnung: Fehlende Daten oder falsches Format in Zeile: {line.strip()} - Fehler: {e}")
    except FileNotFoundError:
        print(f"Fehler: Datei '{GAZEDATA_FILE}' nicht gefunden.")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist beim Lesen der Datei aufgetreten: {e}")
        return

    if not extracted_data:
        print("Keine gültigen Daten zum Verarbeiten gefunden.")
        return

    df_gaze = pd.DataFrame(extracted_data)
    print(f"{len(df_gaze)} gültige Datenpunkte geladen.")

    # 2. Vorwärtsrechnung: 2D-Vorhersagen erstellen
    print("Berechne 2D-Vorhersagen basierend auf dem Modell...")
    
    # Vektorisierte Anwendung der Korrektur
    x3d_corrected, y3d_corrected = convert_d2_and_raw_d3_to_corrected_d3s(
        df_gaze['x2d_actual'], df_gaze['y2d_actual'],
        df_gaze['d3_raw_x'], df_gaze['d3_raw_y'], df_gaze['d3_raw_z']
    )
    df_gaze['x3d_corrected'] = x3d_corrected
    df_gaze['y3d_corrected'] = y3d_corrected

    df_gaze['x2d_predicted'] = corrected_d3_to_d2(df_gaze['x3d_corrected'], SLOPE_X_FROM_RANSAC, INTERCEPT_X_FROM_RANSAC)
    df_gaze['y2d_predicted'] = corrected_d3_to_d2(df_gaze['y3d_corrected'], SLOPE_Y_FROM_RANSAC, INTERCEPT_Y_FROM_RANSAC)

    # Entferne Zeilen, wo Vorhersagen NaN sein könnten (z.B. durch Division durch Null in Korrektur/Normalisierung)
    df_gaze.dropna(subset=['x2d_predicted', 'y2d_predicted'], inplace=True)
    if df_gaze.empty:
        print("Nach der Berechnung der Vorhersagen sind keine gültigen Datenpunkte mehr übrig.")
        return
    print(f"{len(df_gaze)} Datenpunkte nach Vorhersage und NaN-Filterung.")


    # 3. Differenzen berechnen
    df_gaze['error_x'] = df_gaze['x2d_actual'] - df_gaze['x2d_predicted']
    df_gaze['error_y'] = df_gaze['y2d_actual'] - df_gaze['y2d_predicted']
    df_gaze['euclidean_error'] = np.sqrt(df_gaze['error_x']**2 + df_gaze['error_y']**2)

    print("\nStatistiken der Fehler (in Einheiten der 2D-Koordinaten):")
    print("Fehler X (actual - predicted):")
    print(df_gaze['error_x'].describe())
    print("\nFehler Y (actual - predicted):")
    print(df_gaze['error_y'].describe())
    print("\nEuklidischer Fehler:")
    print(df_gaze['euclidean_error'].describe())

    # 4. Visualisierung der Differenzen
    print("\nErstelle Plots...")

    # Plot 1: Scatterplot der Fehler (error_x vs error_y)
    plt.figure(figsize=(8, 8))
    plt.scatter(df_gaze['error_x'], df_gaze['error_y'], alpha=0.3, s=10, edgecolors='none')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
    max_abs_error = max(df_gaze['error_x'].abs().max(), df_gaze['error_y'].abs().max()) * 1.1
    if pd.notna(max_abs_error) and max_abs_error > 1e-6 : # Nur setzen, wenn sinnvoll
        plt.xlim(-max_abs_error, max_abs_error)
        plt.ylim(-max_abs_error, max_abs_error)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('2D-Vorhersagefehler (Actual - Predicted)')
    plt.xlabel('Fehler in X-Koordinate')
    plt.ylabel('Fehler in Y-Koordinate')
    plt.grid(True, linestyle=':', alpha=0.7)
    # Formatter für Achsen, um wissenschaftliche Notation bei sehr kleinen Zahlen zu vermeiden
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2,3)) # Schaltet auf wiss. Notation außerhalb dieses Bereichs
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()

    # Plot 2: Histogramme der Fehler
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    common_bins = 50

    axs[0].hist(df_gaze['error_x'], bins=common_bins, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0].set_title('Histogramm: Fehler in X')
    axs[0].set_xlabel('Fehler X (actual - predicted)')
    axs[0].set_ylabel('Häufigkeit')
    axs[0].grid(True, linestyle=':', alpha=0.5)
    axs[0].xaxis.set_major_formatter(formatter)


    axs[1].hist(df_gaze['error_y'], bins=common_bins, color='lightcoral', edgecolor='black', alpha=0.7)
    axs[1].set_title('Histogramm: Fehler in Y')
    axs[1].set_xlabel('Fehler Y (actual - predicted)')
    axs[1].grid(True, linestyle=':', alpha=0.5)
    axs[1].xaxis.set_major_formatter(formatter)


    axs[2].hist(df_gaze['euclidean_error'], bins=common_bins, color='lightgreen', edgecolor='black', alpha=0.7)
    axs[2].set_title('Histogramm: Euklidischer Fehler')
    axs[2].set_xlabel('Euklidischer Fehler sqrt(err_x^2 + err_y^2)')
    axs[2].grid(True, linestyle=':', alpha=0.5)
    axs[2].xaxis.set_major_formatter(formatter)

    fig.suptitle('Verteilung der Vorhersagefehler', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Platz für suptitle
    plt.show()

    # Optional: Plot Fehler vs. Distanz zum Zentrum (wenn 'distance_from_center' berechnet wurde)
    # Hierfür müssten wir 'distance_from_center' für alle Punkte in df_gaze berechnen
    df_gaze['distance_from_center'], _ = calculate_distance_and_correction_factor(
        df_gaze['x2d_actual'], df_gaze['y2d_actual'],
        DEFAULT_CENTER_X, DEFAULT_CENTER_Y, DEFAULT_SCALE_X, DEFAULT_SCALE_Y,
        DEFAULT_DIST_DIVISOR, DEFAULT_POLY_COEFFS_CORRECTION # Poly coeffs hier nicht relevant für Distanz
    )
    plt.figure(figsize=(10, 6))
    plt.scatter(df_gaze['distance_from_center'], df_gaze['euclidean_error'], alpha=0.3, s=10, edgecolors='none')
    plt.title('Euklidischer Fehler vs. Distanz zum Zentrum (basierend auf Actual 2D)')
    plt.xlabel('Distanz zum Zentrum (Ihre Formel)')
    plt.ylabel('Euklidischer Fehler')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

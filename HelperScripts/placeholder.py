import os
import json

# Folder to scan
folder_path = "D:\\WorkingFolder_PythonD\\2Dto3D_Conversion\\581_dynam"  # <-- Change this to your folder path

results = {}

# Loop over files in the directory
for filename in os.listdir(folder_path):
    #print(f"{filename}")
    if filename.endswith("_ConvResults.json"):
        key = filename.replace("_ConvResults.json", "")  # Remove the suffix
        json_path = os.path.join(folder_path, filename)
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if "globalDiff" in data:
                    value = data["globalDiff"]  # Get absolute value
                    value = float(value)
                    results[key] = abs(value)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to read {filename}: {e}")

# Find the smallest value
if results:
    min_key = min(results, key=results.get)
    print(f"Smallest globalDiff: {min_key}: {results[min_key]}")
else:
    print(f"No valid '_ConvResults.json' files with 'gobalDiff' found. {folder_path}")

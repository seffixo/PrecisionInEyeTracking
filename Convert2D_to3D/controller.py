import subprocess 
from pathlib import Path
import sys

PY = sys.executable

# Step 1: Base directory
root_dir = Path(r"D:\WorkingFolder_PythonD\2Dto3D_Conversion\521_dynam").resolve()

# Step 2: Loop through each "P0xx_statisch" folder
for stat_dir in root_dir.glob("P0*_dynamisch"):
    if not stat_dir.is_dir():
        continue

    # Step 3: Loop through each subfolder (e.g. P0xx_80_stat_3lights)
    for subfolder in stat_dir.iterdir():
        if subfolder.is_dir():
            paramFolder = str(subfolder.relative_to(root_dir)).replace("\\", "/")

            print(f"Running checkParam_3dto2d with paramFolder: {paramFolder}")

            subprocess.run([
                PY, "checkParam_3dto2d.py",
                "--root_dir", str(root_dir),
                "--camParaFolder", paramFolder
            ])

            print("first script is done")

            #replace aspects of subfolder name to shorten the file name with convResults.json to know which camParameter were used
            name = subfolder.name  # Just the folder name
            name = name.replace('_stat', '')
            name = name.replace('3lights', '3L')
            name = name.replace('basicL', 'bL')
            result_name = name + "_ConvResults.json"

            subprocess.run([
                PY, "checkConversion.py",
                "--root_dir", str(root_dir),
                "--result_name", result_name
            ])

            print("second script is done")
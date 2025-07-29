import os
import json

def clean_gaze_data_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "separated_time_gazedata":
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    input_path = os.path.join(dirpath, filename)
                    output_path = os.path.join(dirpath, filename.replace(".jsonl", "_cleaned.jsonl"))

                    try:
                        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                            for line in infile:
                                try:
                                    record = json.loads(line)
                                    # Extract only desired fields
                                    cleaned_record = {
                                        "timestamp": record.get("timestamp"),
                                        "data": {
                                            "gaze2d": record["data"].get("gaze2d"),
                                            "gaze3d": record["data"].get("gaze3d"),
                                        }
                                    }
                                    json.dump(cleaned_record, outfile)
                                    outfile.write("\n")
                                except Exception as parse_err:
                                    print(f"Error parsing line in {filename}: {parse_err}")
                    except Exception as file_err:
                        print(f"Error processing {filename}: {file_err}")

# Run the script
if __name__ == "__main__":
    root_directory = "..\\..\\WorkingFolder_Python\\Conv2D_to3D\\581_stat_conv"
    clean_gaze_data_files(root_directory)

import os

from collections import defaultdict

# Dictionary to store smallest differences
smallest_diffs = defaultdict(lambda: float('inf'))

# Root directory to start searching
root_dir = './Recordings_static'  # Change this to your root directory if needed

# Traverse directories
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "Event_time_ranges.txt" in filenames:
        file_path = os.path.join(dirpath, "Event_time_ranges.txt")
        folder_name = os.path.basename(dirpath)
        
        # Extract participant key (e.g., "P033")
        participant_key = folder_name.split("_")[0]

        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse start times
        try:
            start_times = [float(line.split(',')[0]) for line in lines if line.strip()]
        except ValueError:
            continue  # Skip files with malformed lines

        # Calculate differences
        if len(start_times) > 1:
            start_times.sort()
            diffs = [j - i for i, j in zip(start_times[:-1], start_times[1:])]
            min_diff = min(diffs)
            
            # Store the smallest difference found so far for the participant
            if min_diff < smallest_diffs[participant_key]:
                smallest_diffs[participant_key] = min_diff

# Write results to file
with open('smallest_time_diffs.txt', 'w') as out_file:
    for participant, diff in sorted(smallest_diffs.items()):
        out_file.write(f"{participant}: {diff:.3f}\n")

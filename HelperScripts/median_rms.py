#!/usr/bin/env python3
"""
Compute median RMS by (distance, lighting, label), optionally filtered by participants.

Usage:
  python compute_median_rms_with_participants.py -i /path/to/input.csv -o /path/to/output.tsv \
      --participants P010 P008 P007

The output TSV has columns: distance, lighting, label, median_rms
"""
import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Compute median RMS by (distance, lighting, label) and save as TSV.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output TSV file.")
    parser.add_argument("--participants", nargs="+", help="Optional list of participant IDs to include (e.g., P001 P002).")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(in_path)

    # Validate required columns
    required_cols = {"distance", "lighting", "label", "rms", "participant_id"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(2)

    # Optional filter by participants
    if args.participants:
        df = df[df["participant_id"].isin(args.participants)]
        if df.empty:
            print("No matching participants found in the data.", file=sys.stderr)
            sys.exit(3)

    # Compute median
    result = (
        df.groupby(["distance", "lighting", "label"], as_index=False)["rms"]
          .median()
          .rename(columns={"rms": "median_rms"})
        [["distance", "lighting", "label", "median_rms"]]
    )

    # Save TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)

    print(f"Saved {len(result)} rows to {out_path}")

if __name__ == "__main__":
    main()

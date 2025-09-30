#!/usr/bin/env python3
import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

LABEL_ORDER = ["LU", "MU", "RU", "LM", "MM", "RM", "LD", "MD", "RD"]

TEST_CASE_MAP = {
    ("80", "bL"): 1,
    ("80", "3L"): 2,
    ("120", "bL"): 3,
    ("120", "3L"): 4,
    ("180", "bL"): 5,
    ("180", "3L"): 6,
}

def fmt(x: float) -> str:
    try:
        return f"{float(x):.2f}°"
    except Exception:
        return str(x)

def build_table(df: pd.DataFrame, lighting: str, angle_prefix: str) -> (str, int):
    sub = df[df["lighting"].astype(str) == lighting].copy()

    values = {lab: {"median": None, "whisker_low": None, "whisker_high": None} for lab in LABEL_ORDER}
    sub["label"] = sub["label"].astype(str)

    for _, row in sub.iterrows():
        lab = row["label"]
        if lab in values:
            values[lab]["median"] = row["median"]
            values[lab]["whisker_low"] = row["whisker_low"]
            values[lab]["whisker_high"] = row["whisker_high"]

    header_cols = " & " + " & ".join(LABEL_ORDER) + " \\\\ \\hline"
    med_row = "medAE & " + " & ".join(fmt(values[lab]["median"]) if values[lab]["median"] is not None else "-" for lab in LABEL_ORDER) + "\\\\"
    min_row = "min   & " + " & ".join(fmt(values[lab]["whisker_low"]) if values[lab]["whisker_low"] is not None else "-" for lab in LABEL_ORDER) + "\\\\"
    max_row = "max   & " + " & ".join(fmt(values[lab]["whisker_high"]) if values[lab]["whisker_high"] is not None else "-" for lab in LABEL_ORDER) + "\\\\"

    test_case_num = TEST_CASE_MAP.get((angle_prefix, lighting))
    if test_case_num is None:
        raise ValueError(f"Could not determine Test Case number for angle '{angle_prefix}' and lighting '{lighting}'.")

    table = []
    table.append("\\begin{tabular}{l|ccccccccc}")
    table.append(f"      {header_cols}")
    table.append(f"{med_row}")
    table.append(f"{min_row}")
    table.append(f"{max_row}")
    table.append("\\end{tabular}")
    table.append(f"\\caption{{Test Case {test_case_num}:}}")

    return "\n".join(table), test_case_num

def detect_angle_prefix(filename: str) -> str:
    m = re.match(r"^(80|120|180)", Path(filename).name)
    if not m:
        raise ValueError(f"Filename must start with 80, 120, or 180: {filename}")
    return m.group(1)

def process_file(path: Path) -> List[Path]:
    df = pd.read_csv(path, sep="\t")
    required = {"lighting", "label", "median", "whisker_low", "whisker_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")

    angle = detect_angle_prefix(path.name)

    outputs = []
    for lighting in ["bL", "3L"]:
        if lighting in set(df["lighting"].astype(str)):
            latex, case_num = build_table(df, lighting, angle)
            outpath = path.parent / f"Test Case {case_num}.txt"
            outpath.write_text(latex, encoding="utf-8")
            outputs.append(outpath)
    return outputs

def expand_inputs(paths):
    results = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            for cand in sorted(P.glob("*.tsv")):
                if cand.name.startswith(("80", "120", "180")):
                    results.append(cand)
        else:
            results.append(P)
    return results

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate LaTeX tables for medAE/min/max from TSV stats files.")
    ap.add_argument("paths", nargs="+", help="Input TSV files or directories. If a directory is given, scan for TSVs starting with 80/120/180 (non-recursive).")
    args = ap.parse_args(argv)

    inputs = expand_inputs(args.paths)
    if not inputs:
        print("No matching TSV files found.", file=sys.stderr)
        return 1

    all_outputs = []
    for f in inputs:
        if not f.exists():
            print(f"Warning: {f} not found; skipping.", file=sys.stderr)
            continue
        try:
            outs = process_file(f)
            for o in outs:
                print(f"Wrote: {o}")
                all_outputs.append(o)
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)

    return 0 if all_outputs else 1

if __name__ == "__main__":
    raise SystemExit(main())

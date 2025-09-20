#!/usr/bin/env python3
"""
Compare two eye-tracking approaches from CSV files and save line & scatter plots.

Inputs:
    - CSV #1: first approach (must include columns: camera_id, participant_id, label, distance, lighting, angular_error, rms)
    - CSV #2: second approach (same columns)

Outputs (saved in current working directory by default):
    Aggregate:
        - line_accuracy.png        : Median angular_error per label for both approaches
        - line_precision.png       : Median rms per label for both approaches
        - scatter_accuracy.png     : Point-by-point angular_error comparison (approach1 vs approach2)
        - scatter_precision.png    : Point-by-point rms comparison (approach1 vs approach2)
    Per combination (lighting, distance, label):
        - line_accuracy_<label>_dist<distance>_light<lighting>.png
        - line_precision_<label>_dist<distance>_light<lighting>.png
        - scatter_accuracy_<label>_dist<distance>_light<lighting>.png
        - scatter_precision_<label>_dist<distance>_light<lighting>.png

Usage:
    python compare_eye_tracking.py --first_csv path/to/first.csv --second_csv path/to/second.csv [--group-by label] [--outdir .]

Notes:
    - Uses only matplotlib (no seaborn).
    - Colors are left to matplotlib defaults.
"""
import argparse
import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GROUP_BY = ["label"]  # for aggregate line plots
LUX_MAP = {"3L": "700 Lux", "bL": "300 Lux"}
TITLE_FONTSIZE = 10
CAMERA = "581s"


def ensure_required_columns(df: pd.DataFrame, name: str) -> None:
    required = {"camera_id", "participant_id", "label", "distance", "lighting", "angular_error", "rms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_data(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_required_columns(df, name)
    # Normalize dtypes to avoid merge surprises (e.g., distance int vs str)
    df["distance"] = df["distance"].astype(str)
    df["lighting"] = df["lighting"].astype(str)
    df["label"] = df["label"].astype(str)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def label_order_from_data(df: pd.DataFrame) -> List[str]:
    preferred = ["LD", "LM", "LU", "MD", "MM", "MU", "RD", "RM", "RU"]
    labels = df["label"].dropna().astype(str).unique().tolist()
    if all(l in preferred for l in labels):
        order = [l for l in preferred if l in labels]
    else:
        order = sorted(labels)
    return order


def aggregate_for_lineplot(df: pd.DataFrame, group_by: List[str], value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_by, dropna=False, as_index=False)[value_col]
          .median()
          .sort_values(group_by)
    )
    return grouped


def line_plot(df1: pd.DataFrame, df2: pd.DataFrame, group_by: List[str], value_col: str, out_path: str) -> None:
    if len(group_by) != 1:
        raise ValueError("For line plots, please pass exactly one column to --group-by (e.g., --group-by label).")
    x_key = group_by[0]

    g1 = aggregate_for_lineplot(df1, group_by, value_col)
    g2 = aggregate_for_lineplot(df2, group_by, value_col)

    if x_key == "label":
        order = label_order_from_data(pd.concat([df1[[x_key]], df2[[x_key]]], ignore_index=True))
        g1 = g1.set_index(x_key).reindex(order).reset_index()
        g2 = g2.set_index(x_key).reindex(order).reset_index()

    fig = plt.figure()
    plt.plot(g1[x_key], g1[value_col], marker="o", label="My_Approach")
    plt.plot(g2[x_key], g2[value_col], marker="o", label="glassesValidator")
    plt.xlabel(x_key)
    plt.ylabel(value_col)
    plt.title(f"{CAMERA} Median {value_col} by {x_key}", fontsize=TITLE_FONTSIZE)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def scatter_plot(df1: pd.DataFrame, df2: pd.DataFrame, value_col: str, out_path: str, title_suffix: str = "") -> None:
    keys = ["camera_id", "participant_id", "label", "distance", "lighting"]
    merged = pd.merge(
        df1[keys + [value_col]].rename(columns={value_col: f"{value_col}_mA"}),
        df2[keys + [value_col]].rename(columns={value_col: f"{value_col}_gV"}),
        on=keys,
        how="inner",
        suffixes=("_mA", "_gV"),
    )

    if merged.empty:
        # Nothing to plot
        return

    x = merged[f"{value_col}_mA"]
    y = merged[f"{value_col}_gV"]

    min_val = float(min(x.min(), y.min()))
    max_val = float(max(x.max(), y.max()))

    fig = plt.figure()
    plt.scatter(x, y, alpha=0.7, edgecolors="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel(f"{value_col} (My_Approach)")
    plt.ylabel(f"{value_col} (glassesValidator)")
    title = f"{value_col}: My_Approach vs glassesValidator - Camera {CAMERA}"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def all_points_plot_by_label(df1: pd.DataFrame, df2: pd.DataFrame, value_col: str, out_path: str, title_suffix: str = "") -> None:
    """
    Dot plot of ALL participant values (no aggregation) vs label.
    - Approach 1: blue circles
    - Approach 2: green triangles
    - Small horizontal jitter added so overlapping points are visible
    """
    # Consistent label order
    order = label_order_from_data(pd.concat([df1[["label"]], df2[["label"]]], ignore_index=True))
    label_to_x = {label: i for i, label in enumerate(order)}

    fig = plt.figure()
    ax = plt.gca()

    # Plot A1 as blue circles
    for _, row in df1.iterrows():
        x = label_to_x[row["label"]] + np.random.uniform(-0.1, 0.1)  # jitter
        ax.scatter(x, row[value_col], color="tab:blue", alpha=0.6, marker="o")

    # Plot A2 as green triangles
    for _, row in df2.iterrows():
        x = label_to_x[row["label"]] + np.random.uniform(-0.1, 0.1)  # jitter
        ax.scatter(x, row[value_col], color="tab:green", alpha=0.6, marker="^")

    # Axis ticks and labels
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("label")
    ax.set_ylabel(value_col)

    title = f"All participant {value_col} by label (no aggregation)"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=10)

    # Legend (only one entry per approach)
    ax.scatter([], [], color="tab:blue", marker="o", label="My_Approach")
    ax.scatter([], [], color="tab:green", marker="^", label="glassesValidator")
    ax.legend()

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_combo_plots(df1: pd.DataFrame, df2: pd.DataFrame, outdir: str) -> Tuple[int, int]:
    """
    For each (lighting, distance, label) combination present in the inner-join of both CSVs,
    create:
      - line plot of angular_error per participant (Approach 1 vs 2)
      - line plot of rms per participant (Approach 1 vs 2)
      - scatter plot of angular_error (A1 vs A2)
      - scatter plot of rms (A1 vs A2)
    Returns: (num_line_plots, num_scatter_plots)
    """
    keys = ["lighting", "distance", "label"]

    comb1 = df1[keys].drop_duplicates()
    comb2 = df2[keys].drop_duplicates()
    common = pd.merge(comb1, comb2, on=keys, how="inner")
    if common.empty:
        return (0, 0)

    line_count = 0
    scatter_count = 0

    for _, row in common.sort_values(keys).iterrows():
        lighting = str(row["lighting"])
        distance = str(row["distance"])
        label = str(row["label"])

        lux_str = LUX_MAP.get(lighting, None)
        lighting_label = f"{lighting} ({lux_str})" if lux_str else lighting
        title_suffix = f"{CAMERA} • {label} • {distance} cm • {lighting_label}"

        # Filter rows for this combo
        df1_c = df1[(df1["lighting"] == lighting) & (df1["distance"] == distance) & (df1["label"] == label)].copy()
        df2_c = df2[(df2["lighting"] == lighting) & (df2["distance"] == distance) & (df2["label"] == label)].copy()

        # Merge by participant to line-plot per participant
        pkeys = ["participant_id"]
        m_acc = pd.merge(
            df1_c[pkeys + ["angular_error"]].rename(columns={"angular_error": "angular_error_mA"}),
            df2_c[pkeys + ["angular_error"]].rename(columns={"angular_error": "angular_error_gV"}),
            on=pkeys, how="inner"
        )
        m_rms = pd.merge(
            df1_c[pkeys + ["rms"]].rename(columns={"rms": "rms_mA"}),
            df2_c[pkeys + ["rms"]].rename(columns={"rms": "rms_gV"}),
            on=pkeys, how="inner"
        )

        # Sorting participants
        def sort_participants(df):
            return df.sort_values("participant_id") if not df.empty else df

        m_acc = sort_participants(m_acc)
        m_rms = sort_participants(m_rms)

        # Line plot: accuracy
        if not m_acc.empty:
            fig = plt.figure()
            plt.plot(m_acc["participant_id"], m_acc["angular_error_mA"], marker="o", label="My_Approach")
            plt.plot(m_acc["participant_id"], m_acc["angular_error_gV"], marker="o", label="glassesValidator")
            plt.xlabel("participant_id")
            plt.ylabel("angular_error")
            plt.title(f"angular_error by participant — {title_suffix}", fontsize=TITLE_FONTSIZE)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = f"line_{CAMERA}_accuracy-{label}-{distance}-{lighting}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=150)
            plt.close(fig)
            line_count += 1

        # Line plot: precision (rms)
        if not m_rms.empty:
            fig = plt.figure()
            plt.plot(m_rms["participant_id"], m_rms["rms_mA"], marker="o", label="My_Approach")
            plt.plot(m_rms["participant_id"], m_rms["rms_gV"], marker="o", label="glassesValidator")
            plt.xlabel("participant_id")
            plt.ylabel("rms")
            plt.title(f"rms by participant — {title_suffix}", fontsize=TITLE_FONTSIZE)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = f"line_{CAMERA}_precision-{label}-{distance}-{lighting}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=150)
            plt.close(fig)
            line_count += 1

        # Scatter plots
        keys_full = ["camera_id", "participant_id", "label", "distance", "lighting"]
        merged_c = pd.merge(
            df1_c[keys_full + ["angular_error", "rms"]].rename(columns={
                "angular_error": "angular_error_mA",
                "rms": "rms_mA"
            }),
            df2_c[keys_full + ["angular_error", "rms"]].rename(columns={
                "angular_error": "angular_error_gV",
                "rms": "rms_gV"
            }),
            on=keys_full, how="inner"
        )

        if not merged_c.empty:
            # Accuracy scatter
            x = merged_c["angular_error_mA"]
            y = merged_c["angular_error_gV"]
            min_val = float(min(x.min(), y.min()))
            max_val = float(max(x.max(), y.max()))
            fig = plt.figure()
            plt.scatter(x, y, alpha=0.7, edgecolors="none")
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            plt.xlabel("angular_error (My_Approach)")
            plt.ylabel("angular_error (glassesValidator)")
            plt.title(f"angular_error: My_Approach vs glassesValidator — {title_suffix}", fontsize=TITLE_FONTSIZE)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            fname = f"scatter_{CAMERA}_accuracy-{label}-{distance}-{lighting}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=150)
            plt.close(fig)
            scatter_count += 1

            # Precision scatter
            x = merged_c["rms_mA"]
            y = merged_c["rms_gV"]
            min_val = float(min(x.min(), y.min()))
            max_val = float(max(x.max(), y.max()))
            fig = plt.figure()
            plt.scatter(x, y, alpha=0.7, edgecolors="none")
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            plt.xlabel("rms (My_Approach)")
            plt.ylabel("rms (glassesValidator)")
            plt.title(f"rms: My_Approach vs glassesValidator — {title_suffix}", fontsize=TITLE_FONTSIZE)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            fname = f"scatter_{CAMERA}_precision-{label}-{distance}-{lighting}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=150)
            plt.close(fig)
            scatter_count += 1

    return (line_count, scatter_count)


def main():
    parser = argparse.ArgumentParser(description="Compare two eye-tracking approaches and save plots.")
    parser.add_argument("--first_csv", type=str, required=True, help="Path to CSV for Approach 1")
    parser.add_argument("--second_csv", type=str, required=True, help="Path to CSV for Approach 2")
    parser.add_argument("--group-by", nargs="+", default=DEFAULT_GROUP_BY,
                        help="Columns for grouping in aggregate line plots (default: label). Must be exactly one column.")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory to save plots (default: current directory)")
    args = parser.parse_args()

    df1 = load_data(args.first_csv, "first_csv")
    df2 = load_data(args.second_csv, "second_csv")

    os.makedirs(args.outdir, exist_ok=True)

    # Aggregate plots
    line_accuracy_path = os.path.join(args.outdir, f"{CAMERA}_line_accuracy.png")
    line_precision_path = os.path.join(args.outdir, f"{CAMERA}_line_precision.png")
    line_plot(df1, df2, args.group_by, "angular_error", line_accuracy_path)
    line_plot(df1, df2, args.group_by, "rms", line_precision_path)

    scatter_accuracy_path = os.path.join(args.outdir, f"{CAMERA}_scatter_accuracy.png")
    scatter_precision_path = os.path.join(args.outdir, f"{CAMERA}_scatter_precision.png")
    scatter_plot(df1, df2, "angular_error", scatter_accuracy_path)
    scatter_plot(df1, df2, "rms", scatter_precision_path)

    # Per combo plots
    n_line, n_scatter = per_combo_plots(df1, df2, args.outdir)

    print("Saved aggregate plots:")
    print(" -", line_accuracy_path)
    print(" -", line_precision_path)
    print(" -", scatter_accuracy_path)
    print(" -", scatter_precision_path)

    print(f"Saved per-combination plots: {n_line} line plots, {n_scatter} scatter plots")
    print("Lighting codes mapped as:", LUX_MAP)

    # Extra: ALL participant values (no aggregation) by label
    allpoints_accuracy_path = os.path.join(args.outdir, f"{CAMERA}_allpoints_accuracy.png")
    allpoints_precision_path = os.path.join(args.outdir, f"{CAMERA}_allpoints_precision.png")
    all_points_plot_by_label(df1, df2, "angular_error", allpoints_accuracy_path)
    all_points_plot_by_label(df1, df2, "rms", allpoints_precision_path)
    print(" -", allpoints_accuracy_path)
    print(" -", allpoints_precision_path)


if __name__ == "__main__":
    main()

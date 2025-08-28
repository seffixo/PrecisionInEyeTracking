#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boxplot grid (labels × distance_light) for a chosen camera,
and export to Excel with the figure embedded.

Input CSV must be long-format with at least these columns:
- camera_id (int/str)
- participant_id (str/int)
- distance (e.g., 80, 120, 180)
- lighting (e.g., "3L", "bL")
- label (one of: MM, ML, MR, LU, MU, RU, LD, MD, RD)
- angular_error (float, degrees)

Usage:
    python boxgrid_to_excel.py input.csv --camera 521 --out boxplots_521.xlsx
"""

import io
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# ------------ configurable layout / ordering ------------
LABEL_ORDER = ["LD", "LM", "LU", "MD", "MM", "MU", "RD", "RM", "RU"]
DIST_LIGHT_ORDER = ["80-3L", "80-bL", "120-3L", "120-bL", "180-3L", "180-bL"]
LIGHT_ORDER = ["3L", "bL"]
DIST_ORDER = ["80", "120", "180"]

LABEL_MATRIX = [
    ["LU", "MU", "RU"],
    ["LM", "MM", "RM"],
    ["LD", "MD", "RD"],
]

TITLE_TEMPLATE = "Camera {camera} – Median Angular Error Distributions"
FIGSIZE = (16, 15)  # width, height in inches
DPI = 200         # saved image resolution

# --------------------------------------------------------

def add_distance_light_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # sanitize/normalize values
    df["distance"] = df["distance"].astype(str).str.strip()
    df["lighting"] = df["lighting"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["distance_light"] = df["distance"] + "-" + df["lighting"]
    return df

def _normalize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["label", "distance", "lighting"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def _compute_global_limits(vals: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(vals, [0.5, 99.5])
    pad = 0.12 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad

def plot_label_boxgrid(
    df: pd.DataFrame,
    camera_id,
    columns: str = "distance_light",   # "distance_light" | "lighting" | "distance"
    layout: str = "grid",              # "grid" | "matrix3x3"
    save_path_png: str | None = None,
    show_n: bool = True,
    show_median: bool = False,
    overlay_jitter: bool = False,
    jitter_alpha: float = 0.35,
    orientation: str = "horizontal"    # "horizontal" | "vertical"
) -> io.BytesIO:
    data = df[df["camera_id"].astype(str) == str(camera_id)].copy()
    if data.empty:
        raise ValueError(f"No rows found for camera_id={camera_id}")

    data = _normalize_core_columns(data)
    data = add_distance_light_column(data)

    all_vals = data["angular_error"].dropna().values
    if len(all_vals) == 0:
        raise ValueError("No angular_error values present.")
    lo, hi = _compute_global_limits(all_vals)
    locator = mticker.MaxNLocator(nbins=5)

    if layout == "matrix3x3":
        if columns != "lighting":
            raise ValueError("matrix3x3 layout is supported only with columns='lighting'.")
        return _plot_matrix3x3_by_lighting(
            data, camera_id, lo, hi, locator, save_path_png,
            show_n, show_median, overlay_jitter, jitter_alpha, orientation
        )

    # default rectangular grid
    return _plot_rectangular_grid(
        data, camera_id, columns, lo, hi, locator, save_path_png,
        show_n, show_median, overlay_jitter, jitter_alpha, orientation
    )

def _plot_rectangular_grid(
    data: pd.DataFrame,
    camera_id,
    columns: str,
    lo: float, hi: float, locator: mticker.MaxNLocator,
    save_path_png: str | None,
    show_n: bool, show_median: bool,
    overlay_jitter: bool, jitter_alpha: float,
    orientation: str
) -> io.BytesIO:

    labels = [lab for lab in LABEL_ORDER if lab in data["label"].unique()]

    if columns == "distance_light":
        col_key = "distance_light"
        cols = [dl for dl in DIST_LIGHT_ORDER if dl in data[col_key].unique()]
    elif columns == "lighting":
        col_key = "lighting"
        cols = [lt for lt in LIGHT_ORDER if lt in data[col_key].unique()]
    elif columns == "distance":
        col_key = "distance"
        cols = [d for d in DIST_ORDER if d in data[col_key].unique()]
    else:
        raise ValueError("columns must be one of: distance_light, lighting, distance")

    nrows, ncols = len(labels), len(cols)
    if nrows == 0 or ncols == 0:
        raise ValueError("No matching labels or column categories to plot.")

    share_args = {"sharex": orientation == "horizontal", "sharey": orientation == "vertical"}
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGSIZE, squeeze=False, **share_args)

    pretty_cols = " × ".join(["Labels", columns.replace("_", " ").title()])
    fig.suptitle(
        f"Camera {camera_id} – Median Angular Error Distributions ({pretty_cols})",
        fontsize=14, y=0.995
    )

    for r, lab in enumerate(labels):
        for c, colcat in enumerate(cols):
            ax = axes[r, c]
            subset = data[(data["label"] == lab) & (data[col_key] == colcat)]
            vals = subset["angular_error"].dropna().values
            n = len(vals)

            if n >= 2:
                ax.boxplot(
                    [vals], positions=[0], widths=0.6,
                    vert=(orientation == "vertical"),
                    showfliers=True, whis=1.5, patch_artist=True,
                )
                if overlay_jitter:
                    if orientation == "horizontal":
                        jy = (np.random.rand(n) - 0.5) * 0.3
                        ax.scatter(vals, jy, s=8, alpha=jitter_alpha)
                    else:
                        jx = (np.random.rand(n) - 0.5) * 0.3
                        ax.scatter(jx, vals, s=8, alpha=jitter_alpha)
            elif n == 1:
                if orientation == "horizontal":
                    ax.plot([vals[0]], [0], marker="o", ms=4)
                else:
                    ax.plot([0], [vals[0]], marker="o", ms=4)
            else:
                ax.text(0.5, 0.5, "–", ha="center", va="center",
                        transform=ax.transAxes, alpha=0.4)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
                if c == 0: ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=12, fontsize=10)
                if r == 0: ax.set_title(str(colcat), fontsize=10, pad=6)
                continue

            if orientation == "horizontal":
                ax.set_ylim(-1, 1); ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(locator); ax.set_yticks([])
            else:
                ax.set_xlim(-1, 1); ax.set_ylim(lo, hi); ax.yaxis.set_major_locator(locator); ax.set_xticks([])

            ax.grid(alpha=0.2, axis="x" if orientation == "horizontal" else "y")
            if c == 0: ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=12, fontsize=10)
            if r == 0: ax.set_title(str(colcat), fontsize=10, pad=6)
            if show_n:
                ax.text(0.02, 0.95, f"n={n}", transform=ax.transAxes, ha="left", va="top", fontsize=8, alpha=0.85)
            if show_median and n:
                med = np.median(vals)
                ax.text(0.98, 0.95, f"med={med:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight"); buf.seek(0); plt.close(fig)
    if save_path_png:
        with open(save_path_png, "wb") as f: f.write(buf.getbuffer())
    return buf

def _plot_matrix3x3_by_lighting(
    data: pd.DataFrame,
    camera_id,
    lo: float, hi: float, locator: mticker.MaxNLocator,
    save_path_png: str | None,
    show_n: bool, show_median: bool,
    overlay_jitter: bool, jitter_alpha: float,
    orientation: str
) -> io.BytesIO:

    lights_present = [lt for lt in LIGHT_ORDER if lt in data["lighting"].unique()]
    if not lights_present:
        raise ValueError("No lighting categories found in data for matrix3x3 layout.")

    n_blocks = len(lights_present)
    fig = plt.figure(figsize=(8 * n_blocks, 10))
    outer = gridspec.GridSpec(nrows=1, ncols=n_blocks, figure=fig, wspace=0.15)

    fig.suptitle(
        f"Camera {camera_id} – Median Angular Error Distributions (Labels 3×3 × Lighting x Distance)",
        fontsize=14, y=0.995
    )

    for b, lt in enumerate(lights_present):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 3, subplot_spec=outer[b], wspace=0.25, hspace=0.25
        )

        for r in range(3):
            for c in range(3):
                lab = LABEL_MATRIX[r][c]
                ax = fig.add_subplot(inner[r, c])

                subset = data[(data["label"] == lab) & (data["lighting"] == lt)]
                vals = subset["angular_error"].dropna().values
                n = len(vals)

                if n >= 2:
                    ax.boxplot(
                        [vals], positions=[0], widths=0.6,
                        vert=(orientation == "vertical"),
                        showfliers=True, whis=1.5, patch_artist=True,
                    )
                    if overlay_jitter:
                        if orientation == "horizontal":
                            jy = (np.random.rand(n) - 0.5) * 0.3
                            ax.scatter(vals, jy, s=8, alpha=jitter_alpha)
                        else:
                            jx = (np.random.rand(n) - 0.5) * 0.3
                            ax.scatter(jx, vals, s=8, alpha=jitter_alpha)
                elif n == 1:
                    if orientation == "horizontal":
                        ax.plot([vals[0]], [0], marker="o", ms=4)
                    else:
                        ax.plot([0], [vals[0]], marker="o", ms=4)
                else:
                    ax.text(0.5, 0.5, "–", ha="center", va="center",
                            transform=ax.transAxes, alpha=0.4)
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
                    continue

                if orientation == "horizontal":
                    ax.set_ylim(-1, 1); ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(locator); ax.set_yticks([])
                else:
                    ax.set_xlim(-1, 1); ax.set_ylim(lo, hi); ax.yaxis.set_major_locator(locator); ax.set_xticks([])

                ax.grid(alpha=0.2, axis="x" if orientation == "horizontal" else "y")

                ax.text(0.02, 0.95, lab, transform=ax.transAxes, ha="left", va="top", fontsize=9, alpha=0.9)
                if show_n:
                    ax.text(0.98, 0.95, f"n={n}", transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.85)
                if show_median and n:
                    med = np.median(vals)
                    ax.text(0.98, 0.80, f"med={med:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.96]) 

    #add per-block titles as figure text (so layout won't try to move them)
    dists = data["distance"].unique()
    dist_text = f"{dists[0]} cm" if len(dists) == 1 else ", ".join(sorted(dists))
    for b, lt in enumerate(lights_present):
        bb = outer[b].get_position(fig)  # bounding box of that sub-grid in figure coords
        cx = (bb.x0 + bb.x1) / 2.0       # center x of the block
        ty = bb.y1 + 0.012               # a bit above the block
        fig.text(cx, ty, f"{lt}  –  {dist_text}", ha="center", va="bottom", fontsize=12)

    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight"); buf.seek(0); plt.close(fig)
    if save_path_png:
        with open(save_path_png, "wb") as f: f.write(buf.getbuffer())
    return buf

def write_excel_with_figure(df: pd.DataFrame, png_buffer: io.BytesIO, out_path: str):
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Raw data
        df.to_excel(writer, sheet_name="Data", index=False)

        # Overview sheet with image
        workbook = writer.book
        ws = workbook.add_worksheet("Boxplot_Overview")
        writer.sheets["Boxplot_Overview"] = ws

        # A little caption
        caption_fmt = workbook.add_format({"bold": True})
        ws.write("A1", "Angular Error Box Plots by Label × Distance-Light", caption_fmt)
        ws.insert_image("A3", "boxgrid.png", {"image_data": png_buffer})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Path to long-format CSV.")
    parser.add_argument("--camera", required=True, help="Camera ID to filter (e.g., 521).")
    parser.add_argument("--out", default="boxplots.xlsx", help="Output Excel file path.")
    parser.add_argument("--save-png", default=None, help="(Optional) Also save the figure as PNG.")
    parser.add_argument("--distances", nargs="+", help='Distances to include (e.g., 80 120). If omitted, all distances are used.')
    parser.add_argument("--layout", choices=["grid", "matrix3x3"], default="grid",
        help="Plot layout: 'grid' = small multiples table (current). 'matrix3x3' = spatial label layout with one 3x3 block per lighting.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required_cols = {"camera_id", "participant_id", "label", "distance", "lighting", "angular_error", "matchings"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["distance"] = df["distance"].astype(str).str.strip()  # normalize just once

    if args.distances:
        allowed = {str(d).strip() for d in args.distances}
        df = df[df["distance"].isin(allowed)]
        if df.empty:
            raise ValueError(f"No rows left after filtering for distances={sorted(allowed)}")

    show_n = True
    show_median = False
    overlay_jitter = False
    jitter_alpha = 0.35
    orientation = "horizontal"
    columns = "lighting" # "distance_light" | "lighting" | "distance"
    camera_id = args.camera
    save_path_png = args.save_png
    layout = args.layout
        

    png_buf = plot_label_boxgrid(df, camera_id, columns, layout, save_path_png, show_n, show_median, overlay_jitter, jitter_alpha, orientation)
    write_excel_with_figure(df, png_buf, args.out)

    print(f"Done. Wrote: {args.out}")
    if args.save_png:
        print(f"Also saved PNG: {args.save_png}")

if __name__ == "__main__":
    main()

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
import json
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # must be before 'import matplotlib.pyplot as plt'
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import re
from pathlib import Path
from typing import NamedTuple
from matplotlib.cbook import boxplot_stats

# ------------ configurable layout / ordering ------------
LABEL_ORDER = ["LD", "LM", "LU", "MD", "MM", "MU", "RD", "RM", "RU"]
DYNAM_LABEL = ["MM"]
DIST_LIGHT_ORDER = ["80-3L", "80-bL", "120-3L", "120-bL", "180-3L", "180-bL"]
LIGHT_ORDER = ["3L", "bL"]
LIGHT_MAP = {"3L": "700 Lux", "bL": "300 Lux"}
DIST_ORDER = ["80", "120", "180"]

LABEL_MATRIX = [
    ["LU", "MU", "RU"],
    ["LM", "MM", "RM"],
    ["LD", "MD", "RD"],
]

LIGHT_SYNONYMS = {
    "3l": "3L", "3light": "3L", "3lights": "3L",
    "bl": "bL", "basicl": "bL",
}
TEXT_ALLOWED = {"stat", "dnam"}

TITLE_TEMPLATE = "Camera {camera} – Median Angular Error Distributions"
FIGSIZE = (16, 15)  # width, height in inches (16,10)
DPI = 200         # saved image resolution

# --------------------------------------------------------

class RecordingMeta(NamedTuple):
    participant: str   # e.g., "P006"
    text: str          # "stat" | "dnam"
    distance_cm: str   # e.g., "180cm"
    lighting_std: str  # "3L" | "bL"
    save_dir: Path     # participant folder e.g., .../P006_statisch

def _norm_distance(x: str) -> str:
    x = x.strip()
    m = re.search(r"(\d+)\s*cm?", x, flags=re.I)
    if m: return f"{m.group(1)}cm"
    m = re.search(r"(\d+)", x)
    return f"{m.group(1)}cm" if m else x

def _norm_lighting(x: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]", "", x).lower()
    return LIGHT_SYNONYMS.get(key, x.strip())

def _parse_subfolder_bits(name: str):
    """
    Accepts either:
      participant_distance(_cm)?_text_lighting
      participant_text_distance(_cm)?_lighting
    Returns (participant, distance_raw, text, lighting_raw) or None on fail.
    """
    parts = re.split(r"[_\- ]+", name.strip())
    # heuristic: participant always looks like P\d+
    p_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"[Pp]\d+", p)), None)
    if p_idx is None:
        return None
    participant = parts[p_idx].upper().replace("P", "P")

    # find text token
    text_idx = next((i for i, p in enumerate(parts) if p.lower() in TEXT_ALLOWED), None)
    text = parts[text_idx].lower() if text_idx is not None else "stat"  # default to stat if missing

    # only accept 80/120/180 optionally followed by 'cm'
    dist_idx = next(
        (i for i, p in enumerate(parts)
        if re.fullmatch(r"(?:80|120|180)(?:cm)?", p, flags=re.IGNORECASE)),
        None
    )
    distance_raw = parts[dist_idx] if dist_idx is not None else ""

    # lighting: token that looks like 3L / 3light(s) / bL / basicL
    light_idx = next(
        (i for i, p in enumerate(parts)
         if re.sub(r"[^a-zA-Z0-9]", "", p).lower() in LIGHT_SYNONYMS or p.lower() in {"3l", "bl"}),
        None
    )
    lighting_raw = parts[light_idx] if light_idx is not None else ""

    return participant, distance_raw, text, lighting_raw

def read_recording_dir(recording_raw_accuracy_dir: str) -> tuple[pd.DataFrame, RecordingMeta]:
    """
    recording_raw_accuracy_dir: path to .../<subfolder>/raw_accuracy
    Reads all *_angular_errors.txt files and builds a long-format DataFrame
    with columns: participant_id, label, distance, lighting, angular_error.
    Also returns normalized metadata & target save directory.
    """
    raw_dir = Path(recording_raw_accuracy_dir)
    if not raw_dir.is_dir():
        raise ValueError(f"Not a directory: {raw_dir}")

    subfolder = raw_dir.parent  # e.g., P006_180cm_stat_3lights
    participant_folder = subfolder.parent  # e.g., P006_statisch

    parsed = _parse_subfolder_bits(subfolder.name)
    if not parsed:
        raise ValueError(f"Could not parse subfolder name: {subfolder.name}")
    participant, distance_raw, text, lighting_raw = parsed

    meta = RecordingMeta(
        participant=participant,
        text=text,
        distance_cm=_norm_distance(distance_raw),
        lighting_std=_norm_lighting(lighting_raw),
        save_dir=participant_folder,
    )

    rows = []
    for txt in sorted(raw_dir.glob("*_angular_errors.txt")):
        stem = txt.stem  # label_participant_distance_lighting_angular_errors
        # Lenient filename parse (allow mixed separators)
        tokens = re.split(r"[_\- ]+", stem)
        # label is always first in your scheme; guard if someone prepends path junk
        label = tokens[0].upper() if tokens else "MM"

        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    val = float(line)
                except ValueError:
                    continue
                rows.append({
                    "participant_id": meta.participant,
                    "label": label,
                    "distance": re.sub(r"cm$", "", meta.distance_cm),  # store numeric string like "180"
                    "lighting": meta.lighting_std,
                    "angular_error": val,
                })

    if not rows:
        raise ValueError(f"No angular error values found in: {raw_dir}")

    df = pd.DataFrame(rows)
    # Let downstream util normalize final columns and add distance_light:
    df = _normalize_core_columns(df)
    df = add_distance_light_column(df)
    return df, meta

def _pretty_light_label(val: str) -> str:
    """
    Make lighting labels pretty:
      - 'bL'        -> '300 Lux'
      - '3L'        -> '700 Lux'
    Leaves anything else unchanged.
    """
    mapping = {
        "bL": "300 Lux",
        "3L": "700 Lux",
    }
    # Handle combined distance_light values like '80-bL'
    if isinstance(val, str) and "-" in val:
        dist, light = val.split("-", 1)
        return f"{dist} - {mapping.get(light, light)}"
    return mapping.get(val, val)

def add_distance_light_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # sanitize/normalize values
    df["distance"] = df["distance"].astype(str).str.strip()
    df["lighting"] = df["lighting"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["distance_light"] = df["distance"] + "-" + df["lighting"]
    #df["participant_id"] = df["participant_id"].astype(str).strip()
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

def plot_single_recording_boxplot(
    df: pd.DataFrame,
    meta: RecordingMeta,
    camera: str,
    save_path_png: str | None = None,
    orientation: str = "horizontal",
    show_n: bool = True,
    show_median: bool = False,
    overlay_jitter: bool = False,
    jitter_alpha: float = 0.35,
) -> io.BytesIO:
    """
    For one recording (one distance × one lighting) with either 1 label (MM) or 9 labels.
    Produces either:
      - a compact single-axis box for 1 label, or
      - a 3×3 label matrix for 9 labels.
    """
    labels_present = sorted(df["label"].unique().tolist())
    all_vals = df["angular_error"].dropna().values
    lo, hi = _compute_global_limits(all_vals)
    locator = mticker.MaxNLocator(nbins=5)

    title = f"{camera} – {meta.participant} – {meta.text} – {meta.distance_cm} – {meta.lighting_std}"

    # 9-label case → reuse your matrix routine (one lighting block)
    if set(labels_present) >= set(["LU","MU","RU","LM","MM","RM","LD","MD","RD"]):
        # Reuse _plot_matrix3x3_by_lighting by passing a composed "camera_id" as title
        buf = _plot_matrix3x3_by_lighting(
            data=df,
            camera_id=title,   # gets placed after "Camera" in that routine
            lo=lo, hi=hi, locator=locator,
            save_path_png=save_path_png,
            show_n=show_n, show_median=show_median,
            overlay_jitter=overlay_jitter, jitter_alpha=jitter_alpha,
            orientation=orientation
        )
        return buf

    # Single-label (typically MM) → simple one-axis box
    vals = df["angular_error"].dropna().values
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif len(vals) == 1:
        if orientation == "horizontal":
            ax.plot([vals[0]], [0], marker="o")
            ax.set_xlim(lo, hi); ax.set_yticks([])
        else:
            ax.plot([0], [vals[0]], marker="o")
            ax.set_ylim(lo, hi); ax.set_xticks([])
    else:
        ax.boxplot([vals], positions=[0], widths=0.6,
                   vert=(orientation == "vertical"),
                   showfliers=True, whis=1.5, patch_artist=True)
        if overlay_jitter:
            if orientation == "horizontal":
                jy = (np.random.rand(len(vals)) - 0.5) * 0.3
                ax.scatter(vals, jy, s=10, alpha=jitter_alpha)
            else:
                jx = (np.random.rand(len(vals)) - 0.5) * 0.3
                ax.scatter(jx, vals, s=10, alpha=jitter_alpha)

        if orientation == "horizontal":
            #ax.set_ylim(-1, 1); ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(locator); ax.set_yticks([])
            #ax.grid(alpha=0.2, axis="x")
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(locator)
            ax.set_yticks([])
            # don't fix ylim to [-1, 1]
        else:
            ax.set_xlim(-1, 1); ax.set_ylim(lo, hi); ax.yaxis.set_major_locator(locator); ax.set_xticks([])
            ax.grid(alpha=0.2, axis="y")

    ax.set_title(f"{title} – {labels_present[0] if len(labels_present)==1 else 'Labels'}")
    if show_n:
        ax.text(0.98, 0.95, f"n={len(vals)}", transform=ax.transAxes, ha="right", va="top", fontsize=9, alpha=0.85)

    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight"); buf.seek(0); plt.close(fig)
    if save_path_png:
        with open(save_path_png, "wb") as f:
            f.write(buf.getbuffer())
    return buf

def save_recording_boxplot(recording_raw_accuracy_dir: str, camera_id: str) -> Path:
    df, meta = read_recording_dir(recording_raw_accuracy_dir)

    # Build output filename in the participant folder
    out_name = f"{meta.participant}_{meta.text}_{meta.distance_cm}_{meta.lighting_std}_raw_acc_boxPlot.png"
    out_path = meta.save_dir / out_name

    # Pick plotting flavor based on labels present
    labels_present = df["label"].unique().tolist()
    if len(labels_present) == 1:
        plot_single_recording_boxplot(df, meta, camera_id, save_path_png=str(out_path))
    else:
        # Use the matrix flavor (one lighting block)
        lo, hi = _compute_global_limits(df["angular_error"].values)
        locator = mticker.MaxNLocator(nbins=5)
        _plot_matrix3x3_by_lighting(
            data=df,
            camera_id=f"{camera_id} – {meta.participant} – {meta.text} – {meta.distance_cm} – {meta.lighting_std}",
            lo=lo, hi=hi, locator=locator,
            save_path_png=str(out_path),
            show_n=True, show_median=False,
            overlay_jitter=False, jitter_alpha=0.35,
            orientation="horizontal"
        )
    return out_path

def plot_label_boxgrid(
    df: pd.DataFrame,
    camera_id: str | None,
    columns: str = "distance_light",   # "distance_light" | "lighting" | "distance"
    layout: str = "grid",              # "grid" | "matrix3x3"
    save_path_png: str | None = None,
    show_n: bool = True,
    show_median: bool = False,
    overlay_jitter: bool = False,
    jitter_alpha: float = 0.35,
    orientation: str = "horizontal"    # "horizontal" | "vertical"
) -> io.BytesIO:
    
    if camera_id is None:
        data = df.copy()
    else:
        data = df[df["camera_id"].astype(str) == str(camera_id)].copy()
        if data.empty:
            raise ValueError(f"No rows found for camera_id={camera_id}")

    data = _normalize_core_columns(data)
    data = add_distance_light_column(data)

    all_vals = data["angular_error"].dropna().values
    if len(all_vals) == 0:
        raise ValueError("No angular_error values present.")
    lo = 0
    hi = 20
    if lo is None or hi is None:
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
    camera_id: str | None,
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
    
    if camera_id is None: 
        fig.suptitle(
        f"Median  Angular Error Distributions ({pretty_cols})",
        fontsize=14, y=0.995)
    else:
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
                if c == 0: ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=12, fontsize=13)
                if r == 0: ax.set_title(_pretty_light_label(colcat), fontsize=13, pad=6)
                continue

            if orientation == "horizontal":
                ax.set_ylim(-1, 1); ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(locator); ax.set_yticks([])
            else:
                ax.set_xlim(-1, 1); ax.set_ylim(lo, hi); ax.yaxis.set_major_locator(locator); ax.set_xticks([])

            ax.grid(alpha=0.2, axis="x" if orientation == "horizontal" else "y")
            if c == 0: ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=12, fontsize=13)
            if r == 0: ax.set_title(_pretty_light_label(colcat), fontsize=13, pad=6)
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
    camera_id: str | None,
    lo: float, hi: float, locator: mticker.MaxNLocator,
    save_path_png: str | None,
    show_n: bool, show_median: bool,
    overlay_jitter: bool, jitter_alpha: float,
    orientation: str
) -> io.BytesIO:

    lights_present = [lt for lt in LIGHT_ORDER if lt in data["lighting"].unique()]
    if not lights_present:
        raise ValueError("No lighting categories found in data for matrix3x3 layout.")

    stats_rows: list[dict] = []

    n_blocks = len(lights_present)
    #fig = plt.figure(figsize=(8 * n_blocks, 10))
    # For 3×3 by lighting
    cell = 3.0  # inches per subplot side
    fig_w = cell * 3 * n_blocks
    fig_h = cell * 3
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(nrows=1, ncols=n_blocks, figure=fig, wspace=0.15)

    if camera_id is None:
         fig.suptitle(
            f"Median Angular Error Distributions (Labels 3×3 × Lighting x Distance)",
            fontsize=14, y=0.995
        )
    elif " – " in camera_id:
        fig.suptitle(
            f"Raw Accuracy Per Participant - {camera_id}",
            fontsize=14, y=0.995
        )
    else:
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

                if n >= 1:
                    bstat = boxplot_stats(vals, whis=1.5)[0]  # aligns with ax.boxplot(..., whis=1.5)
                    stats_rows.append({
                        "lighting": lt,
                        "label": lab,
                        "n": int(n),
                        "q1": round(float(bstat["q1"]), 4),
                        "median": round(float(bstat["med"]), 4),
                        "q3": round(float(bstat["q3"]), 4),
                        "iqr": round(float(bstat["iqr"]), 4),
                        "whisker_low": round(float(bstat["whislo"]), 4),
                        "whisker_high": round(float(bstat["whishi"]), 4),
                        "mean": round(float(np.mean(vals)), 4),
                        "min": round(float(np.min(vals)), 4),
                        "max": round(float(np.max(vals)), 4),
                        # store fliers as JSON array for readability
                        "fliers": json.dumps([float(x) for x in bstat["fliers"]]),
                    })

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
        fig.text(cx, ty, f"{LIGHT_MAP.get(lt, lt)}  –  {dist_text}", 
         ha="center", va="bottom", fontsize=12)

    # --- save outputs
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)

    if save_path_png:
        fig.savefig(save_path_png, dpi=200, bbox_inches="tight")
        # write stats next to the PNG
        root, _ = os.path.splitext(save_path_png)
        stats_path = root + "_stats.tsv"
        pd.DataFrame(stats_rows).to_csv(stats_path, sep="\t", index=False)

    plt.close(fig)
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

def find_raw_accuracy_dirs(root_dir: str) -> list[Path]:
    """Return all directories named 'raw_accuracy' under root (depth-agnostic)."""
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Root is not a directory: {root}")
    # rglob is fine; restrict to directories named exactly 'raw_accuracy'
    return [p for p in root.rglob("raw_accuracy") if p.is_dir()]

def process_root_dir(
    root_dir: str,
    camera_id: str,
    only_participants: set[str] | None = None
) -> list[tuple[Path, Path | None]]:
    """
    Walk the tree, find all raw_accuracy folders, and render plots.
    Returns list of (raw_accuracy_dir, saved_png_path_or_None).
    """
    hits = find_raw_accuracy_dirs(root_dir)
    hits.sort()
    results = []

    for raw_dir in hits:
        # Optional participant filtering using parent naming (e.g., .../P006_statisch/...)
        subfolder = raw_dir.parent
        participant_folder = subfolder.parent
        # Heuristic: look for 'P\d+' in the path parts
        p_ids = [s for s in participant_folder.parts if re.fullmatch(r"[Pp]\d+", s)]
        pid = p_ids[0].upper() if p_ids else None

        if only_participants and pid and pid not in only_participants:
            print(f"Skip {raw_dir} (participant {pid} not in filter).")
            results.append((raw_dir, None))
            continue

        print(f"Processing: {raw_dir}")
        try:
            out_path = save_recording_boxplot(str(raw_dir), camera_id)
            print(f"  → Saved: {out_path}")
            results.append((raw_dir, out_path))
        except Exception as e:
            print(f"  ! Failed: {raw_dir} | {e}")
            results.append((raw_dir, None))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Path to long-format CSV.")
    parser.add_argument("--camera", required=True, help="Camera ID to filter (e.g., 521).")
    parser.add_argument("--participants", nargs="+", help="List of participant IDs to include (e.g., --participants P001 P007 P042).")
    parser.add_argument("--out", default="boxplots.xlsx", help="Output Excel file path.")
    parser.add_argument("--save-png", default=None, help="(Optional) Also save the figure as PNG.")
    parser.add_argument("--distances", nargs="+", help='Distances to include (e.g., 80 120). If omitted, all distances are used.')
    parser.add_argument("--layout", choices=["grid", "matrix3x3"], default="grid",
        help="Plot layout: 'grid' = small multiples table (current). 'matrix3x3' = spatial label layout with one 3x3 block per lighting.")
    
    parser.add_argument("--recording-dir", help="Path to a single recording's raw_accuracy folder to create one box plot.")
    parser.add_argument("--root-dir", help="Root folder (e.g., E:\\2Dto3D_Conversion\\581_stat) to auto-scan for raw_accuracy subfolders.")
    args = parser.parse_args()

    # --- NEW: batch mode from a root
    if args.root_dir:
        only = {p.strip().upper() for p in args.participants} if args.participants else None
        process_root_dir(args.root_dir, args.camera, only_participants=only)
        return

    # --- NEW: one-recording mode
    if args.recording_dir:
        out_path = save_recording_boxplot(args.recording_dir, args.camera)
        print(f"Saved single-recording plot: {out_path}")
        return

    # --- Existing CSV-driven mode (minor tweak: camera is only required here)
    if not args.camera:
        raise ValueError("When not using --recording-dir, you must pass --camera together with --input_csv.")
    if not args.input_csv:
        raise ValueError("When not using --recording-dir, you must pass --input_csv.")
    if args.camera == "all": 
        camera_is_all = True
    else: 
        camera_is_all = False
        camera_arg = args.camera
    df = pd.read_csv(args.input_csv)
    required_cols = {"camera_id", "participant_id", "label", "distance", "lighting", "angular_error", "rms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["distance"] = df["distance"].astype(str).str.strip()  # normalize just once

    if args.distances:
        allowed = {str(d).strip() for d in args.distances}
        df = df[df["distance"].isin(allowed)]
        if df.empty:
            raise ValueError(f"No rows left after filtering for distances={sorted(allowed)}")
        
    if args.participants:
        allowed_participants = {str(p).strip() for p in args.participants}
        df = df[df["participant_id"].astype(str).isin(allowed_participants)]
        if df.empty:
            raise ValueError(f"No rows left after filtering for participants={sorted(allowed_participants)}")

    show_n = True
    show_median = False
    overlay_jitter = False
    jitter_alpha = 0.35
    orientation = "horizontal"
    columns = "lighting" # "distance_light" | "lighting" | "distance"
    camera_id=None if camera_is_all else camera_arg
    save_path_png = args.save_png
    layout = args.layout
        

    png_buf = plot_label_boxgrid(df, camera_id, columns, layout, save_path_png, show_n, show_median, overlay_jitter, jitter_alpha, orientation)
    write_excel_with_figure(df, png_buf, args.out)

    print(f"Done. Wrote: {args.out}")
    if args.save_png:
        print(f"Also saved PNG: {args.save_png}")


if __name__ == "__main__":
    main()

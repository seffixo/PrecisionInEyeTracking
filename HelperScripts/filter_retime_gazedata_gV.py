#!/usr/bin/env python3
"""
Filter gazedata to event windows, retime timestamps, rename original to gazedata_old.gz,
and write the filtered+retimed stream back as gazedata.gz in the original format.

Key behavior
------------
- Reads the source gazedata (prefers 'gazedata.gz' in each target folder; falls back to files
  named like gazedata_P0XX_* if needed).
- Finds 'Event_time_ranges.txt' next to the gazedata.
- Filters lines whose 'timestamp' falls inside the union of event windows.
- Merges overlapping/adjacent windows into contiguous 'segments'.
- Retime rules:
  * Segment 0: first kept line -> 0.000000
  * Inside a segment: new_ts(i) = new_ts(i-1) + (orig_ts(i) - orig_ts(i-1))
  * Between segments: first line of segment k>0 -> last_new_ts_of_prev_segment + GAP (default 0.010000)
- Only the numeric value of "timestamp" is changed; the rest of the JSON line stays as-is
  (field order, whitespace, other fields preserved) via regex replacement.
- When complete, renames original gazedata.gz -> gazedata_old.gz (safe, with suffix if exists),
  and places the new filtered+retimed stream as gazedata.gz.

Usage
-----
python filter_and_retime_gazedata.py --root /path/to/study
python filter_and_retime_gazedata.py --root . --exclusive          # exclusive bounds
python filter_and_retime_gazedata.py --root . --gap-ms 10          # default 10 ms
python filter_and_retime_gazedata.py --root . --dry-run            # do everything except rename/replace

Directory expectation
---------------------
Your project structure is assumed to contain subfolders that match:
  ^(P0\\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$
Inside each, there is:
  - gazedata.gz   (preferred), or a file matching gazedata_P0XX_*
  - Event_time_ranges.txt

If your structure differs, adapt discover logic near the bottom.
"""

import argparse
import gzip
import json
import os
import re
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import gzip

import logging

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("retime_gaze")

# ---------------- Patterns ----------------
PREFIX_RE = re.compile(
    r"^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$",
    re.IGNORECASE,
)
GAZEFILE_RE = re.compile(r"^gazedata_P0\d{2}_.+", re.IGNORECASE)

# Replace only the timestamp number, keep original spacing and JSON text
TS_FIELD_RE = re.compile(r'("timestamp"\s*:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

def normalize_prefix(folder_name: str) -> Optional[str]:
    m = PREFIX_RE.match(folder_name)
    if not m:
        return None
    pnum, dist, _cm, light = m.groups()
    l = light.lower()
    light_norm = "bL" if l in ("basicl", "bl") else ("3L" if l in ("3lights", "3light", "3l") else light)
    return f"{pnum}_{dist}_{light_norm}"

# ---------------- Events ----------------
EventRange = Tuple[Decimal, Decimal]  # [start, end]

def parse_event_ranges(path: str) -> List[EventRange]:
    ranges: List[EventRange] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                parts = [p for p in re.split(r"[,\s]+", line) if p]
            if len(parts) != 2:
                raise ValueError(f"Bad event line (need LABEL,START,END): {raw!r}")
            s, e = parts[0], parts[1]
            start = Decimal(s)
            end = Decimal(e)
            if end < start:
                raise ValueError(f"End before start: {start} > {end}")
            ranges.append((start, end))
    # sort by start
    ranges.sort(key=lambda x: x[0])
    return ranges

def merge_ranges(ranges: List[EventRange], inclusive: bool) -> List[EventRange]:
    """
    Merge overlapping or immediately-adjacent ranges into contiguous segments.
    'Adjacent' means touching at a boundary under the chosen inclusivity:
      - inclusive: [a,b] and [b,c] merge into [a,c]
      - exclusive: (a,b) and (b,c) do NOT merge (gap at b)
    """
    if not ranges:
        return []
    merged: List[EventRange] = []
    cur_s, cur_e = ranges[0]
    for s, e in ranges[1:]:
        if inclusive:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        else:
            # exclusive: require actual overlap to merge
            if s < cur_e and e > cur_s:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

# ---------------- IO helpers ----------------
GZIP_MAGIC = b"\x1f\x8b"

def is_really_gzip(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == GZIP_MAGIC
    except FileNotFoundError:
        return False

def open_text_reader_sniff(path: str):
    """
    Return a text reader that auto-detects gzip by magic bytes,
    ignoring the file extension.
    """
    if is_really_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        return open(path, "rt", encoding="utf-8", newline="")

def open_text_writer(path: str, force_gzip: bool):
    """
    Text writer that ignores file extension and writes gzip if requested.
    """
    if force_gzip:
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return open(path, "wt", encoding="utf-8", newline="")

def iter_lines_sniff(path: str):
    with open_text_reader_sniff(path) as f:
        for raw in f:
            yield raw

def extract_timestamp(line: str) -> Optional[Decimal]:
    # fast path: scan via regex
    m = TS_FIELD_RE.search(line)
    if not m:
        # fallback: try JSON load
        try:
            obj = json.loads(line)
        except Exception:
            return None
        ts = obj.get("timestamp", None)
        if ts is None:
            return None
        return Decimal(str(ts))
    try:
        return Decimal(m.group(2))
    except Exception:
        return None

def replace_timestamp(line: str, new_ts: Decimal) -> str:
    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        return f'{prefix}{format_decimal(new_ts)}'
    if TS_FIELD_RE.search(line):
        return TS_FIELD_RE.sub(repl, line, count=1)
    # fallback via JSON if regex didn't match (rare)
    try:
        obj = json.loads(line)
        obj["timestamp"] = float(new_ts)
        # We avoid json.dumps to preserve most of the original text, but if we must:
        return json.dumps(obj, separators=(",", ":")) + ("\n" if not line.endswith("\n") else "")
    except Exception:
        # give up and just return original line
        return line

# ---------------- Retime logic ----------------
def in_window(ts: Decimal, rng: EventRange, inclusive: bool) -> bool:
    s, e = rng
    return (s <= ts <= e) if inclusive else (s < ts < e)

def format_decimal(x: Decimal) -> str:
    # exactly 6 decimals, half-up rounding like typical float formatting
    q = Decimal("0.000001")
    return str(x.quantize(q, rounding=ROUND_HALF_UP))

def safe_rename(src: Path, dst: Path) -> Path:
    """
    Rename src -> dst; if dst exists, add _1, _2, ...
    Returns final dst path.
    """
    if not dst.exists():
        src.rename(dst)
        return dst
    stem = dst.stem
    suffix = "".join(dst.suffixes)  # e.g. ".gz"
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            src.rename(cand)
            return cand
        i += 1

def filter_and_retime_one(
    gaze_path: Path,
    events_path: Path,
    inclusive: bool,
    gap_ms: int,
    dry_run: bool,
) -> None:
    ranges = parse_event_ranges(str(events_path))
    if not ranges:
        log.info(f"[skip] No event ranges in {events_path}")
        return
    segments = merge_ranges(ranges, inclusive=inclusive)

    # We write to a temp file first
    out_tmp = gaze_path.with_suffix(gaze_path.suffix + ".tmp")  # e.g., gazedata.gz.tmp
    is_input_gz = is_really_gzip(str(gaze_path))                # <-- sniff the actual bytes
    gap = Decimal(gap_ms) / Decimal(1000)

    seg_index = 0
    prev_new_ts: Optional[Decimal] = None
    prev_orig_ts_in_seg: Optional[Decimal] = None

    wrote = 0
    total = 0

    # IMPORTANT: write gzip iff the input truly was gzip, not by extension
    with open_text_writer(str(out_tmp), force_gzip=is_input_gz) as out_f:
        for raw in iter_lines_sniff(str(gaze_path)):
            total += 1
            ts = extract_timestamp(raw)
            if ts is None:
                continue

            # Advance segments while this ts is beyond the current segment
            while seg_index < len(segments) and ts > segments[seg_index][1]:
                # leaving segment seg_index
                seg_index += 1
                prev_orig_ts_in_seg = None  # reset for a new segment

            # If we've exhausted segments, stop early
            if seg_index >= len(segments):
                break

            # Skip until we enter the current segment
            if not in_window(ts, segments[seg_index], inclusive=inclusive):
                continue

            # Inside current segment: compute new timestamp
            if prev_orig_ts_in_seg is None:
                # First line of this segment
                if prev_new_ts is None:
                    new_ts = Decimal("0.000000")
                else:
                    new_ts = prev_new_ts + gap
            else:
                delta = ts - prev_orig_ts_in_seg
                # guard against non-monotonicity (shouldn't happen normally)
                if delta < 0:
                    delta = Decimal("0.000000")
                new_ts = prev_new_ts + delta  # continue from last written time

            # Write line with patched timestamp
            patched = replace_timestamp(raw, new_ts)
            out_f.write(patched)

            # Update trackers
            prev_orig_ts_in_seg = ts
            prev_new_ts = new_ts
            wrote += 1

    log.info(f"Kept {wrote} / {total} lines from {gaze_path.name}")

    if dry_run:
        log.info("[dry-run] Skipping rename/replace.")
        try:
            out_tmp.unlink()  # remove temp
        except Exception:
            pass
        return

    # Now swap files:
    # 1) Move original gazedata.gz -> gazedata_old.gz (safe)
    # 2) Move out_tmp -> gazedata.gz
    if gaze_path.name.lower() == "gazedata.gz":
        backup = gaze_path.with_name("gazedata_old.gz")
    else:
        # If the file has another name, still back it up with _old
        backup = gaze_path.with_name(gaze_path.stem + "_old" + "".join(gaze_path.suffixes))

    log.info(f"Renaming original -> {backup.name}")
    final_backup = safe_rename(gaze_path, backup)

    log.info(f"Placing filtered+retimed -> {gaze_path.name}")
    out_tmp.replace(gaze_path)

    log.info(f"Done. Original backed up as: {final_backup}")

# ---------------- Discovery ----------------
def discover_gazedata_file(folder: Path) -> Optional[Path]:
    # Prefer canonical 'gazedata.gz'
    cand = folder / "gazedata.gz"
    if cand.exists():
        return cand
    # Fallback: any file like gazedata_P0XX_*
    for fn in folder.iterdir():
        if fn.is_file() and GAZEFILE_RE.match(fn.name):
            return fn
    return None

def find_event_file(folder: Path) -> Optional[Path]:
    p = folder / "Event_time_ranges.txt"
    return p if p.exists() else None

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Filter, retime, and replace gazedata.gz (backup original).")
    ap.add_argument("--root", required=True, help="Root directory to scan.")
    ap.add_argument("--exclusive", action="store_true", help="Use exclusive bounds (start < ts < end). Default inclusive.")
    ap.add_argument("--gap-ms", type=int, default=10, help="Gap in milliseconds between segments (default 10).")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except rename/replace on disk.")
    args = ap.parse_args()

    inclusive = not args.exclusive

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Iterate two levels deep, similar to your previous layout
    for top in root.iterdir():
        if not top.is_dir():
            continue
        for sub in top.iterdir():
            if not sub.is_dir():
                continue

            prefix = normalize_prefix(sub.name)
            if not prefix:
                continue  # skip non-target folders

            gaze = discover_gazedata_file(sub)
            if not gaze:
                log.warning(f"[skip] No gazedata found in {sub}")
                continue
            events = find_event_file(sub)
            if not events:
                log.warning(f"[skip] No Event_time_ranges.txt in {sub}")
                continue

            log.info(f"Processing: {sub} ({prefix})")
            filter_and_retime_one(
                gaze_path=gaze,
                events_path=events,
                inclusive=inclusive,
                gap_ms=args.gap_ms,
                dry_run=args.dry_run,
            )

if __name__ == "__main__":
    main()

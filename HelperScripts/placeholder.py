#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path

# Your prefix regex (folder name) – case-insensitive
PREFIX_RE = re.compile(
    r"^(P0\d{2})(?:_[^_]+)?_(80|120|180)(cm)?(?:_[^_]+)?_(basicL|bL|3lights|3light|3L)$",
    re.IGNORECASE,
)

FILES_NEEDED = ("poster_reference.png", "poster_targets.json")

def normalize_lights(label: str) -> str:
    """Map variations to canonical values: 'bL' or '3L'."""
    l = label.lower()
    if l in ("basicl", "bl"):
        return "bL"
    if l in ("3l", "3light", "3lights"):
        return "3L"
    return label  # fallback (shouldn't happen with this regex)

def match_key_from_foldername(name: str):
    """
    Return (size, lights) tuple if folder name matches PREFIX_RE,
    with lights normalized and size as plain digits (no 'cm').
    """
    m = PREFIX_RE.match(name)
    if not m:
        return None
    _, size, cm, lights = m.groups()
    size_clean = size  # '80'|'120'|'180'
    lights_norm = normalize_lights(lights)
    return (size_clean, lights_norm)

def find_matching_folders(base_dir: Path):
    """
    Recursively yield (folder_path, key) for every directory under base_dir
    whose name matches PREFIX_RE.
    """
    for folder in base_dir.rglob("*"):
        if folder.is_dir():
            key = match_key_from_foldername(folder.name)
            if key:
                yield folder, key

def source_lookup(source_dir: Path):
    """
    Build a dictionary: (size, lights) -> source_folder_path
    Only include folders that actually contain both required files.
    If multiple exist, first one found wins.
    """
    lookup = {}
    for folder, key in find_matching_folders(source_dir):
        have_all = all((folder / f).is_file() for f in FILES_NEEDED)
        if have_all and key not in lookup:
            lookup[key] = folder
    return lookup

def copy_missing_files(src_folder: Path, dst_folder: Path):
    """
    Copy needed files from src to dst if they are missing in dst.
    Returns list of copied file names.
    """
    copied = []
    for fname in FILES_NEEDED:
        src = src_folder / fname
        dst = dst_folder / fname
        if not dst.exists():
            if src.exists():
                shutil.copy2(src, dst)
                copied.append(fname)
            else:
                # Shouldn't happen because we validated sources, but guard anyway
                print(f"[WARN] Source file missing: {src}")
    return copied

def main():
    ap = argparse.ArgumentParser(
        description="Fill poster files in target folders by matching (size, lights) from source folders."
    )
    ap.add_argument("--source_dir", "-s", type=Path, help="First directory (provides reference files)")
    ap.add_argument("--target_dir", "-t", type=Path, help="Second directory (folders to fill)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be copied without changing files")
    args = ap.parse_args()

    src_dir = args.source_dir.resolve()
    dst_dir = args.target_dir.resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")
    if not dst_dir.is_dir():
        raise SystemExit(f"Target directory not found: {dst_dir}")

    lookup = source_lookup(src_dir)
    if not lookup:
        print(f"[INFO] No valid source folders with both files found in {src_dir}.")
    else:
        print(f"[INFO] Found {len(lookup)} source groups (by (size, lights)).")

    total_copied = 0
    total_targets = 0

    for tgt_folder, key in find_matching_folders(dst_dir):
        total_targets += 1
        # Skip if target already has both files
        if all((tgt_folder / f).is_file() for f in FILES_NEEDED):
            print(f"[SKIP] {tgt_folder.name} already has both files.")
            continue

        src_folder = lookup.get(key)
        if not src_folder:
            print(f"[MISS] No source for {tgt_folder.name} with key {key}.")
            continue

        print(f"[MATCH] {tgt_folder.name} <= {src_folder.name} (key={key})")
        if args.dry_run:
            # Show which files would be copied
            would_copy = [f for f in FILES_NEEDED if not (tgt_folder / f).exists()]
            if would_copy:
                print(f"  [DRY-RUN] Would copy: {', '.join(would_copy)}")
            else:
                print("  [DRY-RUN] Nothing to copy (already present).")
        else:
            copied = copy_missing_files(src_folder, tgt_folder)
            if copied:
                print(f"  [COPIED] {', '.join(copied)}")
                total_copied += len(copied)
            else:
                print("  [INFO] Nothing copied (already present).")

    print(f"\n[SUMMARY] Targets scanned: {total_targets}")
    print(f"[SUMMARY] Files copied: {total_copied}")

if __name__ == "__main__":
    main()

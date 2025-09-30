#!/usr/bin/env python3
"""
Rename all *"_cropped.mp4" files to remove the "_cropped" suffix.

Examples:
  myvideo_cropped.mp4 -> myvideo.mp4

- Works recursively from a base folder (default: current dir).
- If a target name already exists, it will SKIP that file to avoid overwriting.
- Use --dry-run to preview actions without renaming.
"""

import argparse
import os

def rename_files(base: str, dry_run: bool):
    for dirpath, _, files in os.walk(base):
        for f in files:
            if not f.lower().endswith("_cropped.mp4"):
                continue

            src = os.path.join(dirpath, f)
            dst = os.path.join(dirpath, f[:-12] + ".mp4")  # strip "_cropped"

            if os.path.exists(dst):
                print(f"[SKIP] {src} -> {dst} (target exists)")
                continue

            if dry_run:
                print(f"[DRY]  {src} -> {dst}")
            else:
                os.rename(src, dst)
                print(f"RENAMED {src} -> {dst}")

def main():
    ap = argparse.ArgumentParser(description="Rename *_cropped.mp4 files by removing '_cropped'.")
    ap.add_argument("--base", default=".", help="Base directory (default: current dir).")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without renaming.")
    args = ap.parse_args()

    root = os.path.abspath(args.base)
    print(f"Scanning: {root} {'[DRY RUN]' if args.dry_run else ''}")
    rename_files(root, args.dry_run)
    print("Done.")

if __name__ == "__main__":
    main()

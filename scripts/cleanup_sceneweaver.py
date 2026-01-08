#!/usr/bin/env python3
"""
Cleanup SceneWeaver intermediate files to save disk space.

Keeps only files needed for conversion:
- args.json (contains final iteration number)
- roominfo.json (room dimensions)
- record_files/scene_{final_iter}.blend (final Blender file)
- record_scene/layout_{final_iter}.json (final layout)

Two modes of operation:
1. In-place mode (default): Deletes everything else (~8GB per scene) from input directory.
2. Output mode (--output_dir): Copies only essential files to output directory,
   keeps original directory untouched.
"""

import json
import shutil
from pathlib import Path


def cleanup_scene(
    scene_dir: Path, dry_run: bool = False, output_dir: Path = None
) -> dict:
    """
    Clean up a single SceneWeaver scene directory.

    If output_dir is provided, copies only essential files to output_dir
    and keeps original untouched. Otherwise, deletes files in-place.

    Returns dict with stats about what was deleted/copied.
    """
    stats = {
        "scene": scene_dir.name,
        "files_deleted": 0,
        "dirs_deleted": 0,
        "bytes_freed": 0,
        "files_copied": 0,
        "errors": [],
    }

    # Read final iteration from args.json
    args_file = scene_dir / "args.json"
    if not args_file.exists():
        stats["errors"].append("args.json not found")
        return stats

    with open(args_file) as f:
        args = json.load(f)

    final_iter = args.get("iter", 0)

    # Files to KEEP (relative to scene_dir)
    keep_files_relative = [
        "args.json",
        "roominfo.json",
        "objav_cnts.json",
        "objav_files.json",
        f"record_files/scene_{final_iter}.blend",
        f"record_scene/layout_{final_iter}.json",
    ]

    # If output_dir is specified, copy only essential files
    if output_dir is not None:
        output_scene_dir = output_dir / scene_dir.name
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in keep_files_relative:
            src_file = scene_dir / rel_path
            dst_file = output_scene_dir / rel_path

            if src_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                if dry_run:
                    print(f"  [DRY RUN] Would copy: {rel_path}")
                else:
                    shutil.copy2(src_file, dst_file)
                stats["files_copied"] += 1

        if not dry_run:
            print(
                f"  Copied {stats['files_copied']} essential files to {output_scene_dir}"
            )

        return stats

    # Otherwise, delete files in-place (original behavior)
    keep_files = {scene_dir / rel_path for rel_path in keep_files_relative}

    # Directories to completely DELETE
    dirs_to_delete = [
        scene_dir / "pipeline",
        scene_dir / "args",
    ]

    # Delete unnecessary directories
    for dir_path in dirs_to_delete:
        if dir_path.exists():
            size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            if dry_run:
                print(
                    f"  [DRY RUN] Would delete dir: {dir_path} ({size / 1024 / 1024:.1f} MB)"
                )
            else:
                shutil.rmtree(dir_path)
                print(f"  Deleted dir: {dir_path.name}")
            stats["dirs_deleted"] += 1
            stats["bytes_freed"] += size

    # Clean up record_files/ - keep only final blend
    record_files_dir = scene_dir / "record_files"
    if record_files_dir.exists():
        for file_path in record_files_dir.iterdir():
            if file_path not in keep_files:
                size = file_path.stat().st_size if file_path.is_file() else 0
                if dry_run:
                    print(
                        f"  [DRY RUN] Would delete: {file_path.name} ({size / 1024 / 1024:.1f} MB)"
                    )
                else:
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
                stats["files_deleted"] += 1
                stats["bytes_freed"] += size

    # Clean up record_scene/ - keep only final layout
    record_scene_dir = scene_dir / "record_scene"
    if record_scene_dir.exists():
        for file_path in record_scene_dir.iterdir():
            if file_path not in keep_files:
                size = file_path.stat().st_size if file_path.is_file() else 0
                if dry_run:
                    pass  # Don't print all the small files
                else:
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
                stats["files_deleted"] += 1
                stats["bytes_freed"] += size

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up SceneWeaver intermediate files"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/ubuntu/SceneEval/input/SceneWeaver"),
        help="Path to SceneWeaver input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory. If specified, copies only essential files here and keeps original untouched",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted/copied without actually doing it",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output mode: Will copy essential files to {output_dir}")
        print(f"Original files in {input_dir} will be kept untouched")

    scene_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")]
    )

    print(f"Found {len(scene_dirs)} SceneWeaver scenes")
    if args.dry_run:
        print("DRY RUN MODE - no files will be deleted/copied\n")

    total_freed = 0
    total_files = 0
    total_dirs = 0
    total_copied = 0

    for scene_dir in scene_dirs:
        print(f"\nProcessing {scene_dir.name}...")
        stats = cleanup_scene(scene_dir, dry_run=args.dry_run, output_dir=output_dir)

        total_freed += stats["bytes_freed"]
        total_files += stats["files_deleted"]
        total_dirs += stats["dirs_deleted"]
        total_copied += stats["files_copied"]

        if stats["errors"]:
            print(f"  Errors: {stats['errors']}")
        elif output_dir is None:
            print(f"  Freed: {stats['bytes_freed'] / 1024 / 1024 / 1024:.2f} GB")

    print(f"\n{'=' * 50}")
    if output_dir is not None:
        print(f"TOTAL: {total_copied} files copied to {output_dir}")
        print(f"Original files in {input_dir} preserved")
    else:
        print(f"TOTAL: {total_files} files, {total_dirs} dirs deleted")
        print(f"TOTAL FREED: {total_freed / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()

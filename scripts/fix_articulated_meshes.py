#!/usr/bin/env python3
"""
Fix articulated object meshes for SceneEval.

NOTE: For new conversions, use conversion/scene_agent/convert_SceneEval.py which
automatically merges articulated meshes during conversion. This script is for
fixing already-converted scenes.

The combined_scene.gltf files exported by scene-agent contain multiple nodes
(one per articulated link) that Blender imports as separate objects. This script
merges them into a single mesh file that can be imported correctly.

Usage:
    python scripts/fix_articulated_meshes.py input/SceneAgent

This will:
1. Find all combined_scene.gltf files in the input directory
2. Merge all meshes into a single mesh
3. Save as combined_merged.glb (single mesh with materials preserved)

The scene_agent.py asset loader will then use combined_merged.glb if present.
"""

import json
import argparse
from pathlib import Path

import trimesh


def is_articulated_sdf(sdf_dir: Path) -> bool:
    """Check if an SDF directory contains an articulated object."""
    combined_scene = sdf_dir / "combined_scene.gltf"
    if not combined_scene.exists():
        return False

    # Check if there are multiple link meshes (E_*_combined.gltf files)
    link_meshes = list(sdf_dir.glob("E_*_combined.gltf"))
    return len(link_meshes) > 1


def merge_combined_scene(sdf_dir: Path) -> Path | None:
    """
    Merge combined_scene.gltf into a single mesh file.

    Args:
        sdf_dir: Directory containing combined_scene.gltf

    Returns:
        Path to merged file, or None if merging failed
    """
    combined_scene_path = sdf_dir / "combined_scene.gltf"
    merged_output_path = sdf_dir / "combined_merged.glb"

    # Skip if already merged
    if merged_output_path.exists():
        return merged_output_path

    try:
        # Load the GLTF scene
        scene = trimesh.load(str(combined_scene_path))

        if isinstance(scene, trimesh.Scene):
            # Get all geometry from the scene
            meshes = []
            for node_name in scene.graph.nodes_geometry:
                transform, geometry_name = scene.graph[node_name]
                geometry = scene.geometry[geometry_name]

                if isinstance(geometry, trimesh.Trimesh):
                    # Apply the node transform to the mesh
                    mesh_copy = geometry.copy()
                    mesh_copy.apply_transform(transform)
                    meshes.append(mesh_copy)

            if not meshes:
                print(f"    Warning: No meshes found in {combined_scene_path}")
                return None

            # Concatenate all meshes into one
            merged = trimesh.util.concatenate(meshes)

            # Export as GLB (binary format preserves more data)
            merged.export(str(merged_output_path))

            print(f"    Merged {len(meshes)} meshes -> {merged_output_path.name}")
            return merged_output_path

        elif isinstance(scene, trimesh.Trimesh):
            # Already a single mesh, just copy it
            scene.export(str(merged_output_path))
            print(f"    Single mesh copied -> {merged_output_path.name}")
            return merged_output_path

        else:
            print(f"    Warning: Unexpected scene type: {type(scene)}")
            return None

    except Exception as e:
        print(f"    Error merging {combined_scene_path}: {e}")
        return None


def fix_scene_articulated_meshes(scene_dir: Path) -> int:
    """
    Fix all articulated meshes in a scene directory.

    Args:
        scene_dir: Path to scene directory (e.g., input/SceneAgent/scene_0)

    Returns:
        Number of meshes fixed
    """
    assets_dir = scene_dir / "assets"
    if not assets_dir.exists():
        return 0

    fixed_count = 0

    # Find all SDF directories
    for category in ["furniture", "manipuland", "wall_mounted", "ceiling_mounted"]:
        sdf_base = assets_dir / category / "sdf"
        if not sdf_base.exists():
            continue

        for sdf_dir in sdf_base.iterdir():
            if not sdf_dir.is_dir():
                continue

            if is_articulated_sdf(sdf_dir):
                result = merge_combined_scene(sdf_dir)
                if result:
                    fixed_count += 1

    return fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix articulated object meshes for SceneEval"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to SceneEval input directory (e.g., input/SceneAgent)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Fix only a specific scene (e.g., scene_0)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    total_fixed = 0

    if args.scene:
        # Fix single scene
        scene_dir = input_dir / args.scene
        if not scene_dir.exists():
            print(f"Error: Scene directory not found: {scene_dir}")
            return 1

        print(f"Fixing articulated meshes in {args.scene}...")
        fixed = fix_scene_articulated_meshes(scene_dir)
        total_fixed += fixed
        print(f"  Fixed {fixed} articulated objects")
    else:
        # Fix all scenes
        scene_dirs = sorted(
            [
                d
                for d in input_dir.iterdir()
                if d.is_dir() and d.name.startswith("scene_")
            ]
        )

        if not scene_dirs:
            print(f"No scene directories found in {input_dir}")
            return 1

        print(f"Found {len(scene_dirs)} scenes to process")

        for scene_dir in scene_dirs:
            print(f"\nProcessing {scene_dir.name}...")
            fixed = fix_scene_articulated_meshes(scene_dir)
            total_fixed += fixed
            if fixed > 0:
                print(f"  Fixed {fixed} articulated objects")
            else:
                print(f"  No articulated objects to fix")

    print(f"\nTotal: Fixed {total_fixed} articulated objects")
    return 0


if __name__ == "__main__":
    exit(main())

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


def get_sdf_scale_factor(sdf_dir: Path) -> float:
    """
    Extract the uniform scale factor from an SDF file.

    Articulated objects from PartNet-Mobility often have scale elements in
    their SDF mesh definitions.

    Args:
        sdf_dir: Directory containing the SDF file

    Returns:
        Uniform scale factor (1.0 if no scale found)
    """
    import xml.etree.ElementTree as ET

    # Find SDF file in directory
    sdf_files = list(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        return 1.0

    try:
        tree = ET.parse(sdf_files[0])
        root = tree.getroot()

        # Find first scale element
        for scale_elem in root.iter("scale"):
            if scale_elem.text:
                values = [float(v) for v in scale_elem.text.strip().split()]
                if len(values) >= 3:
                    # Check if uniform scale
                    if values[0] == values[1] == values[2]:
                        return values[0]
                    else:
                        # Non-uniform scale - use average
                        return sum(values[:3]) / 3
        return 1.0
    except Exception:
        return 1.0


def merge_combined_scene(sdf_dir: Path, force: bool = False) -> Path | None:
    """
    Merge combined_scene.gltf into a single mesh file.

    Also applies:
    - 180° rotation around Y axis to match Drake's orientation (combined_scene.gltf
      is exported with a different orientation than what Drake produces)
    - Scale factor from the SDF file (PartNet-Mobility objects often have scale
      factors baked into the SDF but not the GLTF meshes)

    Args:
        sdf_dir: Directory containing combined_scene.gltf
        force: If True, regenerate even if merged file exists

    Returns:
        Path to merged file, or None if merging failed
    """
    import numpy as np

    combined_scene_path = sdf_dir / "combined_scene.gltf"
    merged_output_path = sdf_dir / "combined_merged.glb"

    # Skip if already merged (unless force)
    if merged_output_path.exists() and not force:
        return merged_output_path

    # Delete existing merged file if force
    if merged_output_path.exists() and force:
        merged_output_path.unlink()

    try:
        # Get scale factor from SDF
        scale_factor = get_sdf_scale_factor(sdf_dir)

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

            # Apply 180° rotation around Y axis to match Drake's orientation
            # combined_scene.gltf is exported from Blender with a different
            # orientation than what Drake produces when loading the SDF
            rotation_180_y = trimesh.transformations.rotation_matrix(
                np.pi, [0, 1, 0]  # 180° around Y axis
            )
            merged.apply_transform(rotation_180_y)

            # Apply SDF scale factor if not 1.0
            if scale_factor != 1.0:
                merged.apply_scale(scale_factor)
                print(f"    Applied scale factor: {scale_factor}")

            # Export as GLB (binary format preserves more data)
            merged.export(str(merged_output_path))

            print(f"    Merged {len(meshes)} meshes -> {merged_output_path.name}")
            return merged_output_path

        elif isinstance(scene, trimesh.Trimesh):
            # Already a single mesh - apply rotation and scale
            rotation_180_y = trimesh.transformations.rotation_matrix(
                np.pi, [0, 1, 0]
            )
            scene.apply_transform(rotation_180_y)
            if scale_factor != 1.0:
                scene.apply_scale(scale_factor)
                print(f"    Applied scale factor: {scale_factor}")
            scene.export(str(merged_output_path))
            print(f"    Single mesh copied -> {merged_output_path.name}")
            return merged_output_path

        else:
            print(f"    Warning: Unexpected scene type: {type(scene)}")
            return None

    except Exception as e:
        print(f"    Error merging {combined_scene_path}: {e}")
        return None


def fix_scene_articulated_meshes(scene_dir: Path, force: bool = False) -> int:
    """
    Fix all articulated meshes in a scene directory.

    Args:
        scene_dir: Path to scene directory (e.g., input/SceneAgent/scene_0)
        force: If True, regenerate even if merged file exists

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
                result = merge_combined_scene(sdf_dir, force=force)
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate merged files even if they already exist (use if scale was wrong)",
    )
    args = parser.parse_args()

    # If force flag is set, delete existing merged files first
    if args.force:
        print("Force mode: will regenerate existing merged files")

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
        fixed = fix_scene_articulated_meshes(scene_dir, force=args.force)
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
            fixed = fix_scene_articulated_meshes(scene_dir, force=args.force)
            total_fixed += fixed
            if fixed > 0:
                print(f"  Fixed {fixed} articulated objects")
            else:
                print(f"  No articulated objects to fix")

    print(f"\nTotal: Fixed {total_fixed} articulated objects")
    return 0


if __name__ == "__main__":
    exit(main())

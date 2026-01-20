#!/usr/bin/env python3
"""
Fix articulated object meshes for SceneEval using Blender.

This script uses Blender to merge articulated object meshes, preserving PBR materials
and textures that trimesh loses during concatenation.

Usage (run via Blender):
    blender --background --python scripts/fix_articulated_meshes_blender.py -- \
        /path/to/SceneEval_converted/SceneAgent_Ours_Room [--scene scene_0] [--force]

Or use the wrapper script:
    python scripts/fix_articulated_meshes.py /path/to/SceneEval_converted/SceneAgent_Ours_Room
"""

import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

# Check if running in Blender
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def is_articulated_sdf(sdf_dir: Path) -> bool:
    """Check if an SDF directory contains an articulated object.

    An articulated object is detected by either:
    1. Having multiple E_*_combined.gltf link mesh files, OR
    2. Having a combined_scene.gltf file with multiple links
    """
    # Check if there are multiple link meshes (E_*_combined.gltf files)
    link_meshes = list(sdf_dir.glob("E_*_combined.gltf"))
    if len(link_meshes) > 1:
        return True

    # Fallback: check for combined_scene.gltf (older export format)
    combined_scene = sdf_dir / "combined_scene.gltf"
    return combined_scene.exists()


def get_sdf_scale_factor(sdf_dir: Path) -> float:
    """
    Extract the uniform scale factor from an SDF file.

    Articulated objects from PartNet-Mobility often have scale elements in
    their SDF mesh definitions.
    """
    sdf_files = list(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        return 1.0

    try:
        tree = ET.parse(sdf_files[0])
        root = tree.getroot()

        for scale_elem in root.iter("scale"):
            if scale_elem.text:
                values = [float(v) for v in scale_elem.text.strip().split()]
                if len(values) >= 3:
                    if values[0] == values[1] == values[2]:
                        return values[0]
                    else:
                        return sum(values[:3]) / 3
        return 1.0
    except Exception:
        return 1.0


def clear_scene():
    """Clear all objects from the Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Also clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def merge_combined_scene_blender(sdf_dir: Path, force: bool = False) -> Path | None:
    """
    Merge articulated object meshes into a single mesh file using Blender.

    This preserves PBR materials and textures that trimesh would lose.

    Args:
        sdf_dir: Directory containing the articulated object meshes
        force: If True, regenerate even if merged file exists

    Returns:
        Path to merged file, or None if merging failed
    """
    combined_scene_path = sdf_dir / "combined_scene.gltf"
    merged_output_path = sdf_dir / "combined_merged.glb"

    # Skip if already merged (unless force)
    if merged_output_path.exists() and not force:
        return merged_output_path

    # Delete existing merged file if force
    if merged_output_path.exists() and force:
        merged_output_path.unlink()

    try:
        # Clear the scene
        clear_scene()

        # Get scale factor from SDF
        scale_factor = get_sdf_scale_factor(sdf_dir)

        imported_objects = []

        if combined_scene_path.exists():
            # Case 1: Load from combined_scene.gltf
            bpy.ops.import_scene.gltf(filepath=str(combined_scene_path))
            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        else:
            # Case 2: Import individual E_*_combined.gltf link meshes
            link_mesh_files = sorted(sdf_dir.glob("E_*_combined.gltf"))

            if not link_mesh_files:
                print(f"    Warning: No meshes found in {sdf_dir}")
                return None

            for link_file in link_mesh_files:
                try:
                    bpy.ops.import_scene.gltf(filepath=str(link_file))
                    # Get newly imported mesh objects
                    new_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
                    imported_objects.extend(new_objs)
                except Exception as e:
                    print(f"    Warning: Failed to load {link_file.name}: {e}")

        if not imported_objects:
            print(f"    Warning: No meshes found in {sdf_dir}")
            return None

        # Apply all transforms before joining
        for obj in imported_objects:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Join all meshes into one
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported_objects[0]

        if len(imported_objects) > 1:
            bpy.ops.object.join()

        # Get the joined object
        joined_obj = bpy.context.active_object

        # Apply SDF scale factor if not 1.0
        if scale_factor != 1.0:
            joined_obj.scale = (scale_factor, scale_factor, scale_factor)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            print(f"    Applied scale factor: {scale_factor}")

        # Export as GLB (preserves materials and textures)
        bpy.ops.export_scene.gltf(
            filepath=str(merged_output_path),
            export_format='GLB',
            use_selection=True,
            export_materials='EXPORT',
            export_vertex_color='ACTIVE',  # Export vertex colors
            export_texcoords=True,
            export_normals=True,
            export_image_format='AUTO',
        )

        print(f"    Merged {len(imported_objects)} meshes -> {merged_output_path.name}")
        return merged_output_path

    except Exception as e:
        print(f"    Error merging meshes in {sdf_dir}: {e}")
        import traceback
        traceback.print_exc()
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
                result = merge_combined_scene_blender(sdf_dir, force=force)
                if result:
                    fixed_count += 1

    return fixed_count


def main():
    # Parse arguments after "--" separator
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Fix articulated object meshes for SceneEval using Blender"
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
        help="Regenerate merged files even if they already exist",
    )
    args = parser.parse_args(argv)

    if not IN_BLENDER:
        print("Error: This script must be run via Blender")
        print("Usage: blender --background --python scripts/fix_articulated_meshes_blender.py -- <args>")
        return 1

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

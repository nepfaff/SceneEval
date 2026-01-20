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
    python scripts/fix_articulated_meshes.py input/SceneAgent [--scene scene_0] [--force]

    # Or use Blender directly for material preservation:
    blender --background --python scripts/fix_articulated_meshes_blender.py -- input/SceneAgent

This will:
1. Find all combined_scene.gltf files in the input directory
2. Merge all meshes into a single mesh
3. Save as combined_merged.glb (single mesh with materials preserved)

The scene_agent.py asset loader will then use combined_merged.glb if present.

NOTE: This script uses trimesh for mesh merging which may lose PBR materials.
For full material preservation, use fix_articulated_meshes_blender.py instead.
"""

import argparse
import subprocess
import sys
from pathlib import Path


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
        help="Regenerate merged files even if they already exist",
    )
    parser.add_argument(
        "--use-blender",
        action="store_true",
        default=True,
        help="Use Blender for mesh merging (preserves PBR materials, default: True)",
    )
    parser.add_argument(
        "--no-blender",
        action="store_true",
        help="Use trimesh instead of Blender (faster but loses PBR materials)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Determine whether to use Blender or trimesh
    use_blender = args.use_blender and not args.no_blender

    if use_blender:
        # Call the Blender version
        script_dir = Path(__file__).parent
        blender_script = script_dir / "fix_articulated_meshes_blender.py"

        if not blender_script.exists():
            print(f"Error: Blender script not found: {blender_script}")
            return 1

        # Build the Blender command
        cmd = [
            "blender",
            "--background",
            "--python", str(blender_script),
            "--",
            str(input_dir),
        ]

        if args.scene:
            cmd.extend(["--scene", args.scene])
        if args.force:
            cmd.append("--force")

        print(f"Running Blender for mesh merging (preserves PBR materials)...")
        result = subprocess.run(cmd)
        return result.returncode
    else:
        # Use trimesh (legacy behavior)
        return run_trimesh_merge(input_dir, args.scene, args.force)


def run_trimesh_merge(input_dir: Path, scene: str | None, force: bool) -> int:
    """Run the legacy trimesh-based mesh merging (loses PBR materials)."""
    import trimesh
    import xml.etree.ElementTree as ET

    def get_sdf_scale_factor(sdf_dir: Path) -> float:
        """Extract the uniform scale factor from an SDF file."""
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

    def merge_combined_scene(sdf_dir: Path, force: bool = False) -> Path | None:
        """Merge articulated object meshes using trimesh."""
        combined_scene_path = sdf_dir / "combined_scene.gltf"
        merged_output_path = sdf_dir / "combined_merged.glb"

        if merged_output_path.exists() and not force:
            return merged_output_path

        if merged_output_path.exists() and force:
            merged_output_path.unlink()

        try:
            scale_factor = get_sdf_scale_factor(sdf_dir)
            meshes = []

            if combined_scene_path.exists():
                scene = trimesh.load(str(combined_scene_path))
                if isinstance(scene, trimesh.Scene):
                    for node_name in scene.graph.nodes_geometry:
                        transform, geometry_name = scene.graph[node_name]
                        geometry = scene.geometry[geometry_name]
                        if isinstance(geometry, trimesh.Trimesh):
                            mesh_copy = geometry.copy()
                            mesh_copy.apply_transform(transform)
                            meshes.append(mesh_copy)
                elif isinstance(scene, trimesh.Trimesh):
                    meshes.append(scene)
            else:
                link_mesh_files = sorted(sdf_dir.glob("E_*_combined.gltf"))
                if not link_mesh_files:
                    print(f"    Warning: No meshes found in {sdf_dir}")
                    return None

                for link_file in link_mesh_files:
                    try:
                        link_mesh = trimesh.load(str(link_file))
                        if isinstance(link_mesh, trimesh.Scene):
                            for node_name in link_mesh.graph.nodes_geometry:
                                transform, geometry_name = link_mesh.graph[node_name]
                                geometry = link_mesh.geometry[geometry_name]
                                if isinstance(geometry, trimesh.Trimesh):
                                    mesh_copy = geometry.copy()
                                    mesh_copy.apply_transform(transform)
                                    meshes.append(mesh_copy)
                        elif isinstance(link_mesh, trimesh.Trimesh):
                            meshes.append(link_mesh)
                    except Exception as e:
                        print(f"    Warning: Failed to load {link_file.name}: {e}")

            if not meshes:
                print(f"    Warning: No meshes found in {sdf_dir}")
                return None

            merged = trimesh.util.concatenate(meshes)

            if scale_factor != 1.0:
                merged.apply_scale(scale_factor)
                print(f"    Applied scale factor: {scale_factor}")

            merged.export(str(merged_output_path))
            print(f"    Merged {len(meshes)} meshes -> {merged_output_path.name}")
            return merged_output_path

        except Exception as e:
            print(f"    Error merging meshes in {sdf_dir}: {e}")
            return None

    def fix_scene_articulated_meshes(scene_dir: Path, force: bool = False) -> int:
        """Fix all articulated meshes in a scene directory."""
        assets_dir = scene_dir / "assets"
        if not assets_dir.exists():
            return 0

        fixed_count = 0
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

    print("WARNING: Using trimesh for mesh merging. PBR materials will be lost.")
    print("         Use --use-blender (default) to preserve materials.")

    if force:
        print("Force mode: will regenerate existing merged files")

    total_fixed = 0

    if scene:
        scene_dir = input_dir / scene
        if not scene_dir.exists():
            print(f"Error: Scene directory not found: {scene_dir}")
            return 1

        print(f"Fixing articulated meshes in {scene}...")
        fixed = fix_scene_articulated_meshes(scene_dir, force=force)
        total_fixed += fixed
        print(f"  Fixed {fixed} articulated objects")
    else:
        scene_dirs = sorted(
            [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")]
        )

        if not scene_dirs:
            print(f"No scene directories found in {input_dir}")
            return 1

        print(f"Found {len(scene_dirs)} scenes to process")

        for scene_dir in scene_dirs:
            print(f"\nProcessing {scene_dir.name}...")
            fixed = fix_scene_articulated_meshes(scene_dir, force=force)
            total_fixed += fixed
            if fixed > 0:
                print(f"  Fixed {fixed} articulated objects")
            else:
                print(f"  No articulated objects to fix")

    print(f"\nTotal: Fixed {total_fixed} articulated objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())

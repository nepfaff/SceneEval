"""
Conversion script for scene-agent output to SceneEval input format.

Supports both old and new scene-agent output structures:

Old structure (scene_states/):
    scene_XXX/
      scene_states/final_scene/sceneeval_state.json
      generated_assets/
      floor_plan.sdf

New structure (combined_house/):
    scene_XXX/
      combined_house/sceneeval_state.json
      combined_house/house.blend
      room_*/generated_assets/
      floor_plans/floor_plan.sdf (optional)

Usage:
    python conversion/scene_agent/convert_SceneEval.py \
        ~/scene-agent/outputs/2025-11-27/21-25-16 \
        input/SceneAgent

With custom ID mapping:
    python conversion/scene_agent/convert_SceneEval.py \
        ~/scene-agent/outputs/2025-11-28/23-50-40 \
        input/SceneAgent \
        --mapping '{"0": 106, "1": 56, "2": 39, "3": 74, "4": 94}'

This copies scene-agent output to SceneEval input format:
    - sceneeval_state.json -> scene_X.json
    - generated_assets/ -> scene_X/assets/
    - floor_plan.sdf -> scene_X/floor_plan.sdf (if available)
    - house.blend -> scene_X/assets/original_scene_agent.blend (if available)
    - Articulated objects are merged into single meshes (combined_merged.glb)

Note: Scene IDs are extracted from directory names by default (e.g., scene_002 -> scene_2).
"""

import json
import shutil
import argparse
from pathlib import Path

import trimesh


def find_scene_state_path(scene_dir: Path) -> Path | None:
    """Find sceneeval_state.json in either old or new structure."""
    # New structure: combined_house/sceneeval_state.json
    new_path = scene_dir / "combined_house" / "sceneeval_state.json"
    if new_path.exists():
        return new_path

    # Old structure: scene_states/final_scene/sceneeval_state.json
    old_path = scene_dir / "scene_states" / "final_scene" / "sceneeval_state.json"
    if old_path.exists():
        return old_path

    return None


def find_generated_assets_paths(scene_dir: Path) -> list[Path]:
    """Find all generated_assets directories."""
    assets_paths = []

    # Old structure: generated_assets/ at scene root
    old_path = scene_dir / "generated_assets"
    if old_path.exists():
        assets_paths.append(old_path)

    # New structure: room_*/generated_assets/
    for room_dir in scene_dir.glob("room_*"):
        if room_dir.is_dir():
            room_assets = room_dir / "generated_assets"
            if room_assets.exists():
                assets_paths.append(room_assets)

    return assets_paths


def find_floor_plan_sdf(scene_dir: Path) -> Path | None:
    """Find floor_plan.sdf in either old or new structure."""
    # Old structure: floor_plan.sdf at scene root
    old_path = scene_dir / "floor_plan.sdf"
    if old_path.exists():
        return old_path

    # New structure: floor_plans/floor_plan.sdf
    new_path = scene_dir / "floor_plans" / "floor_plan.sdf"
    if new_path.exists():
        return new_path

    return None


def find_blend_file(scene_dir: Path) -> Path | None:
    """Find the main .blend file for the scene."""
    # New structure: combined_house/house.blend
    new_path = scene_dir / "combined_house" / "house.blend"
    if new_path.exists():
        return new_path

    # Old structure: scene_states/final_scene/scene.blend
    old_path = scene_dir / "scene_states" / "final_scene" / "scene.blend"
    if old_path.exists():
        return old_path

    return None


def is_articulated_sdf(sdf_dir: Path) -> bool:
    """Check if an SDF directory contains an articulated object (multiple links)."""
    combined_scene = sdf_dir / "combined_scene.gltf"
    if not combined_scene.exists():
        return False

    # Check if there are multiple link meshes (E_*_combined.gltf files)
    link_meshes = list(sdf_dir.glob("E_*_combined.gltf"))
    return len(link_meshes) > 1


def get_sdf_scale_factor(sdf_dir: Path) -> float:
    """Extract the uniform scale factor from an SDF file.

    Articulated objects from PartNet-Mobility often have scale elements in
    their SDF mesh definitions. This reads the first scale element found.

    Args:
        sdf_dir: Directory containing the SDF file

    Returns:
        Uniform scale factor (1.0 if no scale found or non-uniform)
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
                        # Non-uniform scale - use average (shouldn't happen often)
                        return sum(values[:3]) / 3
        return 1.0
    except Exception:
        return 1.0


def merge_articulated_meshes(sdf_dir: Path) -> Path | None:
    """
    Merge combined_scene.gltf into a single mesh file for articulated objects.

    Articulated objects from scene-agent have separate meshes for each link
    (doors, drawers, body). This merges them into a single mesh that Blender
    can import correctly as one object.

    Also applies the scale factor from the SDF file, as articulated objects
    from PartNet-Mobility often have scale factors baked into the SDF but not
    the GLTF meshes.

    Args:
        sdf_dir: Directory containing combined_scene.gltf

    Returns:
        Path to merged file (combined_merged.glb), or None if merging failed
    """
    combined_scene_path = sdf_dir / "combined_scene.gltf"
    merged_output_path = sdf_dir / "combined_merged.glb"

    # Skip if already merged
    if merged_output_path.exists():
        return merged_output_path

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
                return None

            # Concatenate all meshes into one
            merged = trimesh.util.concatenate(meshes)

            # Apply SDF scale factor if not 1.0
            if scale_factor != 1.0:
                merged.apply_scale(scale_factor)

            merged.export(str(merged_output_path))
            return merged_output_path

        elif isinstance(scene, trimesh.Trimesh):
            # Already a single mesh
            if scale_factor != 1.0:
                scene.apply_scale(scale_factor)
            scene.export(str(merged_output_path))
            return merged_output_path

        return None

    except Exception as e:
        print(f"    Warning: Failed to merge meshes in {sdf_dir.name}: {e}")
        return None


def fix_articulated_meshes_in_scene(assets_dir: Path) -> int:
    """
    Fix all articulated meshes in a scene's assets directory.

    Args:
        assets_dir: Path to scene assets directory (e.g., scene_X/assets/)

    Returns:
        Number of articulated objects fixed
    """
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
                result = merge_articulated_meshes(sdf_dir)
                if result:
                    fixed_count += 1

    return fixed_count


def convert_single_scene(scene_dir: Path, target_dir: Path, scene_id: int) -> None:
    """
    Convert a single scene-agent scene to SceneEval format.

    Args:
        scene_dir: Path to scene-agent scene directory (e.g., scene_000/)
        target_dir: Path to SceneEval input directory (e.g., input/SceneAgent/)
        scene_id: Scene ID for output naming
    """
    # Find sceneeval_state.json
    sceneeval_state_path = find_scene_state_path(scene_dir)
    if sceneeval_state_path is None:
        print(f"  Warning: sceneeval_state.json not found in {scene_dir}, skipping scene")
        return

    # Find generated_assets directories
    assets_paths = find_generated_assets_paths(scene_dir)
    if not assets_paths:
        print(f"  Warning: No generated_assets directories found in {scene_dir}, skipping scene")
        return

    # Create output directory structure
    scene_output_dir = target_dir / f"scene_{scene_id}"
    assets_output_dir = scene_output_dir / "assets"
    assets_output_dir.mkdir(parents=True, exist_ok=True)

    # Load and modify sceneeval_state.json -> scene_X.json
    output_json_path = target_dir / f"scene_{scene_id}.json"
    print(f"  Processing {sceneeval_state_path.relative_to(scene_dir)} -> {output_json_path.name}")

    with open(sceneeval_state_path, "r") as f:
        scene_data = json.load(f)

    # Update modelId and sdfPath for each object
    for obj in scene_data.get("scene", {}).get("object", []):
        old_model_id = obj.get("modelId", "")
        if old_model_id.startswith("scene-agent."):
            obj_id = old_model_id.split(".", 1)[1]
            obj["modelId"] = f"scene-agent.scene_{scene_id}__{obj_id}"

        # Fix sdfPath: remove any leading directory prefix up to "generated_assets/"
        old_sdf_path = obj.get("sdfPath", "")
        if "generated_assets/" in old_sdf_path:
            idx = old_sdf_path.index("generated_assets/")
            obj["sdfPath"] = old_sdf_path[idx + len("generated_assets/"):]
        elif old_sdf_path.startswith("generated_assets/"):
            obj["sdfPath"] = old_sdf_path[len("generated_assets/"):]

    # Set objectFrontVector for SceneAgent scenes
    scene_data["scene"]["objectFrontVector"] = [0, 1, 0]

    with open(output_json_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    # Copy generated_assets from all found directories (only sdf/, skip debug/images/geometry)
    for assets_path in assets_paths:
        for category in ["furniture", "manipuland", "wall_mounted", "ceiling_mounted"]:
            sdf_src = assets_path / category / "sdf"
            if sdf_src.exists():
                sdf_dst = assets_output_dir / category / "sdf"
                print(f"  Copying {assets_path.relative_to(scene_dir)}/{category}/sdf/ -> assets/{category}/sdf/")
                if sdf_dst.exists():
                    # Merge instead of replace
                    for item in sdf_src.iterdir():
                        dst_item = sdf_dst / item.name
                        if item.is_dir():
                            if dst_item.exists():
                                shutil.rmtree(dst_item)
                            shutil.copytree(item, dst_item)
                        else:
                            shutil.copy2(item, dst_item)
                else:
                    shutil.copytree(sdf_src, sdf_dst)

    # Copy floor_plan.sdf if it exists
    floor_plan_path = find_floor_plan_sdf(scene_dir)
    if floor_plan_path:
        output_floor_plan = scene_output_dir / "floor_plan.sdf"
        print(f"  Copying {floor_plan_path.relative_to(scene_dir)} -> scene_{scene_id}/floor_plan.sdf")
        shutil.copy2(floor_plan_path, output_floor_plan)

    # Copy .blend file for high-quality rendering
    blend_path = find_blend_file(scene_dir)
    if blend_path:
        output_blend = assets_output_dir / "original_scene_agent.blend"
        print(f"  Copying {blend_path.relative_to(scene_dir)} -> assets/original_scene_agent.blend")
        shutil.copy2(blend_path, output_blend)

    # Fix articulated objects by merging multi-link meshes into single files
    fixed_count = fix_articulated_meshes_in_scene(assets_output_dir)
    if fixed_count > 0:
        print(f"  Merged {fixed_count} articulated objects into single meshes")

    print(f"  Done: scene_{scene_id}")


def convert_scene_agent_run(source_run_dir: Path, target_dir: Path, mapping: dict = None) -> None:
    """
    Convert all scenes from a scene-agent run to SceneEval format.

    Args:
        source_run_dir: Path to scene-agent run directory
        target_dir: Path to SceneEval input directory
        mapping: Optional dict mapping source index to target scene ID
    """
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()

    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_run_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene directories
    scene_dirs = sorted([
        d for d in source_run_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])

    if not scene_dirs:
        raise ValueError(f"No scene directories found in {source_run_dir}")

    print(f"Found {len(scene_dirs)} scenes to convert")
    print(f"Source: {source_run_dir}")
    print(f"Target: {target_dir}")
    if mapping:
        print(f"Mapping: {mapping}")
    print()

    for scene_dir in scene_dirs:
        # Extract source scene ID from directory name (e.g., "scene_002" -> 2)
        dir_name = scene_dir.name
        if dir_name.startswith("scene_"):
            try:
                src_scene_id = int(dir_name[6:])
            except ValueError:
                print(f"  Warning: Could not parse scene ID from {dir_name}, skipping")
                continue
        else:
            print(f"  Warning: Unexpected directory name {dir_name}, skipping")
            continue

        # Use mapping if provided, otherwise use source scene ID
        if mapping and str(src_scene_id) in mapping:
            scene_id = mapping[str(src_scene_id)]
        else:
            scene_id = src_scene_id

        print(f"Converting {scene_dir.name} -> scene_{scene_id}...")
        convert_single_scene(scene_dir, target_dir, scene_id)
        print()

    print(f"Conversion complete! {len(scene_dirs)} scenes saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert scene-agent output to SceneEval input format"
    )
    parser.add_argument(
        "source_run_dir",
        type=Path,
        help="Path to scene-agent run directory"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to SceneEval input directory"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help='JSON mapping from source index to target ID'
    )
    args = parser.parse_args()

    mapping = None
    if args.mapping:
        mapping = json.loads(args.mapping)

    convert_scene_agent_run(args.source_run_dir, args.target_dir, mapping)


if __name__ == "__main__":
    main()

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

Note: Scene IDs are extracted from directory names by default (e.g., scene_002 -> scene_2).
"""

import json
import shutil
import argparse
from pathlib import Path


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

    # Copy generated_assets from all found directories
    for assets_path in assets_paths:
        for category in ["furniture", "manipulands", "wall_mounted", "ceiling_mounted"]:
            category_src = assets_path / category
            if category_src.exists():
                category_dst = assets_output_dir / category
                print(f"  Copying {assets_path.relative_to(scene_dir)}/{category}/ -> assets/{category}/")
                if category_dst.exists():
                    # Merge instead of replace
                    for item in category_src.iterdir():
                        dst_item = category_dst / item.name
                        if item.is_dir():
                            if dst_item.exists():
                                shutil.rmtree(dst_item)
                            shutil.copytree(item, dst_item)
                        else:
                            shutil.copy2(item, dst_item)
                else:
                    shutil.copytree(category_src, category_dst)

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

    for idx, scene_dir in enumerate(scene_dirs):
        # Use mapping if provided, otherwise extract scene ID from directory name
        if mapping and str(idx) in mapping:
            scene_id = mapping[str(idx)]
        else:
            # Extract scene ID from directory name (e.g., "scene_002" -> 2)
            dir_name = scene_dir.name
            if dir_name.startswith("scene_"):
                try:
                    scene_id = int(dir_name[6:])
                except ValueError:
                    scene_id = idx
            else:
                scene_id = idx
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

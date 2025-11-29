"""
Conversion script for scene-agent output to SceneEval input format.

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
    - scene_states/final_scene/sceneeval_state.json -> scene_X.json
    - generated_assets/ -> scene_X/assets/
    - floor_plan.sdf -> scene_X/floor_plan.sdf (for future Drake support)
"""

import json
import shutil
import argparse
from pathlib import Path


def convert_single_scene(scene_dir: Path, target_dir: Path, scene_id: int) -> None:
    """
    Convert a single scene-agent scene to SceneEval format.

    Args:
        scene_dir: Path to scene-agent scene directory (e.g., scene_000/)
        target_dir: Path to SceneEval input directory (e.g., input/SceneAgent/)
        scene_id: Scene ID for output naming
    """
    # Check that required files exist
    sceneeval_state_path = scene_dir / "scene_states" / "final_scene" / "sceneeval_state.json"
    generated_assets_path = scene_dir / "generated_assets"
    floor_plan_path = scene_dir / "floor_plan.sdf"

    if not sceneeval_state_path.exists():
        print(f"  Warning: {sceneeval_state_path} not found, skipping scene")
        return

    if not generated_assets_path.exists():
        print(f"  Warning: {generated_assets_path} not found, skipping scene")
        return

    # Create output directory structure
    scene_output_dir = target_dir / f"scene_{scene_id}"
    assets_output_dir = scene_output_dir / "assets"
    assets_output_dir.mkdir(parents=True, exist_ok=True)

    # Load and modify sceneeval_state.json -> scene_X.json
    # We need to update modelId to include scene identifier for asset lookup
    output_json_path = target_dir / f"scene_{scene_id}.json"
    print(f"  Processing {sceneeval_state_path.name} -> {output_json_path.name}")

    with open(sceneeval_state_path, "r") as f:
        scene_data = json.load(f)

    # Update modelId and sdfPath for each object
    # modelId: scene-agent.scene_X__{obj_id}
    # sdfPath: strip "generated_assets/" prefix since we copy to assets/ directly
    for obj in scene_data.get("scene", {}).get("object", []):
        old_model_id = obj.get("modelId", "")
        if old_model_id.startswith("scene-agent."):
            # Extract obj_id from "scene-agent.workstation_desk_0"
            obj_id = old_model_id.split(".", 1)[1]
            # Update to "scene-agent.scene_0__workstation_desk_0"
            obj["modelId"] = f"scene-agent.scene_{scene_id}__{obj_id}"

        # Fix sdfPath: remove "generated_assets/" prefix
        old_sdf_path = obj.get("sdfPath", "")
        if old_sdf_path.startswith("generated_assets/"):
            obj["sdfPath"] = old_sdf_path[len("generated_assets/"):]

    # Set objectFrontVector for SceneAgent scenes
    # SceneAgent uses +Y as front due to GLTF coordinate transform
    scene_data["scene"]["objectFrontVector"] = [0, 1, 0]

    with open(output_json_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    # Copy generated_assets/ -> scene_X/assets/
    # We need to copy both furniture and manipulands subdirectories
    for category in ["furniture", "manipulands"]:
        category_src = generated_assets_path / category
        if category_src.exists():
            category_dst = assets_output_dir / category
            print(f"  Copying {category}/ -> assets/{category}/")
            if category_dst.exists():
                shutil.rmtree(category_dst)
            shutil.copytree(category_src, category_dst)

    # Copy floor_plan.sdf if it exists (for future Drake support)
    if floor_plan_path.exists():
        output_floor_plan = scene_output_dir / "floor_plan.sdf"
        print(f"  Copying {floor_plan_path.name} -> scene_{scene_id}/floor_plan.sdf")
        shutil.copy2(floor_plan_path, output_floor_plan)

    # Copy original scene.blend for high-quality rendering
    # (Similar to SceneWeaver's original_sceneweaver.blend)
    scene_blend_path = scene_dir / "scene_states" / "final_scene" / "scene.blend"
    if scene_blend_path.exists():
        output_blend = assets_output_dir / "original_scene_agent.blend"
        print(f"  Copying scene.blend -> assets/original_scene_agent.blend")
        shutil.copy2(scene_blend_path, output_blend)

    print(f"  Done: scene_{scene_id}")


def convert_scene_agent_run(source_run_dir: Path, target_dir: Path, mapping: dict = None) -> None:
    """
    Convert all scenes from a scene-agent run to SceneEval format.

    Args:
        source_run_dir: Path to scene-agent run directory
            (e.g., ~/scene-agent/outputs/2025-11-27/21-25-16/)
        target_dir: Path to SceneEval input directory (e.g., input/SceneAgent/)
        mapping: Optional dict mapping source index to target scene ID
            (e.g., {"0": 106, "1": 56} maps scene_000 -> scene_106)
    """
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()

    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_run_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene directories (scene_000, scene_001, etc.)
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
        # Use mapping if provided, otherwise use sequential index
        if mapping and str(idx) in mapping:
            scene_id = mapping[str(idx)]
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
        help="Path to scene-agent run directory (e.g., ~/scene-agent/outputs/2025-11-27/21-25-16)"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to SceneEval input directory (e.g., input/SceneAgent)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help='JSON mapping from source index to target ID, e.g., \'{"0": 106, "1": 56}\''
    )
    args = parser.parse_args()

    mapping = None
    if args.mapping:
        mapping = json.loads(args.mapping)

    convert_scene_agent_run(args.source_run_dir, args.target_dir, mapping)


if __name__ == "__main__":
    main()

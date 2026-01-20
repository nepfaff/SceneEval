#!/usr/bin/env python3
"""
Re-render articulated objects for a single scene.

This script selectively re-renders only articulated objects (those with combined_merged.glb)
in a scene, leaving non-articulated object renders unchanged. This is much faster than
re-rendering all objects when only articulated meshes have been fixed.

Usage:
    python scripts/rerender_articulated_objects.py \
        --input-dir /path/to/SceneEval_converted \
        --output-dir /path/to/SceneEval_results \
        --method SceneAgent_Ours_Room \
        --scene-id 0

The script will:
1. Load the scene JSON to get object list
2. Identify which objects use articulated assets (have combined_merged.glb)
3. Re-render only those objects (obj_solo and obj_size views)
4. Save renders to the existing output directory structure
"""

import argparse
import json
import logging
import pathlib
import sys

# Add project root to path for imports
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def get_articulated_obj_ids(
    scene_json_path: pathlib.Path,
    input_dir: pathlib.Path,
) -> list[str]:
    """
    Find objects in scene that use articulated assets (have combined_merged.glb).

    Args:
        scene_json_path: Path to the scene JSON file
        input_dir: Root input directory containing scene assets

    Returns:
        List of blender_obj_ids (format: idx{index}_{modelId}) that need re-rendering
    """
    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    articulated_obj_ids = []

    # Get scene directory for assets
    scene_name = scene_json_path.stem  # e.g., "scene_0"
    scene_assets_dir = scene_json_path.parent / scene_name / "assets"

    # Handle different JSON formats
    # SceneAgent format: scene_data["scene"]["object"] with "sdfPath"
    # Other formats: scene_data["objects"] with "model_id" or "assetId"
    if "scene" in scene_data and "object" in scene_data["scene"]:
        objects = scene_data["scene"]["object"]
    else:
        objects = scene_data.get("objects", [])

    for obj in objects:
        obj_id = obj.get("id") or obj.get("obj_id")
        if not obj_id:
            continue

        # Get object index and modelId for constructing Blender object name
        # BlenderScene uses format: idx{index}_{modelId}
        obj_index = obj.get("index")
        model_id = obj.get("modelId") or obj.get("model_id") or obj.get("assetId")

        if obj_index is None or not model_id:
            logger.warning(f"Object {obj_id} missing index or modelId, skipping")
            continue

        # Construct the Blender object ID (same format as BlenderScene)
        blender_obj_id = f"idx{obj_index}_{model_id}"

        # Get the SDF path - SceneAgent uses "sdfPath"
        sdf_path = obj.get("sdfPath")
        if sdf_path:
            # sdfPath format: "furniture/sdf/nightstand_1767831488/nightstand.sdf"
            # We need the directory containing the SDF
            sdf_dir = scene_assets_dir / pathlib.Path(sdf_path).parent
            merged_glb = sdf_dir / "combined_merged.glb"
            if merged_glb.exists():
                articulated_obj_ids.append(blender_obj_id)
                logger.info(f"Found articulated object: {blender_obj_id} (merged: {merged_glb})")
            continue

        # Fallback for other formats - parse model_id to find asset location
        # Format: dataset:category/asset_name (e.g., "sa:furniture/cabinet_001")
        if ":" in model_id:
            _, asset_path = model_id.split(":", 1)
        else:
            asset_path = model_id

        # Check different possible locations for the SDF
        possible_sdf_dirs = []

        # SceneAgent format: assets/furniture/sdf/asset_name/
        if "/" in asset_path:
            category, asset_name = asset_path.split("/", 1)
            possible_sdf_dirs.append(scene_assets_dir / category / "sdf" / asset_name)

        # Also check directly under assets
        possible_sdf_dirs.append(scene_assets_dir / asset_path)

        # Check if any of these directories have combined_merged.glb
        for sdf_dir in possible_sdf_dirs:
            merged_glb = sdf_dir / "combined_merged.glb"
            if merged_glb.exists():
                articulated_obj_ids.append(blender_obj_id)
                logger.info(f"Found articulated object: {blender_obj_id} (merged: {merged_glb})")
                break

    return articulated_obj_ids


def rerender_objects(
    scene_json_path: pathlib.Path,
    obj_ids: list[str],
    output_dir: pathlib.Path,
    input_dir: pathlib.Path,
) -> None:
    """
    Re-render specific objects using Blender.

    Args:
        scene_json_path: Path to the scene JSON file
        obj_ids: List of object IDs to re-render
        output_dir: Output directory for renders
        input_dir: Input directory for assets
    """
    import bpy
    import numpy as np
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from scenes import Scene, SceneState, SceneConfig, BlenderConfig, TrimeshConfig
    from assets import Retriever

    if not obj_ids:
        logger.info("No objects to re-render")
        return

    logger.info(f"Re-rendering {len(obj_ids)} articulated objects: {obj_ids}")

    # Load configuration using Hydra
    config_path = PROJECT_ROOT / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = compose(config_name="config")

    # Set up scene config
    scene_cfg = SceneConfig(**cfg.scene)
    blender_cfg = BlenderConfig(**cfg.blender)
    trimesh_cfg = TrimeshConfig(**cfg.trimesh)

    # Load mesh retriever
    # For SceneAgent, we need the scene_agent asset config
    dataset_cfgs = {"scene_agent": cfg.assets.scene_agent}

    # Update the dataset root path to point to the correct input directory
    dataset_cfgs["scene_agent"] = OmegaConf.to_container(cfg.assets.scene_agent, resolve=True)
    dataset_cfgs["scene_agent"]["dataset_root_path"] = str(input_dir)
    dataset_cfgs["scene_agent"] = OmegaConf.create(dataset_cfgs["scene_agent"])

    mesh_retriever = Retriever(dataset_cfgs)

    # Load scene state
    scene_state = SceneState(scene_json_path)

    # Get scene output directory
    scene_output_dir = output_dir / scene_state.name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    # Create the scene (loads all objects into Blender)
    scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, scene_output_dir)

    # Get render output directory
    render_dir = scene_output_dir / blender_cfg.object_render_subdir
    render_dir.mkdir(parents=True, exist_ok=True)

    # Re-render only the specified objects
    for obj_id in obj_ids:
        if obj_id not in scene.blender_scene.b_objs:
            logger.warning(f"Object {obj_id} not found in Blender scene, skipping")
            continue

        b_obj = scene.blender_scene.b_objs[obj_id]

        # Render obj_solo (front view without surroundings)
        solo_path = render_dir / f"{obj_id}.{blender_cfg.render_file_format.lower()}"
        logger.info(f"Rendering solo view: {solo_path}")
        scene.blender_scene.render_one_obj(
            b_obj,
            pathlib.Path(blender_cfg.object_render_subdir) / f"{obj_id}.{blender_cfg.render_file_format.lower()}",
            hide_others=True,
            zoom_out=False,
            with_human_reference=False,
        )

        # Render obj_size (front view with human reference)
        size_path = render_dir / f"size_{obj_id}.{blender_cfg.render_file_format.lower()}"
        logger.info(f"Rendering size reference view: {size_path}")
        scene.blender_scene.render_one_obj(
            b_obj,
            pathlib.Path(blender_cfg.object_render_subdir) / f"size_{obj_id}.{blender_cfg.render_file_format.lower()}",
            hide_others=True,
            zoom_out=False,
            with_human_reference=True,
            bird_view_degree=80,
        )

    logger.info(f"Re-rendered {len(obj_ids)} objects to {render_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-render articulated objects for a single scene"
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Root input directory (e.g., SceneEval_converted/SceneAgent_Ours_Room)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Root output directory (e.g., SceneEval_results/SceneAgent_Ours_Room)",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        required=True,
        help="Scene ID to process (e.g., 0 for scene_0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list articulated objects, don't render",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find scene JSON
    scene_json_path = args.input_dir / f"scene_{args.scene_id}.json"
    if not scene_json_path.exists():
        logger.error(f"Scene JSON not found: {scene_json_path}")
        return 1

    logger.info(f"Processing scene: {scene_json_path}")

    # Find articulated objects
    articulated_obj_ids = get_articulated_obj_ids(scene_json_path, args.input_dir)

    if not articulated_obj_ids:
        logger.info("No articulated objects found in scene")
        return 0

    logger.info(f"Found {len(articulated_obj_ids)} articulated objects: {articulated_obj_ids}")

    if args.dry_run:
        print(f"Would re-render {len(articulated_obj_ids)} objects:")
        for obj_id in articulated_obj_ids:
            print(f"  - {obj_id}")
        return 0

    # Re-render the objects
    rerender_objects(scene_json_path, articulated_obj_ids, args.output_dir, args.input_dir)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

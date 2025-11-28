"""
Worker function for parallel scene evaluation.

This module contains the worker function that evaluates a single scene.
It's designed to be called from a ProcessPoolExecutor.
"""

import json
import pathlib
import traceback
from typing import Any

from omegaconf import OmegaConf


def evaluate_scene_worker(
    method: str,
    scene_file: pathlib.Path,
    output_dir: pathlib.Path,
    metrics_to_run: list[str],
    metric_configs_dict: dict,
    scene_cfg_dict: dict,
    blender_cfg_dict: dict,
    trimesh_cfg_dict: dict,
    vlm_name: str,
    vlm_config_dict: dict,
    dataset_cfgs_dict: dict,
    annotation_file: str,
    use_existing_matching: bool,
    use_empty_matching_result: bool,
    save_blend_file: bool,
    normal_render_tasks: list[str] | None,
    verbose: bool,
) -> dict:
    """
    Evaluate a single scene in its own process.

    This is a module-level function (not a method) to ensure pickling works
    with ProcessPoolExecutor.

    Args:
        method: The method name (e.g., "LayoutVLM", "SceneWeaver")
        scene_file: Path to the scene JSON file
        output_dir: Output directory for this scene
        metrics_to_run: List of metric names to run
        metric_configs_dict: Dictionary of metric configurations (converted from dataclasses)
        scene_cfg_dict: Scene configuration as dict
        blender_cfg_dict: Blender configuration as dict
        trimesh_cfg_dict: Trimesh configuration as dict
        vlm_name: Name of the VLM to use
        vlm_config_dict: VLM configuration as dict
        dataset_cfgs_dict: Asset dataset configurations as dict
        annotation_file: Path to the annotation file
        use_existing_matching: Whether to use existing object matching results
        use_empty_matching_result: Whether to use empty matching result
        save_blend_file: Whether to save the Blender file
        normal_render_tasks: List of render tasks to run
        verbose: Whether to run metrics in verbose mode

    Returns:
        dict with keys: scene_id, method, success, error (if failed), output_file (if success)
    """
    # Import inside worker to ensure fresh Blender state per process
    from scenes import Scene, SceneState, SceneConfig, BlenderConfig, TrimeshConfig, Annotations
    from metrics import MetricRegistry, ObjMatching, ObjMatchingResults
    from assets import Retriever
    from vlm import VLMRegistry

    scene_id = scene_file.stem

    try:
        # Reconstruct config objects from dicts
        scene_cfg = SceneConfig(**scene_cfg_dict)
        blender_cfg = BlenderConfig(**blender_cfg_dict)
        trimesh_cfg = TrimeshConfig(**trimesh_cfg_dict)

        # Reconstruct DictConfig objects for asset datasets (Retriever needs attribute access)
        dataset_cfgs_dictconfig = {
            name: OmegaConf.create(cfg) for name, cfg in dataset_cfgs_dict.items()
        }

        # Create mesh retriever
        mesh_retriever = Retriever(dataset_cfgs_dictconfig)

        # Create VLM instance for this worker
        # Reconstruct DictConfig for VLM (needs attribute access)
        vlm_cfg = OmegaConf.create(vlm_config_dict)

        vlm = VLMRegistry.instantiate_vlm(vlm_name, vlm_cfg)

        # Load annotations
        annotations = Annotations(annotation_file)

        # Load scene state
        try:
            scene_state = SceneState(scene_file)
        except Exception as e:
            print(f"[{scene_id}] Error loading scene: {e}")
            scene_state = SceneState(pathlib.Path("./input/empty_scene.json"))
            scene_state.name = scene_id

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create scene
        scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, output_dir)

        # Save blend file if configured
        if save_blend_file:
            scene.blender_scene.save_blend()

        # Render tasks if configured
        if normal_render_tasks:
            _render_scene(scene, normal_render_tasks)

        # Get annotation
        scene_file_id = scene_state.name.split("_")[-1]
        annotation = annotations[int(scene_file_id)]

        # Handle object matching
        if use_empty_matching_result:
            matching_result = ObjMatchingResults(per_category={}, not_matched_objs=[], actual_categories={})
        else:
            matching_result = _get_obj_matching(scene, annotation, vlm, use_existing_matching)

        # Reconstruct metric configs from dicts
        metric_configs = _reconstruct_metric_configs(metric_configs_dict)

        # Initialize output JSON
        output_json = {
            "method": method,
            "scene_id": scene_file_id,
            "description": annotation.description,
            "obj_ids": scene.get_obj_ids(),
            "object_descriptions": [scene.obj_descriptions[obj_id] for obj_id in scene.get_obj_ids()],
            "object_matching_per_category": matching_result.per_category,
            "not_matched_objects": matching_result.not_matched_objs,
            "metrics": metrics_to_run,
            "results": {}
        }

        # Common params for all metrics
        common_metric_params = {
            "scene": scene,
            "annotation": annotation,
            "vlm": vlm,
            "matching_result": matching_result,
            "output_dir": output_dir
        }

        # Run metrics in optimized order
        ordered_metrics = MetricRegistry.get_optimized_execution_order(metrics_to_run)

        for metric_name in ordered_metrics:
            print(f"[{scene_id}] Running metric: {metric_name}")

            metric_instance = MetricRegistry.instantiate_metric(
                metric_name, metric_configs, **common_metric_params
            )

            result = metric_instance.run(verbose)
            output_json["results"][metric_name] = {
                "message": result.message,
                "data": result.data
            }

            # Save intermediate results
            output_file = output_dir / "eval_result.json"
            with open(output_file, "w") as f:
                json.dump(output_json, f, indent=4, default=str)

        print(f"[{scene_id}] Completed successfully")

        return {
            "scene_id": scene_id,
            "method": method,
            "success": True,
            "output_file": str(output_file)
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"[{scene_id}] FAILED: {error_msg}")

        return {
            "scene_id": scene_id,
            "method": method,
            "success": False,
            "error": error_msg
        }


def _render_scene(scene, render_tasks: list[str]) -> None:
    """Render scene with specified tasks."""
    for render_task in render_tasks:
        match render_task:
            case "scene_top":
                scene.blender_scene.render_scene_from_top()
            case "obj_solo":
                scene.blender_scene.render_all_objs_front_solo()
            case "obj_size":
                scene.blender_scene.render_all_objs_front_size_reference()
            case "obj_surroundings":
                scene.blender_scene.render_all_objs_front_surroundings()
            case "obj_global_top":
                scene.blender_scene.render_all_objs_global_top()


def _get_obj_matching(scene, annotation, vlm, use_existing_matching: bool):
    """Get object matching results for the scene."""
    from metrics import ObjMatching, ObjMatchingResults
    import json

    matching_result_file = scene.output_dir / "obj_matching_result.json"

    if use_existing_matching and matching_result_file.exists():
        with open(matching_result_file, "r") as f:
            matching_result = ObjMatchingResults.from_dict(json.load(f))
    else:
        obj_matching = ObjMatching(scene, annotation, vlm)
        obj_matching_result = obj_matching.run()
        matching_result = obj_matching_result.data["matching_result"]

        # Save the matching result
        with open(matching_result_file, "w") as f:
            json.dump(matching_result.to_dict(), f, indent=4)

    return matching_result


def _reconstruct_metric_configs(metric_configs_dict: dict) -> dict:
    """
    Reconstruct metric config objects from dictionaries.

    The metric configs are dataclasses that need to be reconstructed.
    We use a simple proxy approach since the exact types aren't critical.
    """
    from metrics import MetricRegistry

    reconstructed = {}

    for metric_name, config_dict in metric_configs_dict.items():
        if config_dict is None:
            continue

        # Try to recreate using the registry's config class
        config_instance = MetricRegistry.create_config(metric_name, config_dict)
        if config_instance is not None:
            reconstructed[metric_name] = config_instance

    return reconstructed

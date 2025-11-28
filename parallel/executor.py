"""
Parallel scene evaluator using ProcessPoolExecutor.

This module provides the ParallelSceneEvaluator class which manages
parallel evaluation of multiple scenes.
"""

import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Any

from tqdm import tqdm

from .worker import evaluate_scene_worker


class ParallelSceneEvaluator:
    """
    Manages parallel scene evaluation using ProcessPoolExecutor.

    Following the pattern from scene-agent/experiments/indoor_scene_generation.py
    """

    def __init__(self, num_workers: int = 20):
        """
        Initialize the parallel evaluator.

        Args:
            num_workers: Number of parallel workers (processes)
        """
        self.num_workers = num_workers

    def evaluate_scenes(
        self,
        scenes_per_method: dict[str, list[pathlib.Path]],
        output_base_dir: pathlib.Path,
        model_output_dir_names: dict[str, str],
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
        method_use_simple_architecture: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """
        Evaluate all scenes in parallel.

        Args:
            scenes_per_method: Dictionary mapping method names to lists of scene files
            output_base_dir: Base output directory
            model_output_dir_names: Dictionary mapping method names to output directory names
            metrics_to_run: List of metric names to run
            metric_configs_dict: Dictionary of metric configurations
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
            method_use_simple_architecture: List of methods that use simple architecture

        Returns:
            Tuple of (successful_results, failed_results)
        """
        # Flatten all scenes into a list of tasks
        tasks = []
        for method, scene_files in scenes_per_method.items():
            # Adjust scene config for this method
            method_scene_cfg = scene_cfg_dict.copy()
            method_scene_cfg["use_simple_architecture"] = method in method_use_simple_architecture

            for scene_file in scene_files:
                output_dir = output_base_dir / model_output_dir_names[method] / scene_file.stem
                tasks.append({
                    "method": method,
                    "scene_file": scene_file,
                    "output_dir": output_dir,
                    "metrics_to_run": metrics_to_run,
                    "metric_configs_dict": metric_configs_dict,
                    "scene_cfg_dict": method_scene_cfg,
                    "blender_cfg_dict": blender_cfg_dict,
                    "trimesh_cfg_dict": trimesh_cfg_dict,
                    "vlm_name": vlm_name,
                    "vlm_config_dict": vlm_config_dict,
                    "dataset_cfgs_dict": dataset_cfgs_dict,
                    "annotation_file": annotation_file,
                    "use_existing_matching": use_existing_matching,
                    "use_empty_matching_result": use_empty_matching_result,
                    "save_blend_file": save_blend_file,
                    "normal_render_tasks": normal_render_tasks,
                    "verbose": verbose,
                })

        total_scenes = len(tasks)
        num_workers = min(self.num_workers, total_scenes)

        print(f"\n{'='*60}")
        print(f"Parallel Evaluation: {total_scenes} scenes with {num_workers} workers")
        print(f"{'='*60}\n")

        successes = []
        failures = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task: dict[Future, dict] = {}
            for task in tasks:
                future = executor.submit(evaluate_scene_worker, **task)
                future_to_task[future] = task
                print(f"Submitted: {task['method']}/{task['scene_file'].stem}")

            print()

            # Process results with tqdm progress bar
            with tqdm(total=total_scenes, desc="Evaluating scenes", unit="scene") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    scene_id = task["scene_file"].stem

                    try:
                        result = future.result()
                        if result["success"]:
                            successes.append(result)
                            pbar.set_postfix_str(f"{scene_id}: OK")
                        else:
                            failures.append(result)
                            pbar.set_postfix_str(f"{scene_id}: FAIL")
                    except Exception as e:
                        # Handle executor-level errors
                        failures.append({
                            "scene_id": scene_id,
                            "method": task["method"],
                            "success": False,
                            "error": str(e)
                        })
                        pbar.set_postfix_str(f"{scene_id}: ERROR")

                    pbar.update(1)

        return successes, failures

    @staticmethod
    def print_summary(successes: list[dict], failures: list[dict]) -> None:
        """Print evaluation summary."""
        total = len(successes) + len(failures)

        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total scenes: {total}")
        print(f"Successful: {len(successes)}")
        print(f"Failed: {len(failures)}")

        if failures:
            print(f"\nFailed scenes:")
            for fail in failures:
                error_preview = fail.get("error", "Unknown error")[:100]
                print(f"  - {fail['method']}/{fail['scene_id']}: {error_preview}...")

        print()

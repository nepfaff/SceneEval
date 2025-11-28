"""Drake-based collision metrics using convex decomposition.

This module provides collision detection metrics using Drake's collision geometry
with both CoACD and VHACD convex decomposition methods. These are more accurate
for physics simulation purposes than trimesh-based collision detection.
"""

import pathlib
import statistics
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric
from .drake_utils import (
    create_drake_plant_from_scene,
    detect_penetrating_pairs,
)


@dataclass
class DrakeCollisionMetricConfigCoACD:
    """Configuration for the Drake collision metric using CoACD.

    Attributes:
        penetration_threshold: Minimum penetration depth to report (meters).
        coacd_threshold: CoACD decomposition threshold (0.01-0.1 typical).
    """

    penetration_threshold: float = 0.001
    coacd_threshold: float = 0.05


@dataclass
class DrakeCollisionMetricConfigVHACD:
    """Configuration for the Drake collision metric using VHACD.

    Attributes:
        penetration_threshold: Minimum penetration depth to report (meters).
    """

    penetration_threshold: float = 0.001


class DrakeCollisionMetricBase(BaseMetric):
    """Base class for Drake collision metrics.

    Detects collisions using Drake's collision geometry with convex decomposition.
    Subclasses specify which decomposition method to use.
    """

    # Subclasses must define these
    decomposition_method: Literal["coacd", "vhacd"]
    drake_scene_folder: str

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        """Initialize the metric.

        Args:
            scene: The scene to evaluate.
            output_dir: Output directory for saving Drake scene files.
            cfg: The configuration for the metric.
        """
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def run(self, verbose: bool = False) -> MetricResult:
        """Run the metric.

        Args:
            verbose: Whether to print verbose output.

        Returns:
            MetricResult with collision data.
        """
        # Use output directory for Drake files (persisted for debugging/inspection).
        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get coacd_threshold if applicable.
        coacd_threshold = getattr(self.cfg, "coacd_threshold", 0.05)

        # Create Drake plant (time_step=0 for static collision query).
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene(
            scene=self.scene,
            time_step=0.0,  # Static query, no simulation.
            temp_dir=drake_scene_dir,
            weld_to_world=[],
            coacd_threshold=coacd_threshold,
            decomposition_method=self.decomposition_method,
        )

        # Build diagram and create context.
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Detect penetrating pairs.
        penetrating_pairs = detect_penetrating_pairs(
            plant=plant,
            scene_graph=scene_graph,
            context=context,
            threshold=self.cfg.penetration_threshold,
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Build collision results for object-object collisions.
        collision_results = {
            obj_id: {
                "in_collision": False,
                "colliding_with": [],
                "collision_details": [],
            }
            for obj_id in self.scene.get_obj_ids()
        }

        # Build floor collision results separately.
        floor_collision_results = {
            obj_id: {
                "in_collision_with_floor": False,
                "penetration_depth": 0.0,
            }
            for obj_id in self.scene.get_obj_ids()
        }

        # Process penetrating pairs.
        for obj_a, obj_b, depth in penetrating_pairs:
            # Handle floor_plan collisions separately.
            if obj_a == "floor_plan":
                if obj_b in floor_collision_results:
                    floor_collision_results[obj_b]["in_collision_with_floor"] = True
                    floor_collision_results[obj_b]["penetration_depth"] = max(
                        floor_collision_results[obj_b]["penetration_depth"], depth
                    )
                continue
            if obj_b == "floor_plan":
                if obj_a in floor_collision_results:
                    floor_collision_results[obj_a]["in_collision_with_floor"] = True
                    floor_collision_results[obj_a]["penetration_depth"] = max(
                        floor_collision_results[obj_a]["penetration_depth"], depth
                    )
                continue

            # Skip if not in our object list (shouldn't happen but be safe).
            if obj_a not in collision_results or obj_b not in collision_results:
                continue

            # Update collision results for both objects.
            collision_results[obj_a]["in_collision"] = True
            collision_results[obj_a]["colliding_with"].append(obj_b)
            collision_results[obj_a]["collision_details"].append({
                "other_obj": obj_b,
                "penetration_depth": depth,
            })

            collision_results[obj_b]["in_collision"] = True
            collision_results[obj_b]["colliding_with"].append(obj_a)
            collision_results[obj_b]["collision_details"].append({
                "other_obj": obj_a,
                "penetration_depth": depth,
            })

        # Compute aggregate statistics for object-object collisions.
        num_obj_in_collision = sum(
            obj_result["in_collision"] for obj_result in collision_results.values()
        )
        scene_in_collision = num_obj_in_collision > 0

        # Count unique collision pairs and compute depth statistics.
        all_depths = []
        num_collision_pairs = 0
        seen_pairs = set()

        for obj_id, obj_result in collision_results.items():
            for detail in obj_result["collision_details"]:
                pair_key = tuple(sorted([obj_id, detail["other_obj"]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_depths.append(detail["penetration_depth"])
                    num_collision_pairs += 1

        max_penetration_depth = max(all_depths) if all_depths else 0.0
        min_penetration_depth = min(all_depths) if all_depths else 0.0
        mean_penetration_depth = sum(all_depths) / len(all_depths) if all_depths else 0.0
        median_penetration_depth = statistics.median(all_depths) if all_depths else 0.0

        # Compute floor collision statistics.
        num_obj_in_floor_collision = sum(
            1 for r in floor_collision_results.values() if r["in_collision_with_floor"]
        )
        floor_depths = [
            r["penetration_depth"]
            for r in floor_collision_results.values()
            if r["in_collision_with_floor"]
        ]
        max_floor_penetration = max(floor_depths) if floor_depths else 0.0
        mean_floor_penetration = sum(floor_depths) / len(floor_depths) if floor_depths else 0.0

        method_name = self.decomposition_method.upper()
        result = MetricResult(
            message=(
                f"Drake collision ({method_name}): scene_in_collision={scene_in_collision}, "
                f"{num_obj_in_collision}/{len(self.scene.get_obj_ids())} objects in collision, "
                f"{num_collision_pairs} collision pairs, "
                f"max_depth={max_penetration_depth:.4f}m; "
                f"floor_collisions={num_obj_in_floor_collision}"
            ),
            data={
                # Object-object collision stats.
                "scene_in_collision": scene_in_collision,
                "num_obj_in_collision": num_obj_in_collision,
                "num_collision_pairs": num_collision_pairs,
                "max_penetration_depth": max_penetration_depth,
                "min_penetration_depth": min_penetration_depth,
                "mean_penetration_depth": mean_penetration_depth,
                "median_penetration_depth": median_penetration_depth,
                "collision_results": collision_results,
                # Floor collision stats.
                "num_obj_in_floor_collision": num_obj_in_floor_collision,
                "max_floor_penetration_depth": max_floor_penetration,
                "mean_floor_penetration_depth": mean_floor_penetration,
                "floor_collision_results": floor_collision_results,
                # Decomposition method used.
                "decomposition_method": self.decomposition_method,
            },
        )

        if verbose:
            print(f"\n{result.message}\n")

        return result


@register_non_vlm_metric(config_class=DrakeCollisionMetricConfigCoACD)
class DrakeCollisionMetricCoACD(DrakeCollisionMetricBase):
    """Drake collision metric using CoACD convex decomposition.

    CoACD (Convex Approximate Convex Decomposition) often inflates meshes
    to ensure convexity, so collisions detected here are the ones that
    will actually cause objects to move in Drake physics simulation.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "coacd"
    drake_scene_folder: str = "drake_collision_coacd"


@register_non_vlm_metric(config_class=DrakeCollisionMetricConfigVHACD)
class DrakeCollisionMetricVHACD(DrakeCollisionMetricBase):
    """Drake collision metric using VHACD convex decomposition.

    VHACD (Volumetric Hierarchical Approximate Convex Decomposition) provides
    an alternative decomposition method that may produce different collision
    geometry compared to CoACD.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "vhacd"
    drake_scene_folder: str = "drake_collision_vhacd"

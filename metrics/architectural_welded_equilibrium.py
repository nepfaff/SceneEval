"""Architectural welded equilibrium metrics using Drake physics simulation.

This module provides equilibrium metrics where architectural objects (floor, wall,
ceiling supported) are welded to world, while objects supported by other objects
remain free to move. This simulates a realistic scenario where large static
architectural elements are fixed but interactive objects can move.
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric
from .drake_utils import (
    create_drake_plant_from_scene,
    measure_displacement,
    run_simulation,
)


@dataclass
class ArchitecturalWeldedEquilibriumMetricConfigCoACD:
    """Configuration for the architectural welded equilibrium metric using CoACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        coacd_threshold: CoACD decomposition threshold.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        save_simulation_html: If True, save meshcat visualization to HTML file.
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    coacd_threshold: float = 0.05
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    save_simulation_html: bool = False
    hydroelastic_modulus: float | None = None
    weld_floor_objects: bool = False


@dataclass
class ArchitecturalWeldedEquilibriumMetricConfigVHACD:
    """Configuration for the architectural welded equilibrium metric using VHACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        save_simulation_html: If True, save meshcat visualization to HTML file.
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    save_simulation_html: bool = False
    hydroelastic_modulus: float | None = None
    weld_floor_objects: bool = False


class ArchitecturalWeldedEquilibriumMetricBase(BaseMetric):
    """Base class for architectural welded equilibrium metrics.

    This metric welds objects supported by architectural elements (floor, wall,
    ceiling) to world, while letting objects supported by other objects move
    freely. This represents a realistic simulation where large static furniture
    is fixed but smaller interactive objects can move.

    The primary metric is mean_displacement (lower is better).
    Subclasses specify which convex decomposition method to use.
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

    def _get_architectural_objects(self) -> list[str]:
        """Get object IDs for objects supported by architectural elements.

        Returns:
            List of object IDs that are supported by wall, ceiling, or optionally floor.
        """
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        # Build list of support types to weld (wall/ceiling always, floor optionally)
        architectural_support_types = ["wall", "ceiling"]
        if self.cfg.weld_floor_objects:
            architectural_support_types.append("ground")

        architectural_objects = []
        for obj_id, support_type in support_types.items():
            if support_type in architectural_support_types:
                architectural_objects.append(obj_id)

        return architectural_objects

    def run(self, verbose: bool = False) -> MetricResult:
        """Run the metric.

        Args:
            verbose: Whether to print verbose output.

        Returns:
            MetricResult with equilibrium data.
        """
        # Use output directory for Drake files (persisted for debugging/inspection).
        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get architectural objects to weld.
        objects_to_weld = self._get_architectural_objects()
        if verbose and objects_to_weld:
            print(f"Welding architectural objects: {objects_to_weld}")

        # Get coacd_threshold if applicable.
        coacd_threshold = getattr(self.cfg, "coacd_threshold", 0.05)
        vhacd_max_convex_hulls = getattr(self.cfg, "vhacd_max_convex_hulls", 64)

        # Get hydroelastic_modulus if applicable.
        hydroelastic_modulus = getattr(self.cfg, "hydroelastic_modulus", None)

        # Create Drake plant with dynamics (time_step > 0).
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene(
            scene=self.scene,
            time_step=self.cfg.time_step,
            temp_dir=drake_scene_dir / "simulation",
            weld_to_world=objects_to_weld,
            coacd_threshold=coacd_threshold,
            vhacd_max_convex_hulls=vhacd_max_convex_hulls,
            decomposition_method=self.decomposition_method,
            hydroelastic_modulus=hydroelastic_modulus,
        )

        # Determine HTML output path if visualization is enabled.
        html_path = None
        if getattr(self.cfg, "save_simulation_html", False):
            html_path = drake_scene_dir / "simulation" / "simulation.html"

        # Run simulation.
        diagram, initial_context, final_context = run_simulation(
            builder=builder,
            plant=plant,
            simulation_time=self.cfg.simulation_time,
            scene_graph=scene_graph,
            output_html_path=html_path,
        )

        # Get plant contexts.
        initial_plant_context = plant.GetMyContextFromRoot(initial_context)
        final_plant_context = plant.GetMyContextFromRoot(final_context)

        # Measure displacement for non-welded objects only.
        non_welded_obj_id_to_model_name = {
            obj_id: model_name
            for obj_id, model_name in obj_id_to_model_name.items()
            if obj_id not in objects_to_weld
        }

        displacement_results = measure_displacement(
            plant=plant,
            initial_context=initial_plant_context,
            final_context=final_plant_context,
            obj_id_to_model_name=non_welded_obj_id_to_model_name,
        )

        # Compute per-object stability and aggregate statistics.
        per_object_results = {}
        displacements = []
        rotations = []

        # Process non-welded objects.
        for obj_id, data in displacement_results.items():
            displacement = data["displacement"]
            rotation = data["rotation"]

            # Check if stable (within thresholds).
            stable = (
                not np.isnan(displacement)
                and not np.isnan(rotation)
                and displacement <= self.cfg.displacement_threshold
                and rotation <= self.cfg.rotation_threshold
            )

            per_object_results[obj_id] = {
                "stable": stable,
                "displacement": displacement,
                "rotation": rotation,
                "initial_position": data["initial_position"],
                "final_position": data["final_position"],
                "welded": False,
            }

            # Collect valid values for statistics.
            if not np.isnan(displacement):
                displacements.append(displacement)
            if not np.isnan(rotation):
                rotations.append(rotation)

        # Add welded objects (they have zero displacement by definition).
        for obj_id in objects_to_weld:
            per_object_results[obj_id] = {
                "stable": True,  # Welded objects don't move.
                "displacement": 0.0,
                "rotation": 0.0,
                "initial_position": None,
                "final_position": None,
                "welded": True,
                "weld_reason": "architectural_support",
            }

        # Compute aggregate statistics (only for non-welded objects).
        num_stable = sum(
            1 for r in per_object_results.values()
            if r["stable"] and not r.get("welded", False)
        )
        num_unstable = sum(
            1 for r in per_object_results.values()
            if not r["stable"] and not r.get("welded", False)
        )
        num_non_welded = num_stable + num_unstable
        scene_stable = num_unstable == 0

        max_displacement = max(displacements) if displacements else 0.0
        mean_displacement = np.mean(displacements) if displacements else 0.0
        total_displacement = sum(displacements)

        max_rotation = max(rotations) if rotations else 0.0
        mean_rotation = np.mean(rotations) if rotations else 0.0

        method_name = self.decomposition_method.upper()
        result = MetricResult(
            message=(
                f"Architectural welded equilibrium ({method_name}): scene_stable={scene_stable}, "
                f"{num_stable}/{num_non_welded} stable (non-welded) objects, "
                f"{len(objects_to_weld)} welded (architectural), "
                f"mean_displacement={mean_displacement:.4f}m, "
                f"max_displacement={max_displacement:.4f}m"
            ),
            data={
                "scene_stable": scene_stable,
                "num_stable_objects": num_stable,
                "num_unstable_objects": num_unstable,
                "max_displacement": float(max_displacement),
                "mean_displacement": float(mean_displacement),
                "max_rotation": float(max_rotation),
                "mean_rotation": float(mean_rotation),
                "total_displacement": float(total_displacement),
                "per_object_results": per_object_results,
                "welded_objects": objects_to_weld,
                "num_welded_objects": len(objects_to_weld),
                # Decomposition method used.
                "decomposition_method": self.decomposition_method,
            },
        )

        if verbose:
            print(f"\n{result.message}\n")
            print("Non-welded objects:")
            for obj_id, obj_result in per_object_results.items():
                if not obj_result.get("welded", False):
                    status = "STABLE" if obj_result["stable"] else "UNSTABLE"
                    print(
                        f"  {obj_id}: {status} "
                        f"(displacement={obj_result['displacement']:.4f}m, "
                        f"rotation={obj_result['rotation']:.4f}rad)"
                    )
            if objects_to_weld:
                print(f"\nWelded objects (architectural support): {objects_to_weld}")

        return result


@register_non_vlm_metric(config_class=ArchitecturalWeldedEquilibriumMetricConfigCoACD)
class ArchitecturalWeldedEquilibriumMetricCoACD(ArchitecturalWeldedEquilibriumMetricBase):
    """Architectural welded equilibrium metric using CoACD convex decomposition.

    CoACD (Convex Approximate Convex Decomposition) is used for generating
    collision geometry for Drake physics simulation.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "coacd"
    drake_scene_folder: str = "architectural_welded_equilibrium_coacd"


@register_non_vlm_metric(config_class=ArchitecturalWeldedEquilibriumMetricConfigVHACD)
class ArchitecturalWeldedEquilibriumMetricVHACD(ArchitecturalWeldedEquilibriumMetricBase):
    """Architectural welded equilibrium metric using VHACD convex decomposition.

    VHACD (Volumetric Hierarchical Approximate Convex Decomposition) provides
    an alternative decomposition method for Drake physics simulation.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "vhacd"
    drake_scene_folder: str = "architectural_welded_equilibrium_vhacd"

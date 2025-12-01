"""Static equilibrium metrics using Drake physics simulation.

This module provides static equilibrium metrics using Drake physics simulation
with both CoACD and VHACD convex decomposition methods. Lower movement after
simulation = better static equilibrium = more physically plausible scene.
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
class StaticEquilibriumMetricConfigCoACD:
    """Configuration for the static equilibrium metric using CoACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        use_trimesh_inertia: If True, compute mass/inertia from mesh volume.
            If False, use default mass of 1kg.
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        coacd_threshold: CoACD decomposition threshold.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        weld_wall_ceiling_objects: If True, weld wall and ceiling mounted objects
            to world before simulation (requires obj_support_type_result.json).
        save_simulation_html: If True, save meshcat visualization to HTML file.
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    use_trimesh_inertia: bool = False
    density: float = 1000.0
    coacd_threshold: float = 0.05
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    weld_wall_ceiling_objects: bool = True
    save_simulation_html: bool = True
    hydroelastic_modulus: float | None = None


@dataclass
class StaticEquilibriumMetricConfigVHACD:
    """Configuration for the static equilibrium metric using VHACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        use_trimesh_inertia: If True, compute mass/inertia from mesh volume.
            If False, use default mass of 1kg.
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        weld_wall_ceiling_objects: If True, weld wall and ceiling mounted objects
            to world before simulation (requires obj_support_type_result.json).
        save_simulation_html: If True, save meshcat visualization to HTML file.
        hydroelastic_modulus: If set, adds compliant hydroelastic properties
            with this modulus (Pa). If None, no hydroelastic properties are added.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    use_trimesh_inertia: bool = False
    density: float = 1000.0
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    weld_wall_ceiling_objects: bool = True
    save_simulation_html: bool = True
    hydroelastic_modulus: float | None = None


class StaticEquilibriumMetricBase(BaseMetric):
    """Base class for static equilibrium metrics.

    Loads a scene into Drake, simulates it for a specified time, and measures
    how much each object moves. Objects that move less are considered more
    stable and the scene is more physically plausible.

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

    def _get_wall_ceiling_objects(self) -> list[str]:
        """Get object IDs for wall and ceiling mounted objects.

        Returns:
            List of object IDs that are mounted on walls or ceilings.
        """
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        wall_ceiling_objects = []
        for obj_id, support_type in support_types.items():
            if support_type in ("wall", "ceiling"):
                wall_ceiling_objects.append(obj_id)

        return wall_ceiling_objects

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

        # Get wall/ceiling objects to weld (if enabled).
        objects_to_weld = []
        if self.cfg.weld_wall_ceiling_objects:
            objects_to_weld = self._get_wall_ceiling_objects()
            if verbose and objects_to_weld:
                print(f"Welding wall/ceiling objects: {objects_to_weld}")

        # Get coacd_threshold if applicable.
        coacd_threshold = getattr(self.cfg, "coacd_threshold", 0.05)
        vhacd_max_convex_hulls = getattr(self.cfg, "vhacd_max_convex_hulls", 64)

        # Get hydroelastic_modulus if applicable.
        hydroelastic_modulus = getattr(self.cfg, "hydroelastic_modulus", None)

        # Create Drake plant with dynamics (time_step > 0).
        # Try with hydroelastic first, fall back to point contact if it fails.
        def create_and_run_simulation(use_hydroelastic_modulus, time_step):
            """Create plant and run simulation with given hydroelastic setting."""
            builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene(
                scene=self.scene,
                time_step=time_step,
                temp_dir=drake_scene_dir / "simulation",
                weld_to_world=objects_to_weld,
                coacd_threshold=coacd_threshold,
                vhacd_max_convex_hulls=vhacd_max_convex_hulls,
                decomposition_method=self.decomposition_method,
                hydroelastic_modulus=use_hydroelastic_modulus,
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

            return diagram, plant, initial_context, final_context, obj_id_to_model_name

        # Try with hydroelastic first.
        try:
            diagram, plant, initial_context, final_context, obj_id_to_model_name = (
                create_and_run_simulation(hydroelastic_modulus, self.cfg.time_step)
            )
        except RuntimeError as e:
            if "Cannot instantiate plane from normal" in str(e) and hydroelastic_modulus is not None:
                # Hydroelastic failed due to degenerate mesh geometry.
                # Fall back to point contact (no hydroelastic) with smaller timestep.
                point_contact_timestep = 1e-3
                print(
                    f"WARNING: Hydroelastic contact failed with degenerate mesh. "
                    f"Falling back to point contact with timestep={point_contact_timestep}."
                )
                diagram, plant, initial_context, final_context, obj_id_to_model_name = (
                    create_and_run_simulation(None, point_contact_timestep)
                )
            else:
                raise

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
                "weld_reason": "wall_ceiling_mounted",
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
                f"Static equilibrium ({method_name}): scene_stable={scene_stable}, "
                f"{num_stable}/{num_non_welded} stable (non-welded) objects, "
                f"{len(objects_to_weld)} welded (wall/ceiling), "
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
                print(f"\nWelded objects (wall/ceiling mounted): {objects_to_weld}")

        return result


@register_non_vlm_metric(config_class=StaticEquilibriumMetricConfigCoACD)
class StaticEquilibriumMetricCoACD(StaticEquilibriumMetricBase):
    """Static equilibrium metric using CoACD convex decomposition.

    CoACD (Convex Approximate Convex Decomposition) is used for generating
    collision geometry for Drake physics simulation.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "coacd"
    drake_scene_folder: str = "static_equilibrium_coacd"


@register_non_vlm_metric(config_class=StaticEquilibriumMetricConfigVHACD)
class StaticEquilibriumMetricVHACD(StaticEquilibriumMetricBase):
    """Static equilibrium metric using VHACD convex decomposition.

    VHACD (Volumetric Hierarchical Approximate Convex Decomposition) provides
    an alternative decomposition method for Drake physics simulation.
    """

    decomposition_method: Literal["coacd", "vhacd"] = "vhacd"
    drake_scene_folder: str = "static_equilibrium_vhacd"

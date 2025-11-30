"""Combined welded equilibrium metrics using Drake physics simulation.

This module provides equilibrium metrics that combine both penetration-based
welding (from WeldedEquilibriumMetric) and architecture-based welding (from
ArchitecturalWeldedEquilibriumMetric). Objects are welded if they:
1. Are supported by architectural elements (floor, wall, ceiling), OR
2. Are involved in penetrations with other objects
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
    detect_penetrating_pairs,
    measure_displacement,
    run_simulation,
)


@dataclass
class CombinedWeldedEquilibriumMetricConfigCoACD:
    """Configuration for the combined welded equilibrium metric using CoACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        use_trimesh_inertia: If True, compute mass/inertia from mesh volume.
            If False, use default mass of 1kg.
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        coacd_threshold: CoACD decomposition threshold.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        penetration_threshold: Penetration depth threshold for welding objects (meters).
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
    penetration_threshold: float = 0.001
    save_simulation_html: bool = True
    hydroelastic_modulus: float | None = None
    weld_floor_objects: bool = False


@dataclass
class CombinedWeldedEquilibriumMetricConfigVHACD:
    """Configuration for the combined welded equilibrium metric using VHACD.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        use_trimesh_inertia: If True, compute mass/inertia from mesh volume.
            If False, use default mass of 1kg.
        density: Density in kg/m³ (only used if use_trimesh_inertia=True).
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        penetration_threshold: Penetration depth threshold for welding objects (meters).
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
    penetration_threshold: float = 0.001
    save_simulation_html: bool = True
    hydroelastic_modulus: float | None = None
    weld_floor_objects: bool = False


class CombinedWeldedEquilibriumMetricBase(BaseMetric):
    """Base class for combined welded equilibrium metrics.

    This metric combines two welding strategies:
    1. Architectural welding: Objects supported by floor, wall, or ceiling
    2. Penetration welding: Objects involved in penetrations with other objects

    Objects that meet EITHER criterion are welded to world. This provides a
    realistic simulation where both large architectural furniture and objects
    with penetration issues are stabilized.

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
            MetricResult with equilibrium data including welded objects info.
        """
        # Use output directory for Drake files (persisted for debugging/inspection).
        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get coacd_threshold if applicable.
        coacd_threshold = getattr(self.cfg, "coacd_threshold", 0.05)

        # Get hydroelastic_modulus if applicable.
        hydroelastic_modulus = getattr(self.cfg, "hydroelastic_modulus", None)

        # Get architectural objects to weld.
        architectural_objects = self._get_architectural_objects()
        architectural_objects_set = set(architectural_objects)

        if verbose and architectural_objects:
            print(f"Architectural objects to weld: {architectural_objects}")

        # Phase 1: Create Drake plant to detect penetrations.
        # Note: hydroelastic not needed for static penetration detection.
        builder1, plant1, scene_graph1, obj_id_to_model_name = create_drake_plant_from_scene(
            scene=self.scene,
            time_step=0.0,  # Static query for penetration detection.
            temp_dir=drake_scene_dir / "penetration_detection",
            weld_to_world=[],
            coacd_threshold=coacd_threshold,
            decomposition_method=self.decomposition_method,
            hydroelastic_modulus=hydroelastic_modulus,
        )

        # Build diagram and detect penetrations.
        diagram1 = builder1.Build()
        context1 = diagram1.CreateDefaultContext()

        penetrating_pairs = detect_penetrating_pairs(
            plant=plant1,
            scene_graph=scene_graph1,
            context=context1,
            threshold=self.cfg.penetration_threshold,
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Collect all objects involved in ANY penetration.
        penetrating_objects = set()
        penetrating_pairs_info = []

        for obj_a, obj_b, depth in penetrating_pairs:
            # Skip floor_plan (already static).
            if obj_a != "floor_plan":
                penetrating_objects.add(obj_a)
            if obj_b != "floor_plan":
                penetrating_objects.add(obj_b)

            penetrating_pairs_info.append({
                "obj_a": obj_a,
                "obj_b": obj_b,
                "penetration_depth": depth,
            })

        if verbose:
            print(f"\nDetected {len(penetrating_pairs)} penetrating pairs")
            if penetrating_objects:
                print(f"Penetrating objects to weld: {penetrating_objects}")

        # Combine architectural and penetrating objects.
        all_objects_to_weld = architectural_objects_set | penetrating_objects
        objects_to_weld = list(all_objects_to_weld)

        # Track weld reasons for each object.
        weld_reasons = {}
        for obj_id in all_objects_to_weld:
            is_architectural = obj_id in architectural_objects_set
            is_penetrating = obj_id in penetrating_objects
            if is_architectural and is_penetrating:
                weld_reasons[obj_id] = "both"
            elif is_architectural:
                weld_reasons[obj_id] = "architectural_support"
            else:
                weld_reasons[obj_id] = "penetration"

        if verbose:
            print(f"Total objects to weld: {len(objects_to_weld)}")

        # Phase 2: Create new plant with all objects welded.
        # Try with hydroelastic first, fall back to point contact if it fails.
        def create_and_run_simulation(use_hydroelastic_modulus, time_step):
            """Create plant and run simulation with given hydroelastic setting."""
            builder2, plant2, scene_graph2, obj_id_to_model_name2 = create_drake_plant_from_scene(
                scene=self.scene,
                time_step=time_step,  # Dynamics for simulation.
                temp_dir=drake_scene_dir / "simulation",
                weld_to_world=objects_to_weld,
                coacd_threshold=coacd_threshold,
                decomposition_method=self.decomposition_method,
                hydroelastic_modulus=use_hydroelastic_modulus,
            )

            # Determine HTML output path if visualization is enabled.
            html_path = None
            if getattr(self.cfg, "save_simulation_html", False):
                html_path = drake_scene_dir / "simulation" / "simulation.html"

            # Run simulation.
            diagram2, initial_context, final_context = run_simulation(
                builder=builder2,
                plant=plant2,
                simulation_time=self.cfg.simulation_time,
                scene_graph=scene_graph2,
                output_html_path=html_path,
            )

            return diagram2, plant2, initial_context, final_context, obj_id_to_model_name2

        # Try with hydroelastic first.
        try:
            diagram2, plant2, initial_context, final_context, obj_id_to_model_name2 = (
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
                diagram2, plant2, initial_context, final_context, obj_id_to_model_name2 = (
                    create_and_run_simulation(None, point_contact_timestep)
                )
            else:
                raise

        # Get plant contexts.
        initial_plant_context = plant2.GetMyContextFromRoot(initial_context)
        final_plant_context = plant2.GetMyContextFromRoot(final_context)

        # Measure displacement for non-welded objects only.
        non_welded_obj_id_to_model_name = {
            obj_id: model_name
            for obj_id, model_name in obj_id_to_model_name2.items()
            if obj_id not in all_objects_to_weld
        }

        displacement_results = measure_displacement(
            plant=plant2,
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
                "weld_reason": weld_reasons[obj_id],
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
                f"Combined welded equilibrium ({method_name}): scene_stable={scene_stable}, "
                f"{num_stable}/{num_non_welded} stable (non-welded) objects, "
                f"{len(objects_to_weld)} welded ({len(architectural_objects)} architectural, "
                f"{len(penetrating_objects)} penetrating), "
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
                # All welded objects.
                "welded_objects": objects_to_weld,
                "num_welded_objects": len(objects_to_weld),
                # Breakdown by reason.
                "welded_objects_architectural": list(architectural_objects_set),
                "welded_objects_penetrating": list(penetrating_objects),
                "num_welded_architectural": len(architectural_objects_set),
                "num_welded_penetrating": len(penetrating_objects),
                # Penetration info.
                "penetrating_pairs": penetrating_pairs_info,
                "num_penetrating_pairs": len(penetrating_pairs_info),
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
                print(f"\nWelded objects:")
                for obj_id in objects_to_weld:
                    reason = weld_reasons[obj_id]
                    print(f"  {obj_id}: {reason}")

        return result


@register_non_vlm_metric(config_class=CombinedWeldedEquilibriumMetricConfigCoACD)
class CombinedWeldedEquilibriumMetricCoACD(CombinedWeldedEquilibriumMetricBase):
    """Combined welded equilibrium metric using CoACD convex decomposition.

    CoACD (Convex Approximate Convex Decomposition) is used for generating
    collision geometry for penetration detection and Drake physics simulation.

    This metric welds objects that are either:
    1. Supported by architectural elements (floor, wall, ceiling), OR
    2. Involved in penetrations with other objects
    """

    decomposition_method: Literal["coacd", "vhacd"] = "coacd"
    drake_scene_folder: str = "combined_welded_equilibrium_coacd"


@register_non_vlm_metric(config_class=CombinedWeldedEquilibriumMetricConfigVHACD)
class CombinedWeldedEquilibriumMetricVHACD(CombinedWeldedEquilibriumMetricBase):
    """Combined welded equilibrium metric using VHACD convex decomposition.

    VHACD (Volumetric Hierarchical Approximate Convex Decomposition) provides
    an alternative decomposition method for penetration detection and physics.

    This metric welds objects that are either:
    1. Supported by architectural elements (floor, wall, ceiling), OR
    2. Involved in penetrations with other objects
    """

    decomposition_method: Literal["coacd", "vhacd"] = "vhacd"
    drake_scene_folder: str = "combined_welded_equilibrium_vhacd"

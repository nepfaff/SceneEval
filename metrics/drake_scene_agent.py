"""Drake-based metrics for SceneAgent using pre-computed SDFs.

SceneAgent exports scenes with pre-computed CoACD collision geometry and
full inertia properties. These metrics use those SDFs directly instead of
regenerating collision geometry.

Note: Only CoACD metrics are supported for SceneAgent since scene-agent only
computes CoACD decomposition (not VHACD).
"""

import json
import pathlib
import statistics
from dataclasses import dataclass

import numpy as np

from scenes import Scene
from .base import BaseMetric, MetricResult
from .registry import register_non_vlm_metric
from .drake_utils import (
    create_drake_plant_from_scene_agent,
    detect_penetrating_pairs,
    measure_displacement,
    run_simulation,
)


def _get_scene_agent_paths(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Get scene JSON path and assets dir for SceneAgent from output directory.

    SceneAgent output follows this structure:
        output_eval/SceneAgent/scene_X/  <- output_dir
        input/SceneAgent/scene_X.json     <- scene_json_path
        input/SceneAgent/scene_X/assets/  <- assets_dir

    Args:
        output_dir: Output directory for the scene (e.g., output_eval/SceneAgent/scene_0)

    Returns:
        Tuple of (scene_json_path, assets_dir)
    """
    scene_name = output_dir.name  # e.g., "scene_0"
    method_name = output_dir.parent.name  # e.g., "SceneAgent"

    # Construct input paths
    input_base = pathlib.Path("input") / method_name
    scene_json_path = input_base / f"{scene_name}.json"
    assets_dir = input_base / scene_name / "assets"

    return scene_json_path, assets_dir


@dataclass
class DrakeCollisionMetricSceneAgentConfig:
    """Configuration for the Drake collision metric for SceneAgent.

    Attributes:
        penetration_threshold: Minimum penetration depth to report (meters).
    """

    penetration_threshold: float = 0.001


@register_non_vlm_metric(config_class=DrakeCollisionMetricSceneAgentConfig)
class DrakeCollisionMetricSceneAgent(BaseMetric):
    """Drake collision metric for SceneAgent using pre-computed CoACD SDFs.

    Uses scene-agent's pre-computed CoACD collision geometry instead of
    regenerating it. This is faster and more accurate since scene-agent
    uses the same collision geometry for physics simulation.
    """

    drake_scene_folder: str = "drake_collision_scene_agent"

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def run(self, verbose: bool = False) -> MetricResult:
        # Get SceneAgent-specific paths
        scene_json_path, assets_dir = _get_scene_agent_paths(self.output_dir)

        # Validate paths exist
        if not scene_json_path.exists():
            return MetricResult(
                message=f"SceneAgent scene JSON not found: {scene_json_path}",
                data={"error": f"Scene JSON not found: {scene_json_path}"},
            )
        if not assets_dir.exists():
            return MetricResult(
                message=f"SceneAgent assets dir not found: {assets_dir}",
                data={"error": f"Assets dir not found: {assets_dir}"},
            )

        # Use output directory for Drake files
        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Create Drake plant using pre-computed SDFs (time_step=0 for static query)
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=0.0,  # Static query, no simulation
            temp_dir=drake_scene_dir,
            weld_to_world=[],
        )

        # Build diagram and create context
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Detect penetrating pairs
        penetrating_pairs = detect_penetrating_pairs(
            plant=plant,
            scene_graph=scene_graph,
            context=context,
            threshold=self.cfg.penetration_threshold,
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Build collision results
        collision_results = {
            obj_id: {
                "in_collision": False,
                "colliding_with": [],
                "collision_details": [],
            }
            for obj_id in self.scene.get_obj_ids()
        }

        floor_collision_results = {
            obj_id: {
                "in_collision_with_floor": False,
                "penetration_depth": 0.0,
            }
            for obj_id in self.scene.get_obj_ids()
        }

        # Process penetrating pairs
        for obj_a, obj_b, depth in penetrating_pairs:
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

            if obj_a not in collision_results or obj_b not in collision_results:
                continue

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

        # Compute statistics
        num_obj_in_collision = sum(
            obj_result["in_collision"] for obj_result in collision_results.values()
        )
        scene_in_collision = num_obj_in_collision > 0

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

        result = MetricResult(
            message=(
                f"Drake collision (SceneAgent CoACD): scene_in_collision={scene_in_collision}, "
                f"{num_obj_in_collision}/{len(self.scene.get_obj_ids())} objects in collision, "
                f"{num_collision_pairs} collision pairs, "
                f"max_depth={max_penetration_depth:.4f}m; "
                f"floor_collisions={num_obj_in_floor_collision}"
            ),
            data={
                "scene_in_collision": scene_in_collision,
                "num_obj_in_collision": num_obj_in_collision,
                "num_collision_pairs": num_collision_pairs,
                "max_penetration_depth": max_penetration_depth,
                "min_penetration_depth": min_penetration_depth,
                "mean_penetration_depth": mean_penetration_depth,
                "median_penetration_depth": median_penetration_depth,
                "collision_results": collision_results,
                "num_obj_in_floor_collision": num_obj_in_floor_collision,
                "max_floor_penetration_depth": max_floor_penetration,
                "mean_floor_penetration_depth": mean_floor_penetration,
                "floor_collision_results": floor_collision_results,
                "decomposition_method": "coacd",
                "source": "scene_agent_precomputed",
            },
        )

        if verbose:
            print(f"\n{result.message}\n")

        return result


@dataclass
class StaticEquilibriumMetricSceneAgentConfig:
    """Configuration for the static equilibrium metric for SceneAgent.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        weld_wall_ceiling_objects: If True, weld wall and ceiling mounted objects.
        save_simulation_html: If True, save meshcat visualization to HTML file.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    weld_wall_ceiling_objects: bool = True
    save_simulation_html: bool = True


@register_non_vlm_metric(config_class=StaticEquilibriumMetricSceneAgentConfig)
class StaticEquilibriumMetricSceneAgent(BaseMetric):
    """Static equilibrium metric for SceneAgent using pre-computed CoACD SDFs.

    Uses scene-agent's pre-computed SDFs with full inertia tensors for
    accurate physics simulation.
    """

    drake_scene_folder: str = "static_equilibrium_scene_agent"

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def _get_wall_ceiling_objects(self) -> list[str]:
        """Get object IDs for wall and ceiling mounted objects."""
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        return [
            obj_id for obj_id, support_type in support_types.items()
            if support_type in ("wall", "ceiling")
        ]

    def run(self, verbose: bool = False) -> MetricResult:
        # Get SceneAgent-specific paths
        scene_json_path, assets_dir = _get_scene_agent_paths(self.output_dir)

        # Validate paths exist
        if not scene_json_path.exists():
            return MetricResult(
                message=f"SceneAgent scene JSON not found: {scene_json_path}",
                data={"error": f"Scene JSON not found: {scene_json_path}"},
            )
        if not assets_dir.exists():
            return MetricResult(
                message=f"SceneAgent assets dir not found: {assets_dir}",
                data={"error": f"Assets dir not found: {assets_dir}"},
            )

        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get wall/ceiling objects to weld
        objects_to_weld = []
        if self.cfg.weld_wall_ceiling_objects:
            objects_to_weld = self._get_wall_ceiling_objects()
            if verbose and objects_to_weld:
                print(f"Welding wall/ceiling objects: {objects_to_weld}")

        # Create Drake plant with dynamics
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=self.cfg.time_step,
            temp_dir=drake_scene_dir / "simulation",
            weld_to_world=objects_to_weld,
        )

        # Determine HTML output path if visualization is enabled
        html_path = None
        if getattr(self.cfg, "save_simulation_html", False):
            html_path = drake_scene_dir / "simulation" / "simulation.html"

        # Run simulation
        diagram, initial_context, final_context = run_simulation(
            builder=builder,
            plant=plant,
            simulation_time=self.cfg.simulation_time,
            scene_graph=scene_graph,
            output_html_path=html_path,
        )

        # Get plant contexts
        initial_plant_context = plant.GetMyContextFromRoot(initial_context)
        final_plant_context = plant.GetMyContextFromRoot(final_context)

        # Measure displacement for non-welded objects only
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

        # Compute per-object stability and aggregate statistics
        per_object_results = {}
        displacements = []
        rotations = []

        for obj_id, data in displacement_results.items():
            displacement = data["displacement"]
            rotation = data["rotation"]

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

            if not np.isnan(displacement):
                displacements.append(displacement)
            if not np.isnan(rotation):
                rotations.append(rotation)

        # Add welded objects
        for obj_id in objects_to_weld:
            per_object_results[obj_id] = {
                "stable": True,
                "displacement": 0.0,
                "rotation": 0.0,
                "initial_position": None,
                "final_position": None,
                "welded": True,
                "weld_reason": "wall_ceiling_mounted",
            }

        # Compute aggregate statistics
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

        result = MetricResult(
            message=(
                f"Static equilibrium (SceneAgent CoACD): scene_stable={scene_stable}, "
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
                "decomposition_method": "coacd",
                "source": "scene_agent_precomputed",
            },
        )

        if verbose:
            print(f"\n{result.message}\n")

        return result


@dataclass
class WeldedEquilibriumMetricSceneAgentConfig:
    """Configuration for the welded equilibrium metric for SceneAgent.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        penetration_threshold: Minimum penetration depth to weld objects (meters).
        weld_wall_ceiling_objects: If True, weld wall and ceiling mounted objects.
        save_simulation_html: If True, save meshcat visualization to HTML file.
    """

    simulation_time: float = 2.0
    time_step: float = 0.01
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    penetration_threshold: float = 0.001
    weld_wall_ceiling_objects: bool = True
    save_simulation_html: bool = True


@register_non_vlm_metric(config_class=WeldedEquilibriumMetricSceneAgentConfig)
class WeldedEquilibriumMetricSceneAgent(BaseMetric):
    """Welded equilibrium metric for SceneAgent using pre-computed CoACD SDFs.

    First detects penetrating objects and welds them to world, then runs
    physics simulation. This isolates stability issues from penetration-induced
    movement.
    """

    drake_scene_folder: str = "welded_equilibrium_scene_agent"

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def _get_wall_ceiling_objects(self) -> list[str]:
        """Get object IDs for wall and ceiling mounted objects."""
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        return [
            obj_id for obj_id, support_type in support_types.items()
            if support_type in ("wall", "ceiling")
        ]

    def run(self, verbose: bool = False) -> MetricResult:
        # Get SceneAgent-specific paths
        scene_json_path, assets_dir = _get_scene_agent_paths(self.output_dir)

        # Validate paths exist
        if not scene_json_path.exists():
            return MetricResult(
                message=f"SceneAgent scene JSON not found: {scene_json_path}",
                data={"error": f"Scene JSON not found: {scene_json_path}"},
            )
        if not assets_dir.exists():
            return MetricResult(
                message=f"SceneAgent assets dir not found: {assets_dir}",
                data={"error": f"Assets dir not found: {assets_dir}"},
            )

        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Detect penetrating objects
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=0.0,  # Static query
            temp_dir=drake_scene_dir / "penetration_detection",
            weld_to_world=[],
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        penetrating_pairs = detect_penetrating_pairs(
            plant=plant,
            scene_graph=scene_graph,
            context=context,
            threshold=self.cfg.penetration_threshold,
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Collect objects to weld (penetrating objects)
        penetrating_objects = set()
        for obj_a, obj_b, _ in penetrating_pairs:
            if obj_a != "floor_plan":
                penetrating_objects.add(obj_a)
            if obj_b != "floor_plan":
                penetrating_objects.add(obj_b)

        # Add wall/ceiling objects to weld list
        wall_ceiling_objects = []
        if self.cfg.weld_wall_ceiling_objects:
            wall_ceiling_objects = self._get_wall_ceiling_objects()

        objects_to_weld = list(set(penetrating_objects) | set(wall_ceiling_objects))

        if verbose:
            print(f"Welding {len(penetrating_objects)} penetrating objects: {list(penetrating_objects)}")
            if wall_ceiling_objects:
                print(f"Welding {len(wall_ceiling_objects)} wall/ceiling objects: {wall_ceiling_objects}")

        # Step 2: Create new plant with welded objects and simulate
        builder2, plant2, scene_graph2, obj_id_to_model_name2 = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=self.cfg.time_step,
            temp_dir=drake_scene_dir / "simulation",
            weld_to_world=objects_to_weld,
        )

        html_path = None
        if getattr(self.cfg, "save_simulation_html", False):
            html_path = drake_scene_dir / "simulation" / "simulation.html"

        diagram2, initial_context2, final_context2 = run_simulation(
            builder=builder2,
            plant=plant2,
            simulation_time=self.cfg.simulation_time,
            scene_graph=scene_graph2,
            output_html_path=html_path,
        )

        # Get plant contexts
        initial_plant_context = plant2.GetMyContextFromRoot(initial_context2)
        final_plant_context = plant2.GetMyContextFromRoot(final_context2)

        # Measure displacement for non-welded objects
        non_welded_obj_id_to_model_name = {
            obj_id: model_name
            for obj_id, model_name in obj_id_to_model_name2.items()
            if obj_id not in objects_to_weld
        }

        displacement_results = measure_displacement(
            plant=plant2,
            initial_context=initial_plant_context,
            final_context=final_plant_context,
            obj_id_to_model_name=non_welded_obj_id_to_model_name,
        )

        # Compute per-object stability
        per_object_results = {}
        displacements = []
        rotations = []

        for obj_id, data in displacement_results.items():
            displacement = data["displacement"]
            rotation = data["rotation"]

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

            if not np.isnan(displacement):
                displacements.append(displacement)
            if not np.isnan(rotation):
                rotations.append(rotation)

        # Add welded objects
        for obj_id in objects_to_weld:
            weld_reason = []
            if obj_id in penetrating_objects:
                weld_reason.append("penetrating")
            if obj_id in wall_ceiling_objects:
                weld_reason.append("wall_ceiling_mounted")

            per_object_results[obj_id] = {
                "stable": True,
                "displacement": 0.0,
                "rotation": 0.0,
                "initial_position": None,
                "final_position": None,
                "welded": True,
                "weld_reason": "_".join(weld_reason) if weld_reason else "unknown",
            }

        # Compute aggregate statistics
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

        result = MetricResult(
            message=(
                f"Welded equilibrium (SceneAgent CoACD): scene_stable={scene_stable}, "
                f"{num_stable}/{num_non_welded} stable (non-welded) objects, "
                f"{len(objects_to_weld)} welded ({len(penetrating_objects)} penetrating, "
                f"{len(wall_ceiling_objects)} wall/ceiling), "
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
                "num_penetrating_objects": len(penetrating_objects),
                "penetrating_objects": list(penetrating_objects),
                "wall_ceiling_objects": wall_ceiling_objects,
                "decomposition_method": "coacd",
                "source": "scene_agent_precomputed",
            },
        )

        if verbose:
            print(f"\n{result.message}\n")

        return result


@dataclass
class ArchitecturalWeldedEquilibriumMetricSceneAgentConfig:
    """Configuration for the architectural welded equilibrium metric for SceneAgent.

    This metric welds objects supported by architectural elements (floor, wall,
    ceiling) to world, while letting objects supported by other objects move freely.

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        save_simulation_html: If True, save meshcat visualization to HTML file.
    """

    simulation_time: float = 5.0
    time_step: float = 0.001
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    save_simulation_html: bool = True


@register_non_vlm_metric(config_class=ArchitecturalWeldedEquilibriumMetricSceneAgentConfig)
class ArchitecturalWeldedEquilibriumMetricSceneAgent(BaseMetric):
    """Architectural welded equilibrium metric for SceneAgent.

    Welds objects supported by architectural elements (floor, wall, ceiling) to
    world, while letting objects supported by other objects move freely. Uses
    SceneAgent's pre-computed CoACD SDFs with full inertia tensors.

    Unlike WeldedEquilibriumMetricSceneAgent which welds penetrating objects,
    this metric welds based on support type from obj_support_type_result.json.
    """

    drake_scene_folder: str = "architectural_welded_equilibrium_scene_agent"

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def _get_architectural_objects(self) -> list[str]:
        """Get object IDs for objects supported by architectural elements.

        Returns:
            List of object IDs that are supported by floor, wall, or ceiling.
        """
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        return [
            obj_id for obj_id, support_type in support_types.items()
            if support_type in ("ground", "wall", "ceiling")
        ]

    def run(self, verbose: bool = False) -> MetricResult:
        # Get SceneAgent-specific paths
        scene_json_path, assets_dir = _get_scene_agent_paths(self.output_dir)

        # Validate paths exist
        if not scene_json_path.exists():
            return MetricResult(
                message=f"SceneAgent scene JSON not found: {scene_json_path}",
                data={"error": f"Scene JSON not found: {scene_json_path}"},
            )
        if not assets_dir.exists():
            return MetricResult(
                message=f"SceneAgent assets dir not found: {assets_dir}",
                data={"error": f"Assets dir not found: {assets_dir}"},
            )

        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get architectural objects to weld
        objects_to_weld = self._get_architectural_objects()
        if verbose and objects_to_weld:
            print(f"Welding architectural objects: {objects_to_weld}")

        # Create Drake plant with dynamics (single plant, no penetration detection)
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=self.cfg.time_step,
            temp_dir=drake_scene_dir / "simulation",
            weld_to_world=objects_to_weld,
        )

        # Determine HTML output path if visualization is enabled
        html_path = None
        if getattr(self.cfg, "save_simulation_html", False):
            html_path = drake_scene_dir / "simulation" / "simulation.html"

        # Run simulation
        diagram, initial_context, final_context = run_simulation(
            builder=builder,
            plant=plant,
            simulation_time=self.cfg.simulation_time,
            scene_graph=scene_graph,
            output_html_path=html_path,
        )

        # Get plant contexts
        initial_plant_context = plant.GetMyContextFromRoot(initial_context)
        final_plant_context = plant.GetMyContextFromRoot(final_context)

        # Measure displacement for non-welded objects only
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

        # Compute per-object stability and aggregate statistics
        per_object_results = {}
        displacements = []
        rotations = []

        # Process non-welded objects
        for obj_id, data in displacement_results.items():
            displacement = data["displacement"]
            rotation = data["rotation"]

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

            if not np.isnan(displacement):
                displacements.append(displacement)
            if not np.isnan(rotation):
                rotations.append(rotation)

        # Add welded objects (they have zero displacement by definition)
        for obj_id in objects_to_weld:
            per_object_results[obj_id] = {
                "stable": True,
                "displacement": 0.0,
                "rotation": 0.0,
                "initial_position": None,
                "final_position": None,
                "welded": True,
                "weld_reason": "architectural_support",
            }

        # Compute aggregate statistics (only for non-welded objects)
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

        result = MetricResult(
            message=(
                f"Architectural welded equilibrium (SceneAgent CoACD): scene_stable={scene_stable}, "
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
                "decomposition_method": "coacd",
                "source": "scene_agent_precomputed",
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


@dataclass
class CombinedWeldedEquilibriumMetricSceneAgentConfig:
    """Configuration for the combined welded equilibrium metric for SceneAgent.

    This metric combines both welding strategies:
    1. Architectural welding: Objects supported by floor, wall, or ceiling
    2. Penetration welding: Objects involved in penetrations with other objects

    Attributes:
        simulation_time: Time to simulate in seconds.
        time_step: Drake simulation time step.
        displacement_threshold: Maximum displacement for an object to be "stable" (meters).
        rotation_threshold: Maximum rotation for an object to be "stable" (radians).
        penetration_threshold: Minimum penetration depth to weld objects (meters).
        save_simulation_html: If True, save meshcat visualization to HTML file.
    """

    simulation_time: float = 5.0
    time_step: float = 0.001
    displacement_threshold: float = 0.01
    rotation_threshold: float = 0.1
    penetration_threshold: float = 0.001
    save_simulation_html: bool = True


@register_non_vlm_metric(config_class=CombinedWeldedEquilibriumMetricSceneAgentConfig)
class CombinedWeldedEquilibriumMetricSceneAgent(BaseMetric):
    """Combined welded equilibrium metric for SceneAgent.

    Combines both welding strategies:
    1. Architectural welding: Objects supported by floor, wall, or ceiling
    2. Penetration welding: Objects involved in penetrations with other objects

    Uses SceneAgent's pre-computed CoACD SDFs with full inertia tensors.
    Objects that meet EITHER criterion are welded to world.
    """

    drake_scene_folder: str = "combined_welded_equilibrium_scene_agent"

    def __init__(self, scene: Scene, output_dir: pathlib.Path, cfg, **kwargs) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.cfg = cfg

    def _get_architectural_objects(self) -> list[str]:
        """Get object IDs for objects supported by architectural elements.

        Returns:
            List of object IDs that are supported by floor, wall, or ceiling.
        """
        support_type_file = self.scene.output_dir / "obj_support_type_result.json"
        if not support_type_file.exists():
            return []

        with open(support_type_file) as f:
            support_types = json.load(f)

        return [
            obj_id for obj_id, support_type in support_types.items()
            if support_type in ("ground", "wall", "ceiling")
        ]

    def run(self, verbose: bool = False) -> MetricResult:
        # Get SceneAgent-specific paths
        scene_json_path, assets_dir = _get_scene_agent_paths(self.output_dir)

        # Validate paths exist
        if not scene_json_path.exists():
            return MetricResult(
                message=f"SceneAgent scene JSON not found: {scene_json_path}",
                data={"error": f"Scene JSON not found: {scene_json_path}"},
            )
        if not assets_dir.exists():
            return MetricResult(
                message=f"SceneAgent assets dir not found: {assets_dir}",
                data={"error": f"Assets dir not found: {assets_dir}"},
            )

        drake_scene_dir = self.output_dir / self.drake_scene_folder
        drake_scene_dir.mkdir(parents=True, exist_ok=True)

        # Get architectural objects to weld
        architectural_objects = self._get_architectural_objects()
        architectural_objects_set = set(architectural_objects)

        if verbose and architectural_objects:
            print(f"Architectural objects to weld: {architectural_objects}")

        # Step 1: Detect penetrating objects
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=0.0,  # Static query
            temp_dir=drake_scene_dir / "penetration_detection",
            weld_to_world=[],
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        penetrating_pairs = detect_penetrating_pairs(
            plant=plant,
            scene_graph=scene_graph,
            context=context,
            threshold=self.cfg.penetration_threshold,
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Collect objects involved in penetrations
        penetrating_objects = set()
        penetrating_pairs_info = []

        for obj_a, obj_b, depth in penetrating_pairs:
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
            print(f"Detected {len(penetrating_pairs)} penetrating pairs")
            if penetrating_objects:
                print(f"Penetrating objects to weld: {penetrating_objects}")

        # Combine architectural and penetrating objects
        all_objects_to_weld = architectural_objects_set | penetrating_objects
        objects_to_weld = list(all_objects_to_weld)

        # Track weld reasons for each object
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

        # Step 2: Create new plant with all welded objects and simulate
        builder2, plant2, scene_graph2, obj_id_to_model_name2 = create_drake_plant_from_scene_agent(
            scene=self.scene,
            scene_json_path=scene_json_path,
            assets_dir=assets_dir,
            time_step=self.cfg.time_step,
            temp_dir=drake_scene_dir / "simulation",
            weld_to_world=objects_to_weld,
        )

        html_path = None
        if getattr(self.cfg, "save_simulation_html", False):
            html_path = drake_scene_dir / "simulation" / "simulation.html"

        diagram2, initial_context2, final_context2 = run_simulation(
            builder=builder2,
            plant=plant2,
            simulation_time=self.cfg.simulation_time,
            scene_graph=scene_graph2,
            output_html_path=html_path,
        )

        # Get plant contexts
        initial_plant_context = plant2.GetMyContextFromRoot(initial_context2)
        final_plant_context = plant2.GetMyContextFromRoot(final_context2)

        # Measure displacement for non-welded objects
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

        # Compute per-object stability
        per_object_results = {}
        displacements = []
        rotations = []

        for obj_id, data in displacement_results.items():
            displacement = data["displacement"]
            rotation = data["rotation"]

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

            if not np.isnan(displacement):
                displacements.append(displacement)
            if not np.isnan(rotation):
                rotations.append(rotation)

        # Add welded objects
        for obj_id in objects_to_weld:
            per_object_results[obj_id] = {
                "stable": True,
                "displacement": 0.0,
                "rotation": 0.0,
                "initial_position": None,
                "final_position": None,
                "welded": True,
                "weld_reason": weld_reasons[obj_id],
            }

        # Compute aggregate statistics
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

        result = MetricResult(
            message=(
                f"Combined welded equilibrium (SceneAgent CoACD): scene_stable={scene_stable}, "
                f"{num_stable}/{num_non_welded} stable (non-welded) objects, "
                f"{len(objects_to_weld)} welded ({len(architectural_objects_set)} architectural, "
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
                # All welded objects
                "welded_objects": objects_to_weld,
                "num_welded_objects": len(objects_to_weld),
                # Breakdown by reason
                "welded_objects_architectural": list(architectural_objects_set),
                "welded_objects_penetrating": list(penetrating_objects),
                "num_welded_architectural": len(architectural_objects_set),
                "num_welded_penetrating": len(penetrating_objects),
                # Penetration info
                "penetrating_pairs": penetrating_pairs_info,
                "num_penetrating_pairs": len(penetrating_pairs_info),
                "decomposition_method": "coacd",
                "source": "scene_agent_precomputed",
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

import os
import random
import pathlib
import shutil
import hydra
import json
import numpy as np
from natsort import natsorted
from dataclasses import dataclass
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict

from omegaconf import OmegaConf

from scenes import *
from metrics import MetricRegistry, ObjMatching, ObjMatchingResults
from assets import Retriever
from semantic_colors import apply_semantic_colors
from vlm import VLMRegistry, BaseVLM
from parallel.worker import FileLoggingContext

load_dotenv()

# ========================================================================================

@dataclass
class EvaluationConfig:
    metrics: list[str]
    output_dir: str
    save_blend_file: bool
    save_trimesh_glb: bool
    vlm: str
    use_existing_matching: bool
    use_empty_matching_result: bool
    support_metric_use_existing_support_type_assessment: bool
    no_eval: bool
    verbose: bool

@dataclass
class InputConfig:
    root_dir: str
    scene_methods: list[str]
    method_use_simple_architecture: list[str]
    scene_mode: str
    scene_range: list[int]
    scene_list: list[int]
    annotation_file: str

@dataclass
class RenderConfig:
    normal_render_tasks: list[str] | None = None
    semantic_render_tasks: list[str] | None = None
    semantic_color_reference: str | None = None

@dataclass
class EvaluationPlan:
    evaluation_cfg: EvaluationConfig
    input_cfg: InputConfig
    render_cfg: RenderConfig
    
    def __post_init__(self):
        self.evaluation_cfg = EvaluationConfig(**self.evaluation_cfg)
        self.input_cfg = InputConfig(**self.input_cfg)
        self.render_cfg = RenderConfig(**self.render_cfg)

# ========================================================================================

def _fetch_scene_state_files(model_cfg: DictConfig, evaluation_plan: EvaluationPlan) -> dict[str, list[pathlib.Path]]:
    """
    Fetch scene state files for the given methods based on the evaluation plan.
    
    Args:
        model_cfg: the model configuration containing asset dataset information.
        evaluation_plan: the evaluation plan containing input configurations.
        
    Returns:
        scenes_per_method: a dictionary mapping method names to lists of scene state files.
    """
    
    scenes_per_method = {}
    
    for method in evaluation_plan.input_cfg.scene_methods:
        method_dir = pathlib.Path(evaluation_plan.input_cfg.root_dir) / model_cfg[method].input_dir_name
        scene_files = natsorted(list(method_dir.expanduser().resolve().glob("*.json")))
        id_to_file = {int(scene_file.stem.split("_")[-1]): scene_file for scene_file in scene_files}
        
        match evaluation_plan.input_cfg.scene_mode:
            case "all":
                scene_files = scene_files
            case "range":
                scene_range = evaluation_plan.input_cfg.scene_range
                scene_files = [id_to_file[scene_id] for scene_id in range(scene_range[0], scene_range[1]) if scene_id in id_to_file]
            case "list":
                scene_list = evaluation_plan.input_cfg.scene_list
                scene_files = [id_to_file[scene_id] for scene_id in scene_list if scene_id in id_to_file]
                
        scenes_per_method[method] = scene_files
    
    return scenes_per_method
    
def _render_scene(scene: Scene, render_tasks: list[str]) -> None:
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


def _copy_and_render_original_sceneweaver_blend(scene: Scene, method_scene_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """
    Copy the original SceneWeaver blend file to output and render from it.

    The GLB exports have texture baking issues (procedural materials use 3D coordinates
    that don't bake well to UV-mapped images). The original blend has perfect materials.

    Args:
        scene: The scene object (used for render settings)
        method_scene_file: Path to the scene JSON file
        output_dir: Output directory for this scene
    """
    import bpy
    import math

    # Find the original blend file in the input assets directory
    scene_dir = method_scene_file.parent / method_scene_file.stem / "assets"
    original_blend = scene_dir / "original_sceneweaver.blend"

    if not original_blend.exists():
        print(f"  Original SceneWeaver blend not found at: {original_blend}")
        return

    # Copy to output directory
    output_blend = output_dir / "original_sceneweaver.blend"
    shutil.copy2(original_blend, output_blend)
    print(f"  Copied original SceneWeaver blend to: {output_blend}")

    # Render from original blend file
    print(f"  Rendering from original SceneWeaver blend...")

    # Save current Blender state
    current_blend = bpy.data.filepath

    # Open the original blend file
    bpy.ops.wm.open_mainfile(filepath=str(original_blend))

    # Set up render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.film_transparent = False

    # Hide room enclosure meshes and placeholder/bounding box objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Hide newroom meshes except floor (they block light from above)
            if 'newroom' in obj.name and 'floor' not in obj.name:
                obj.hide_render = True
                obj.hide_viewport = True
            # Hide placeholder and bounding box objects (white boxes)
            if 'placeholder' in obj.name or 'bbox_placeholder' in obj.name:
                obj.hide_render = True
                obj.hide_viewport = True

    # Find scene bounds (only from visible spawn_asset objects)
    import mathutils as mu
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.visible_get() and 'spawn_asset' in obj.name:
            for v in obj.bound_box:
                world_v = obj.matrix_world @ mu.Vector(v)
                min_x = min(min_x, world_v.x)
                max_x = max(max_x, world_v.x)
                min_y = min(min_y, world_v.y)
                max_y = max(max_y, world_v.y)
                min_z = min(min_z, world_v.z)
                max_z = max(max_z, world_v.z)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scene_size = max(max_x - min_x, max_y - min_y)

    # Create a new top-down camera (don't use existing ones as they may have different settings)
    cam_data = bpy.data.cameras.new("TopDownCamera")
    cam = bpy.data.objects.new("TopDownCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    # Position camera above scene center looking down
    cam_height = max_z + scene_size * 1.5
    cam.location = (center_x, center_y, cam_height)
    cam.rotation_euler = (0, 0, 0)  # Looking straight down
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scene_size * 1.3
    bpy.context.scene.camera = cam

    # Add balanced lighting for good color reproduction
    # 1. Sun light from above and slightly angled
    sun_data = bpy.data.lights.new("RenderSun", 'SUN')
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("RenderSun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (center_x, center_y, cam_height + 5)
    sun.rotation_euler = (math.radians(45), math.radians(20), 0)

    # 2. Area light for soft fill
    area_data = bpy.data.lights.new("RenderArea", 'AREA')
    area_data.energy = 200.0
    area_data.size = scene_size
    area = bpy.data.objects.new("RenderArea", area_data)
    bpy.context.scene.collection.objects.link(area)
    area.location = (center_x, center_y, cam_height - 1)
    area.rotation_euler = (math.radians(180), 0, 0)  # Pointing down

    # Set up world background with ambient light
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1.0)
        bg_node.inputs['Strength'].default_value = 0.3

    # Render
    output_image = output_dir / "original_sceneweaver_render.png"
    bpy.context.scene.render.filepath = str(output_image)
    bpy.ops.render.render(write_still=True)
    print(f"  Saved original SceneWeaver render to: {output_image}")

    # Reload the previous blend file
    if current_blend:
        bpy.ops.wm.open_mainfile(filepath=current_blend)
    else:
        bpy.ops.wm.read_homefile()


def _copy_and_render_original_scene_agent_blend(scene: Scene, method_scene_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """
    Copy the original Scene Agent blend file to output and render from it.

    Scene Agent generates high-quality .blend files with proper materials.
    Similar to SceneWeaver handling.

    Args:
        scene: The scene object (used for render settings)
        method_scene_file: Path to the scene JSON file
        output_dir: Output directory for this scene
    """
    import bpy
    import math

    # Find the original blend file in the input assets directory
    scene_dir = method_scene_file.parent / method_scene_file.stem / "assets"
    original_blend = scene_dir / "original_scene_agent.blend"

    if not original_blend.exists():
        print(f"  Original Scene Agent blend not found at: {original_blend}")
        return

    # Copy to output directory
    output_blend = output_dir / "original_scene_agent.blend"
    shutil.copy2(original_blend, output_blend)
    print(f"  Copied original Scene Agent blend to: {output_blend}")

    # Render from original blend file
    print(f"  Rendering from original Scene Agent blend...")

    # Save current Blender state
    current_blend = bpy.data.filepath

    # Open the original blend file
    bpy.ops.wm.open_mainfile(filepath=str(original_blend))

    # Set up render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.film_transparent = False

    # Find scene bounds from all mesh objects
    import mathutils as mu
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.visible_get():
            # Skip architecture (floor, walls)
            if 'floor' in obj.name.lower() or 'wall' in obj.name.lower():
                continue
            for v in obj.bound_box:
                world_v = obj.matrix_world @ mu.Vector(v)
                min_x = min(min_x, world_v.x)
                max_x = max(max_x, world_v.x)
                min_y = min(min_y, world_v.y)
                max_y = max(max_y, world_v.y)
                min_z = min(min_z, world_v.z)
                max_z = max(max_z, world_v.z)

    # Handle empty scene case
    if min_x == float('inf'):
        min_x, max_x = -3, 3
        min_y, max_y = -3, 3
        min_z, max_z = 0, 2

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scene_size = max(max_x - min_x, max_y - min_y)

    # Create a new top-down camera
    cam_data = bpy.data.cameras.new("TopDownCamera")
    cam = bpy.data.objects.new("TopDownCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    # Position camera above scene center looking down
    cam_height = max_z + scene_size * 1.5
    cam.location = (center_x, center_y, cam_height)
    cam.rotation_euler = (0, 0, 0)  # Looking straight down
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scene_size * 1.3
    bpy.context.scene.camera = cam

    # Add balanced lighting for good color reproduction
    # 1. Sun light from above and slightly angled
    sun_data = bpy.data.lights.new("RenderSun", 'SUN')
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("RenderSun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (center_x, center_y, cam_height + 5)
    sun.rotation_euler = (math.radians(45), math.radians(20), 0)

    # 2. Area light for soft fill
    area_data = bpy.data.lights.new("RenderArea", 'AREA')
    area_data.energy = 200.0
    area_data.size = scene_size
    area = bpy.data.objects.new("RenderArea", area_data)
    bpy.context.scene.collection.objects.link(area)
    area.location = (center_x, center_y, cam_height - 1)
    area.rotation_euler = (math.radians(180), 0, 0)  # Pointing down

    # Set up world background with ambient light
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1.0)
        bg_node.inputs['Strength'].default_value = 0.3

    # Render
    output_image = output_dir / "original_scene_agent_render.png"
    bpy.context.scene.render.filepath = str(output_image)
    bpy.ops.render.render(write_still=True)
    print(f"  Saved original Scene Agent render to: {output_image}")

    # Reload the previous blend file
    if current_blend:
        bpy.ops.wm.open_mainfile(filepath=current_blend)
    else:
        bpy.ops.wm.read_homefile()


def _get_obj_matching(scene: Scene,
                      annotation: Annotation,
                      vlm: BaseVLM,
                      use_existing_matching: bool) -> ObjMatchingResults:
    """
    Get object matching results for the scene and annotation using a VLM.
    
    Args:
        scene: the scene to evaluate
        annotation: the annotation for the scene
        vlm: the VLM to use for object matching
        use_existing_matching: whether to use existing matching results if available
        
    Returns:
        matching_result: the object matching results
    """
    
    # Match object descriptions to target categories in annotation - used by other metrics
    matching_result_file = scene.output_dir / f"obj_matching_result.json"
    
    if use_existing_matching and matching_result_file.exists():
        print("\nUsing existing object matching result...")
        with open(matching_result_file, "r") as f:
            matching_result = ObjMatchingResults.from_dict(json.load(f))
        print("Existing object matching loaded.\n")
    else:
        print("\nCreating new object matching...")
        obj_matching = ObjMatching(scene, annotation, vlm)
        obj_matching_result = obj_matching.run()
        matching_result: ObjMatchingResults = obj_matching_result.data["matching_result"]
        
        # Save the matching result
        with open(matching_result_file, "w") as f:
            json.dump(matching_result.to_dict(), f, indent=4)
    
        print(f"New object matching done. Saved to: {matching_result_file}\n")
        
    return matching_result

# ========================================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seeds
    if cfg.seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        print(f"\nRandom seed set to: {cfg.seed}\n")
    else:
        print("\nNo random seed set.\n")

    # ----------------------------------------------------------------------------------------

    # Load evaluation plan
    evaluation_plan = EvaluationPlan(**cfg.evaluation_plan)
    
    if evaluation_plan.render_cfg.semantic_render_tasks and not evaluation_plan.evaluation_cfg.no_eval:
        input((
            "\nNote:\n"
            "Semantic rendering tasks are set, but evaluation is not skipped.\n"
            "Future renderings from metrics will all be in semantic colors.\n"
            "Press Enter to continue or Ctrl+C to abort.\n"
        ))
    
    # Fetch all scene state files that are to be evaluated
    scenes_per_method = _fetch_scene_state_files(cfg.models, evaluation_plan)
    print(f"\nEvaluating scenes for methods: {list(scenes_per_method.keys())}")
    for method, scene_files in scenes_per_method.items():
        print(f"{method} scenes:")
        [print(f" - {scene_file}") for scene_file in scene_files]
    print()

    # Load annotations
    annotations = Annotations(evaluation_plan.input_cfg.annotation_file)

    # Print the metrics to run
    metrics_to_run = list(evaluation_plan.evaluation_cfg.metrics) if evaluation_plan.evaluation_cfg.metrics else []
    print("\nRunning metrics:")
    [print(f" - {metric}") for metric in metrics_to_run]
    print()
    
    # Load metric configurations
    with open_dict(cfg):
        cfg.metrics.SupportMetric.use_existing_support_type_assessment = evaluation_plan.evaluation_cfg.support_metric_use_existing_support_type_assessment
    metric_configs = MetricRegistry.load_all_configs(cfg.metrics, metrics_to_run)
    
    # Load a mesh retriever for the asset datasets used by the methods
    all_asset_datasets = set()
    for method in evaluation_plan.input_cfg.scene_methods:
        all_asset_datasets.update(cfg.models[method].asset_datasets)
    dataset_cfgs = {asset: cfg.assets[asset] for asset in all_asset_datasets}
    mesh_retriever = Retriever(dataset_cfgs)
    
    # Load a VLM for object matching and other tasks
    vlm_config = getattr(cfg.vlms, evaluation_plan.evaluation_cfg.vlm)
    if cfg.seed:
        with open_dict(vlm_config):
            vlm_config.seed = cfg.seed
    vlm = VLMRegistry.instantiate_vlm(evaluation_plan.evaluation_cfg.vlm, vlm_config)

    # ----------------------------------------------------------------------------------------
    
    # Load Blender and scene configurations
    scene_cfg = SceneConfig(**cfg.scene)
    blender_cfg = BlenderConfig(**cfg.blender)
    trimesh_cfg = TrimeshConfig(**cfg.trimesh)

    # ----------------------------------------------------------------------------------------
    # SCENE EVALUATION
    # ----------------------------------------------------------------------------------------

    # Load scene states and do the evaluations
    for method in scenes_per_method.keys():
        
        print(f"Evaluting: {method} - {len(scenes_per_method[method])} scenes")
        
        # Adjust whether to use simple architecture for all scenes of the method based on the evaluation plan
        scene_cfg.use_simple_architecture = method in evaluation_plan.input_cfg.method_use_simple_architecture
        
        # Evaluate each scene for the method
        method_scene_files = scenes_per_method[method]
        for i, method_scene_file in enumerate(method_scene_files):

            print(f"--- {method} ({i+1}/{len(method_scene_files)}) --- {method_scene_file}\n")
            print(f"*** Load scene with simple architecture? -> {scene_cfg.use_simple_architecture} ***\n")

            # ---------------------------------------------------
            
            # Load scene state, if it fails, load an empty scene
            try:
                scene_state = SceneState(method_scene_file)
            except Exception as e:
                print(f"Error loading scene: {method_scene_file} - error: {e}")
                scene_state = SceneState(pathlib.Path("./input/empty_scene.json")) # TODO: Make this a config
                if hasattr(method_scene_file, "stem"):
                    scene_state.name = method_scene_file.stem
            
            # Create the output directory
            output_dir: pathlib.Path = pathlib.Path(evaluation_plan.evaluation_cfg.output_dir) / cfg.models[method].output_dir_name / scene_state.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Set up per-scene logging (logs to both file and stdout)
            log_path = output_dir / "eval.log"
            with FileLoggingContext(log_file_path=log_path, suppress_stdout=False):

                # Create the scene with output directory
                scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, output_dir)

                # ---------------------------------------------------

                # Save the Blender scene for reference if configured
                if evaluation_plan.evaluation_cfg.save_blend_file:
                    scene.blender_scene.save_blend()

                # Save trimesh scene as GLB for debugging coordinate transforms
                if evaluation_plan.evaluation_cfg.save_trimesh_glb:
                    scene.export_trimesh(str(output_dir / "trimesh_scene.glb"))

                # ---------------------------------------------------

                # Render as requested
                if evaluation_plan.render_cfg.normal_render_tasks:
                    _render_scene(scene, evaluation_plan.render_cfg.normal_render_tasks)

                # For SceneWeaver: Copy and render original blend file for comparison
                # (GLB exports have texture baking issues, original blend has perfect materials)
                if method == "SceneWeaver":
                    _copy_and_render_original_sceneweaver_blend(scene, method_scene_file, output_dir)
                    # Recreate scene after rendering original blend - the bpy.ops.wm.open_mainfile()
                    # call invalidates all Blender object references in the Scene object
                    scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, output_dir)

                # For SceneAgent: Copy and render original blend file for comparison
                if method == "SceneAgent":
                    _copy_and_render_original_scene_agent_blend(scene, method_scene_file, output_dir)
                    # Recreate scene after rendering original blend
                    scene = Scene(mesh_retriever, scene_state, scene_cfg, blender_cfg, trimesh_cfg, output_dir)

                # If no_eval and not semantic_render, can skip the rest
                if evaluation_plan.evaluation_cfg.no_eval and not evaluation_plan.render_cfg.semantic_render_tasks:
                    continue

                # ---------------------------------------------------

                # Get the corresponding annotation
                scene_file_id = scene_state.name.split("_")[-1]
                annotation = annotations[int(scene_file_id)]

                # ---------------------------------------------------

                # Use empty matching result if configured
                # Useful if only running metrics that do not require object matching (e.g., collision)
                if evaluation_plan.evaluation_cfg.use_empty_matching_result:
                    print("Using empty matching result as configured.")
                    matching_result = ObjMatchingResults(per_category={}, not_matched_objs=[], actual_categories={})
                else:
                    matching_result = _get_obj_matching(scene, annotation, vlm, evaluation_plan.evaluation_cfg.use_existing_matching)

                print("Using object matching:")
                for category, obj_ids in matching_result.per_category.items():
                    print(f" - {category}: {obj_ids}")
                print()

                # ---------------------------------------------------

                # Apply semantic colors, render, and save the scene
                if evaluation_plan.render_cfg.semantic_render_tasks:
                    color_reference_path = pathlib.Path(evaluation_plan.render_cfg.semantic_color_reference.replace("*", scene_file_id)) if evaluation_plan.render_cfg.semantic_color_reference else None
                    apply_semantic_colors(scene, matching_result, vlm, color_reference_path)
                    _render_scene(scene, evaluation_plan.render_cfg.semantic_render_tasks)
                    if evaluation_plan.evaluation_cfg.save_blend_file:
                        scene.blender_scene.save_blend("scene_semantic_colors.blend")

                # ---------------------------------------------------

                # Skip evaluation if no_eval is set
                if evaluation_plan.evaluation_cfg.no_eval:
                    continue

                # Initialize output json
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

                # Run metrics
                common_metric_params = {
                    "scene": scene,
                    "annotation": annotation,
                    "vlm": vlm,
                    "matching_result": matching_result,
                    "output_dir": output_dir
                }
                for metric_name in metrics_to_run:
                    print(f"----- Running metric: {metric_name}")
                    metric_instance = MetricRegistry.instantiate_metric(metric_name, metric_configs, **common_metric_params)

                    result = metric_instance.run(evaluation_plan.evaluation_cfg.verbose)
                    output_json["results"][metric_name] = {
                        "message": result.message,
                        "data": result.data
                    }

                    # Save results up to this point
                    output_file = output_dir / f"eval_result.json"
                    with open(output_file, "w") as f:
                        json.dump(output_json, f, indent=4)

                    print()

                print(f"All done. Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()

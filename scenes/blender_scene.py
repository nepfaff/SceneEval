import logging
import warnings
import pathlib
import json
import subprocess
import tempfile
import shutil
import bpy
import shapely
import numpy as np
from dataclasses import dataclass, field
from mathutils import Vector, Matrix

logger = logging.getLogger(__name__)


def _check_glb_has_basisu(file_path: pathlib.Path) -> bool:
    """Check if a GLB file uses KHR_texture_basisu extension."""
    try:
        with open(file_path, 'rb') as f:
            # Read first 100KB to check for the extension string
            header = f.read(100000)
            return b'KHR_texture_basisu' in header
    except Exception:
        return False


def _convert_basisu_glb(file_path: pathlib.Path) -> pathlib.Path:
    """
    Convert a GLB with KHR_texture_basisu to standard textures using gltf-transform.
    Returns path to converted file (cached in same directory with _converted suffix).
    """
    converted_path = file_path.with_stem(file_path.stem + "_converted")

    # Return cached conversion if it exists and is newer than source
    if converted_path.exists():
        if converted_path.stat().st_mtime >= file_path.stat().st_mtime:
            return converted_path

    logger.info(f"Converting basisu textures in {file_path.name}")

    try:
        # Use gltf-transform ktxdecompress to decode KTX2/basisu textures to PNG
        result = subprocess.run(
            ['npx', 'gltf-transform', 'ktxdecompress', str(file_path), str(converted_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=pathlib.Path(__file__).parent.parent  # SceneEval root where node_modules is
        )

        if result.returncode != 0:
            logger.warning(f"gltf-transform ktxdecompress failed: {result.stderr}")
            # Fall back to original file
            return file_path

        logger.info(f"Converted {file_path.name} -> {converted_path.name}")
        return converted_path

    except subprocess.TimeoutExpired:
        logger.warning(f"gltf-transform timed out for {file_path}")
        return file_path
    except FileNotFoundError:
        logger.warning("gltf-transform (npx) not found, skipping basisu conversion")
        return file_path
    except Exception as e:
        logger.warning(f"Failed to convert basisu: {e}")
        return file_path


def _load_door_window_transforms(scene_source_path: pathlib.Path) -> dict:
    """
    Load door/window transform data from the assets directory if available.

    Args:
        scene_source_path: Path to the scene JSON file

    Returns:
        Dictionary mapping hole IDs to transform data, or empty dict if not found
    """
    if scene_source_path is None:
        return {}

    # Look for assets directory next to scene JSON
    assets_dir = scene_source_path.parent / "assets"
    scene_name = scene_source_path.stem  # e.g., "scene_106"

    # Try scene-specific file first, then fall back to legacy filename
    transforms_file = assets_dir / f"door_window_transforms_{scene_name}.json"
    if not transforms_file.exists():
        transforms_file = assets_dir / "door_window_transforms.json"

    if transforms_file.exists():
        with open(transforms_file, 'r') as f:
            return json.load(f)
    return {}


def _import_door_window_obj(obj_path: pathlib.Path, hole_name: str,
                             hole_box_min: list, hole_box_max: list,
                             half_wall_width: float) -> bpy.types.Object:
    """
    Import a door/window OBJ file and position it at hole center.

    Args:
        obj_path: Path to the OBJ file
        hole_name: Name for the Blender object
        hole_box_min: [x, z] min of hole in local wall coords
        hole_box_max: [x, z] max of hole in local wall coords
        half_wall_width: Half the wall width for positioning

    Returns:
        The imported Blender object positioned at hole center in local wall coords
    """
    # Import OBJ - the mesh is in Unity coordinates (Y-up)
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = hole_name

    # The OBJ mesh is in Unity local coordinates (Y-up)
    # Rotate -90° around X to convert Y-up to Z-up
    # Then rotate 180° around Z to flip right-side up (door frame at top, not bottom)
    imported_obj.rotation_euler = (np.radians(-90), 0, np.radians(180))

    # Apply the rotation so mesh vertices are in Z-up space
    bpy.ops.object.select_all(action="DESELECT")
    imported_obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Set origin to geometry center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Calculate hole center in local wall coordinates (same system as placeholders)
    # hole_box is [x_along_wall, z_height] in local wall space
    hole_center_x = -half_wall_width + (hole_box_min[0] + hole_box_max[0]) / 2
    hole_center_z = (hole_box_min[1] + hole_box_max[1]) / 2

    # Move mesh center to hole center position
    imported_obj.location = (hole_center_x, 0, hole_center_z)

    # BAKE the location into vertices so location becomes (0,0,0)
    # This matches how placeholder holes work - vertices at hole position, location at origin
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    return imported_obj

def _log_memory(label: str) -> None:
    """Log current memory usage from /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = line.split()[1]
                    unit = line.split()[2] if len(line.split()) > 2 else "kB"
                    rss_gb = int(rss) / (1024 * 1024) if unit == "kB" else int(rss) / 1024
                    logger.debug(f"[MEMORY] {label}: {rss_gb:.2f} GB")
                    return
    except Exception:
        pass
from .scene_state import SceneState
from .config import SceneConfig
from .obj import Obj
from .architecture import Architecture
from assets import Retriever

# NOTE: prefix b_ for Blender related objects

# ----------------------------------------------------------------------------------------

@dataclass
class BlenderConfig:
    """
    Configuration for Blender.

    Attributes:
        suppress_warning: whether to suppress the warning when initializing a new scene
        render_engine: the render engine to use
        taa_render_samples: the number of render samples
        use_raytracing: whether to use raytracing
        raytracing_method: the raytracing method to use ("SCREEN" or "PROBE")
        trace_max_roughness: the maximum roughness when raytracing
        resolution_x: the resolution in the x direction
        resolution_y: the resolution in the y direction
        environment_map: the environment map file path
        environment_map_strength: the strength of the environment map
        use_transparent_film: whether to use transparent film
        render_file_format: the file format of the render output
        hide_holes_in_render: whether to hide holes in the render output

        object_render_subdir: the subdirectory to save object renders
        default_render_filename: the default render output file name
        default_blend_filename: the default Blender output file name
        default_glb_filename: the default glTF output file name

        camera_location: the camera location
        camera_rotation_euler: the camera rotation in Euler angles
        camera_type: the camera type
        camera_lens_unit: the camera lens unit
        camera_lens: the camera lens value
        camera_bird_view_degree: the camera bird view degree for side-view cameras
        
        human_model: the .glb human model to use
    """
    
    suppress_warning: bool = False

    render_engine: str = "BLENDER_EEVEE"
    taa_render_samples: int = 1024
    use_raytracing: bool = True
    raytracing_method: str = "PROBE"
    trace_max_roughness: float = 0.025
    resolution_x: int = 1024
    resolution_y: int = 1024
    environment_map: str = None
    environment_map_strength: float = 0.3
    use_transparent_film: bool = True
    render_file_format: str = "PNG"
    hide_holes_in_render: bool = True

    object_render_subdir: str = "obj_render"
    default_render_filename: str = "render.png"
    default_blend_filename: str = "scene.blend"
    default_glb_filename: str = "scene.glb"

    camera_location: list[float] = field(default_factory=lambda: [0, 0, 0])
    camera_rotation_euler: list[float] = field(default_factory=lambda: [0, 0, 0])
    camera_type: str = "PERSP"
    camera_lens_unit: str = "MILLIMETERS"
    camera_lens: float = 35
    camera_bird_view_degree: float = 60
    
    human_model: str = None

# ----------------------------------------------------------------------------------------

class BlenderScene:
    def __init__(self,
                 retriever: Retriever,
                 scene_state: SceneState | None,
                 scene_cfg: SceneConfig,
                 blender_cfg: BlenderConfig,
                 output_dir: pathlib.Path) -> None:
        """
        Initialize a BlenderScene object. Note that only one scene can exist at a time.

        Args:
            retriever: the mesh retriever to use for getting object files
            scene_state: the scene state to load
            scene_cfg: the scene configuration
            blender_cfg: the Blender configuration
            output_dir: the output directory for this scene
        """

        self.retriever = retriever
        self.scene_state = scene_state
        self.scene_cfg = scene_cfg
        self.blender_cfg = blender_cfg
        self.output_dir = output_dir
        
        self.b_objs: dict[str, bpy.types.Object] = {}
        self.obj_descriptions: dict[str, str] = {}
        self.inverse_obj_descriptions: dict[str, str] = {}
        self.b_architecture: dict[str, bpy.types.Object] = {}
        self.b_cameras: dict[str, bpy.types.Object] = {}
        self.b_human: bpy.types.Object = None
        
        self.applied_semantic_colors = False

        # Initialize the Blender scene
        self.initialize(blender_cfg)
        if not blender_cfg.suppress_warning:
            warnings.warn("Initialized a new scene. Only one scene can exist at a time.")

        # Load the scene state
        if self.scene_state is not None:
            self.load(scene_state)
        self.obj_pairwise_distances = self.compute_obj_pairwise_distances()
        
        # Load the human model for rendering if provided
        if blender_cfg.human_model is not None:
            bpy.ops.import_scene.gltf(filepath=str(pathlib.Path(blender_cfg.human_model).expanduser().resolve()))
            self.b_human = bpy.context.active_object
            self.b_human.name = "human"
            self.b_human.hide_render = True
    
    def initialize(self, blender_cfg: BlenderConfig) -> None:
        """
        Initialize the Blender scene.

        Args:
            blender_cfg: the Blender configuration
        """

        # Start with a empty scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.types.PreferencesEdit.undo_steps = 1
        bpy.types.PreferencesEdit.undo_memory_limit = 1

        # Render settings
        bpy.context.scene.render.engine = blender_cfg.render_engine
        bpy.context.scene.eevee.taa_render_samples = blender_cfg.taa_render_samples
        bpy.context.scene.eevee.use_raytracing = blender_cfg.use_raytracing
        bpy.context.scene.eevee.ray_tracing_method = blender_cfg.raytracing_method
        bpy.context.scene.eevee.ray_tracing_options.trace_max_roughness = blender_cfg.trace_max_roughness
        bpy.context.scene.render.resolution_x = blender_cfg.resolution_x
        bpy.context.scene.render.resolution_y = blender_cfg.resolution_y
        bpy.context.scene.render.film_transparent = blender_cfg.use_transparent_film
        bpy.context.scene.render.image_settings.file_format = blender_cfg.render_file_format

        # Set environment map
        if blender_cfg.environment_map is not None:
            file_path = pathlib.Path(blender_cfg.environment_map).expanduser().resolve()

            bpy.ops.world.new()
            world = bpy.data.worlds[-1]
            bpy.context.scene.world = world
            world.use_nodes = True
            environment_map_node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            environment_map_node.image = bpy.data.images.load(str(file_path))
            world.node_tree.links.new(world.node_tree.nodes["Background"].inputs[0], environment_map_node.outputs[0])   # Color
            world.node_tree.nodes["Background"].inputs[1].default_value = blender_cfg.environment_map_strength  # Strength
    
    def load(self, scene_state: SceneState) -> None:
        """
        Load a scene state.

        Args:
            scene_state: the scene state to load
        """

        # Load the objects
        for obj in scene_state.objs:
            self.load_obj(obj, self.scene_cfg.skip_missing_obj)

        # Load the architecture
        self.load_architecture(scene_state.architecture, self.scene_cfg.use_simple_architecture)
    
    def update(self) -> None:
        """
        Update the scene to avoid stale data.
        See: https://docs.blender.org/api/current/info_gotchas_internal_data_and_python_objects.html#no-updates-after-setting-values
        """

        bpy.context.view_layer.update()
    
    def get_obj_bounds(self, obj_id: str) -> tuple[np.ndarray]:
        """
        Get the bounding corners of an object.

        Args:
            obj_id: the object ID
        
        Returns:
            min_corner: the minimum corner of the bounding box
            max_corner: the maximum corner of the bounding box
        """

        self.update()
        
        # Gather the bounding points of the object
        bounding_points = []
        obj_parts = [self.b_objs[obj_id]] + list(self.b_objs[obj_id].children_recursive)
        for obj_part in obj_parts:
            if obj_part.type == "MESH":
                obj_part_world_bounding_points = [obj_part.matrix_world @ Vector(obj_part.bound_box[i]) for i in range(8)]
                bounding_points.extend(obj_part_world_bounding_points)
        bounding_points = np.asarray(bounding_points)
        
        if len(bounding_points) == 0:
            # If no meshes are present, return zeros
            warnings.warn(f"No valid bounds for object '{obj_id}'. Are there any meshes in the object?")
            min_corner = np.zeros(3)
            max_corner = np.zeros(3)
        else:
            # Get the min and max corners
            min_corner = np.min(bounding_points, axis=0)
            max_corner = np.max(bounding_points, axis=0)

        return min_corner, max_corner

    def bounding_corners(self, include_walls: bool = False) -> tuple[np.ndarray]:
        """
        Get the bounding corners of all objects in the scene.

        Args:
            include_walls: whether to include the walls in the calculation
        
        Returns:
            min_corner: the minimum corner of the bounding box
            max_corner: the maximum corner of the bounding box
        """

        self.update()

        # Find the bounding box of all objects
        bounding_points = []
        for b_obj in bpy.data.objects:
            if b_obj.type == "MESH":
                if not include_walls and b_obj.name.startswith("wall"):
                    continue

                b_obj_world_bounding_points = [b_obj.matrix_world @ Vector(b_obj.bound_box[i]) for i in range(8)]
                bounding_points.extend(b_obj_world_bounding_points)
        
        bounding_points = np.asarray(bounding_points)
        
        # If no objects are present, return zeros
        if len(bounding_points) == 0:
            warnings.warn("No valid bounds for the scene. Are there any objects in the scene?")
            min_corner = np.zeros(3)
            max_corner = np.zeros(3)
        else:
            min_corner = np.min(bounding_points, axis=0)
            max_corner = np.max(bounding_points, axis=0)

        return min_corner, max_corner
    
    def centroid(self, include_walls: bool = False) -> np.ndarray:
        """
        Get the centroid of all objects in the scene.

        Args:
            include_walls: whether to include the walls in the calculation
        
        Returns:
            centroid: the centroid of all objects
        """

        min_corner, max_corner = self.bounding_corners(include_walls)
        centroid = (min_corner + max_corner) / 2

        return centroid
    
    def compute_obj_pairwise_distances(self) -> dict[str, dict[str, float]]:
        """
        Compute the pairwise distances between all objects in the scene.
        
        Returns:
            pairwise_distances: the pairwise distances between all objects
        """
        
        self.update()
        obj_bounds = {obj_id: self.get_obj_bounds(obj_id) for obj_id in self.b_objs.keys()}
        
        pairwise_distances = {obj_id: {} for obj_id in self.b_objs.keys()}
        for obj_id_1, obj_bounds_1 in obj_bounds.items():
            pairwise_distances[obj_id_1][obj_id_1] = 0
            
            for obj_id_2, obj_bounds_2 in obj_bounds.items():
                if obj_id_1 == obj_id_2:
                    continue
                
                bbox_1_min, bbox_1_max = obj_bounds_1
                bbox_2_min, bbox_2_max = obj_bounds_2
                
                dx = max(0, max(bbox_1_min[0] - bbox_2_max[0], bbox_2_min[0] - bbox_1_max[0]))
                dy = max(0, max(bbox_1_min[1] - bbox_2_max[1], bbox_2_min[1] - bbox_1_max[1]))
                dz = max(0, max(bbox_1_min[2] - bbox_2_max[2], bbox_2_min[2] - bbox_1_max[2]))
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                
                pairwise_distances[obj_id_1][obj_id_2] = distance
                pairwise_distances[obj_id_2][obj_id_1] = distance

        return pairwise_distances
    
    def load_obj(self, obj: Obj, skip_missing: bool = False) -> None:
        """
        Load an object into the scene.

        Args:
            obj: the object to load
            skip_missing: whether to skip missing object files peacefully
        """
        
        self.update()

        # Get the object information from the mesh retriever
        try:
            asset_info = self.retriever.get_asset_info(obj.model_id)
        except FileNotFoundError as e:
            if skip_missing:
                logger.warning(f"Skipping missing asset: {obj.model_id} - {e}")
                return
            else:
                raise
        file_path = asset_info.file_path
        obj_description = asset_info.description
        extra_rotation_transform = asset_info.extra_rotation_transform

        # Check if the object file exists
        if not file_path.is_file():
            if skip_missing:
                logger.warning(f"Object file '{file_path}' is missing.")
                return
            else:
                raise FileNotFoundError(f"Object file '{file_path}' is missing.")
        
        # Import the object based on the file extension
        file_extension = file_path.suffix.lower()
        match file_extension:
            case ".glb" | ".gltf":
                # Check for KHR_texture_basisu and convert if needed (Blender doesn't support it)
                import_path = file_path
                if _check_glb_has_basisu(file_path):
                    import_path = _convert_basisu_glb(file_path)
                bpy.ops.import_scene.gltf(filepath=str(import_path))
            case ".obj":
                bpy.ops.wm.obj_import(filepath=str(file_path.absolute()))
                bpy.ops.object.join()
            case _:
                raise NotImplementedError(f"File extension '{file_extension}' is not supported.")
        
        # The imported object becomes the active object, rename for identification
        b_new_obj = bpy.context.active_object
        expected_name = f"idx{obj.index}_{obj.model_id}"
        b_new_obj.name = expected_name  # Blender may truncate to 63 chars
        # Use expected_name as key (not b_new_obj.name) to match trimesh_scene naming
        self.b_objs[expected_name] = b_new_obj

        # Apply object scale to mesh vertices (normalize the object representation)
        # This ensures matrix_world only contains rotation and translation, not scale
        # Fixes mismatch between Blender (preserves GLB node scale) and Trimesh (flattens to vertices)
        # For objects with scale=1.0 (e.g., SceneWeaver), this is a no-op
        bpy.ops.object.select_all(action='DESELECT')
        b_new_obj.select_set(True)
        bpy.context.view_layer.objects.active = b_new_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Merge duplicate vertices and decimate if needed (to prevent memory exhaustion)
        # Merge first preserves mesh quality better than decimation alone
        MAX_FACES = 250000
        total_faces = sum(len(mesh_obj.data.polygons) for mesh_obj in [b_new_obj] + list(b_new_obj.children_recursive) if mesh_obj.type == 'MESH')
        if total_faces > MAX_FACES:
            # First, merge vertices by distance (Blender's default threshold is 0.001m)
            for mesh_obj in [b_new_obj] + list(b_new_obj.children_recursive):
                if mesh_obj.type == 'MESH':
                    bpy.context.view_layer.objects.active = mesh_obj
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.remove_doubles(threshold=0.001)
                    bpy.ops.object.mode_set(mode='OBJECT')

            # Check face count after merge
            merged_faces = sum(len(mesh_obj.data.polygons) for mesh_obj in [b_new_obj] + list(b_new_obj.children_recursive) if mesh_obj.type == 'MESH')

            if merged_faces < total_faces:
                logger.info(f"Merged vertices {b_new_obj.name}: {total_faces} -> {merged_faces} faces")

            # Only decimate if still over threshold after merge
            if merged_faces > MAX_FACES:
                ratio = MAX_FACES / merged_faces
                for mesh_obj in [b_new_obj] + list(b_new_obj.children_recursive):
                    if mesh_obj.type == 'MESH':
                        modifier = mesh_obj.modifiers.new(name="Decimate", type='DECIMATE')
                        modifier.ratio = ratio
                        bpy.context.view_layer.objects.active = mesh_obj
                        bpy.ops.object.modifier_apply(modifier="Decimate")
                logger.info(f"Decimated {b_new_obj.name}: {merged_faces} -> ~{MAX_FACES} faces")

        # NOTE: For 3D-FUTURE assets, adjust the material's transmission to 0 because they are too shiny
        # This is a workaround for the issue that the 3D-FUTURE assets are too shiny and cause issues with rendering
        # Not well tested, but seems to work for most cases
        if obj.model_id.startswith("3dfModel"):
            for b_mat in b_new_obj.data.materials:
                b_mat.node_tree.nodes["Principled BSDF"].inputs["Transmission Weight"].default_value = 0
        
        # Adjust the object description to be explicit about duplicates
        same_obj_description_count = sum([1 for desc in self.obj_descriptions.values() if desc.startswith(obj_description)])
        if same_obj_description_count > 0:
            obj_description += f" - instance {same_obj_description_count + 1}"
        self.obj_descriptions[b_new_obj.name] = obj_description
        self.inverse_obj_descriptions[obj_description] = b_new_obj.name

        # For 3D-FUTURE objects, absorb any existing rotation into the object
        if "3dfModel" in obj.model_id:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            self.update()
        
        # Apply extra rotation transform from the asset itself to normalize the mesh
        if extra_rotation_transform is not None:
            extra_rotation_transform_matrix = Matrix(extra_rotation_transform)
            b_new_obj.matrix_world = extra_rotation_transform_matrix @ b_new_obj.matrix_world
            
            # Absorb the rotation into the object
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            self.update()
        
        scene_transform = Matrix(np.asarray(obj.transform.data).reshape(4, 4).T)
        b_new_obj.matrix_world = scene_transform @ b_new_obj.matrix_world
        self.update()
        
        # NOTE: This fixes the orientation for some asset types but NOT for SceneWeaver, Scene Agent, or IDesign
        # SceneWeaver and Scene Agent exports are already correctly oriented (Z-up via glTF Y-up conversion)
        # IDesign GLBs are kept in Y-up so Blender's automatic Y-up to Z-up conversion works correctly
        if not obj.model_id.startswith(("sceneweaver", "sw.")) and not obj.model_id.startswith("scene-agent") and not obj.model_id.startswith("idesign"):
            neg_90_x = Matrix.Rotation(-np.pi / 2, 4, "X")
            b_new_obj.matrix_world = b_new_obj.matrix_world @ neg_90_x

        self.update()
    
    def load_architecture(self, architecture: Architecture = None, use_simple: bool = False) -> None:
        """
        Load an architecture into the scene.

        Args:
            architecture: the architecture to load
            use_simple: whether to use a simple architecture instead of the provided architecture
        """

        self.update()

        # Get the bounding box of all objects
        objs_bbox_min, objs_bbox_max = self.bounding_corners()
        length_x = objs_bbox_max[0] - objs_bbox_min[0]
        length_y = objs_bbox_max[1] - objs_bbox_min[1]

        # Get the floor height
        floor_height = objs_bbox_min[2]

        # Load door/window OBJ transforms if available
        door_window_transforms = {}
        if self.scene_state and self.scene_state.source_path:
            door_window_transforms = _load_door_window_transforms(self.scene_state.source_path)
            if door_window_transforms:
                logger.info(f"Loaded {len(door_window_transforms)} door/window transforms")

        # Check for converted SceneWeaver architecture GLBs if enabled
        # These have baked textures from the original scene
        converted_floor_loaded = False
        converted_walls_loaded = False

        # Auto-detect GLB architecture based on model ID prefix (SceneWeaver, Holodeck)
        if self.scene_state and self.scene_state.objs:
            # Check if this is a SceneWeaver scene by looking at object model IDs
            first_obj_model_id = self.scene_state.objs[0].model_id if self.scene_state.objs else None
            logger.info(f"first_obj_model_id: {first_obj_model_id}")
            if first_obj_model_id and first_obj_model_id.startswith(("sceneweaver.", "sw.")):
                # Extract scene ID and find assets directory
                # Format: sceneweaver.scene_0__obj_name or sw.s0__obj_name
                try:
                    asset_info = self.retriever.get_asset_info(first_obj_model_id)
                    # Asset path is like: .../scene_X/assets/obj_name.glb
                    # Architecture GLBs are in the same directory
                    assets_dir = asset_info.file_path.parent

                    # Check for converted architecture GLBs
                    floor_glb = assets_dir / "architecture_floor.glb"
                    walls_glb = assets_dir / "architecture_walls.glb"

                    if floor_glb.exists():
                        bpy.ops.import_scene.gltf(filepath=str(floor_glb))
                        # Register imported objects as architecture
                        for obj in bpy.context.selected_objects:
                            obj.name = f"floor_{obj.name}"
                            self.b_architecture[obj.name] = obj
                            # Position at correct floor height
                            obj.location.z = floor_height
                        converted_floor_loaded = True

                    if walls_glb.exists():
                        bpy.ops.import_scene.gltf(filepath=str(walls_glb))
                        # Register imported objects as architecture
                        for obj in bpy.context.selected_objects:
                            obj.name = f"wall_{obj.name}"
                            self.b_architecture[obj.name] = obj
                        converted_walls_loaded = True

                except Exception as e:
                    # Fall back to procedural architecture on any error
                    pass

            # Check for Holodeck scenes (model IDs starting with "objaverse.")
            elif first_obj_model_id and first_obj_model_id.startswith("objaverse."):
                try:
                    # For Holodeck, the architecture GLB is in the assets directory
                    # alongside the scene JSON. Use the scene_state file path to find it.
                    if hasattr(self.scene_state, 'source_path') and self.scene_state.source_path:
                        from pathlib import Path
                        scene_json_path = Path(self.scene_state.source_path)
                        assets_dir = scene_json_path.parent / "assets"
                        scene_name = scene_json_path.stem  # e.g., "scene_106"

                        # Check for scene-specific architecture GLB
                        arch_glb = assets_dir / f"architecture_{scene_name}.glb"
                        logger.info(f"Looking for architecture GLB at: {arch_glb}")
                        if arch_glb.exists():
                            logger.info(f"Loading architecture GLB: {arch_glb}")
                            bpy.ops.import_scene.gltf(filepath=str(arch_glb))
                            # Register imported objects as architecture
                            imported_objs = list(bpy.context.selected_objects)

                            # Fix Unity to Blender coordinate system mismatch
                            # Unity is left-handed (X-right, Y-up, Z-forward)
                            # glTF/Blender are right-handed with different axis mapping
                            # The glTF importer's Y-up to Z-up rotation maps:
                            #   glTF X → Blender X
                            #   glTF Y → Blender Z
                            #   glTF Z → Blender -Y
                            # If UnityGLTF doesn't properly negate Z for handedness,
                            # we get both X and Y flipped in Blender's top-down view.
                            # Fix: 180° rotation around Z (negate both X and Y scale)
                            # This has determinant=1, so normals are preserved.
                            logger.info("Applying 180° rotation (XY flip) to fix coordinate system")
                            for obj in imported_objs:
                                if obj.parent is None:
                                    # 180° rotation via scale negation on root objects
                                    obj.scale.x = -obj.scale.x
                                    obj.scale.y = -obj.scale.y

                            # Make mesh data single-user before applying transform
                            # (some imported objects share mesh data which blocks transform_apply)
                            for obj in imported_objs:
                                if obj.type == 'MESH' and obj.data and obj.data.users > 1:
                                    obj.data = obj.data.copy()

                            # Apply the scale transform so mesh vertices are actually mirrored
                            bpy.ops.object.select_all(action='DESELECT')
                            for obj in imported_objs:
                                obj.select_set(True)
                            bpy.context.view_layer.objects.active = imported_objs[0] if imported_objs else None
                            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                            # Calculate offset to align GLB with scene coordinates
                            # The GLB is exported in Unity's coordinate system, but
                            # scene objects are converted with a different transform.
                            # We need to shift the architecture to match.
                            # Get floor bounds from scene JSON
                            with open(scene_json_path, 'r') as f:
                                scene_json = json.load(f)
                            arch_elements = scene_json.get('scene', {}).get('arch', {}).get('elements', [])
                            floor_elem = next((e for e in arch_elements if e.get('type') == 'Floor'), None)

                            if floor_elem and imported_objs:
                                # Get expected floor bounds from scene JSON
                                floor_points = floor_elem.get('points', [])
                                if floor_points:
                                    xs = [p[0] for p in floor_points]
                                    ys = [p[1] for p in floor_points]
                                    expected_min_x = min(xs)
                                    expected_min_y = min(ys)
                                    expected_max_x = max(xs)
                                    expected_max_y = max(ys)

                                    # Find the floor object in imported objects
                                    # Look for room-named objects that are NOT doors, windows, walls, ceilings
                                    # Floor must have mesh data (not an empty container)
                                    floor_obj = None
                                    room_id = floor_elem.get('roomId', '')

                                    # First priority: exact match with room ID and has mesh data
                                    for obj in imported_objs:
                                        if obj.name == room_id and obj.type == 'MESH' and obj.data:
                                            floor_obj = obj
                                            logger.info(f"Found floor object by room ID: {obj.name}")
                                            break

                                    # Second priority: floor-like names with mesh data
                                    if floor_obj is None:
                                        for obj in imported_objs:
                                            if obj.type != 'MESH' or not obj.data:
                                                continue
                                            name_lower = obj.name.lower()
                                            # Skip non-floor objects
                                            if any(x in name_lower for x in ['door', 'window', 'wall', 'ceiling']):
                                                continue
                                            # Look for floor-like objects
                                            if 'floor' in name_lower or obj.name in ['bedroom', 'living_room', 'kitchen', 'bathroom', 'office', 'dining_room']:
                                                floor_obj = obj
                                                logger.info(f"Found floor object: {obj.name}")
                                                break

                                    if floor_obj:
                                        # Get imported floor bounds (in Blender coords after glTF import and mirror)
                                        # Need to get world-space bounds
                                        bbox = [floor_obj.matrix_world @ Vector(corner) for corner in floor_obj.bound_box]
                                        imported_min_x = min(v.x for v in bbox)
                                        imported_min_y = min(v.y for v in bbox)

                                        # Calculate offset needed
                                        offset_x = expected_min_x - imported_min_x
                                        offset_y = expected_min_y - imported_min_y

                                        logger.info(f"Architecture offset: ({offset_x:.2f}, {offset_y:.2f}, 0)")

                                        # Apply offset only to ROOT objects (no parent)
                                        # Child objects inherit the transform from parents
                                        for obj in imported_objs:
                                            if obj.parent is None:
                                                obj.location.x += offset_x
                                                obj.location.y += offset_y

                            # Remove ceiling objects - we don't want them in renders
                            ceilings_to_remove = [obj for obj in imported_objs if 'ceiling' in obj.name.lower()]
                            for ceiling_obj in ceilings_to_remove:
                                logger.info(f"Removing ceiling object: {ceiling_obj.name}")
                                bpy.data.objects.remove(ceiling_obj, do_unlink=True)
                                imported_objs.remove(ceiling_obj)

                            # Remove exterior wall objects - we only want interior walls for renders
                            # Exterior walls have "|exterior" suffix and different materials
                            exteriors_to_remove = [obj for obj in imported_objs if '|exterior' in obj.name.lower()]
                            for ext_obj in exteriors_to_remove:
                                logger.info(f"Removing exterior wall: {ext_obj.name}")
                                bpy.data.objects.remove(ext_obj, do_unlink=True)
                                imported_objs.remove(ext_obj)

                            for obj in imported_objs:
                                self.b_architecture[obj.name] = obj
                            logger.info(f"Loaded {len(imported_objs)} objects from architecture GLB")
                            # Mark as loaded to skip procedural generation
                            converted_floor_loaded = True
                            converted_walls_loaded = True
                        else:
                            logger.info(f"Architecture GLB not found: {arch_glb}")
                    else:
                        logger.info("No source_path on scene_state for Holodeck scene")
                except Exception as e:
                    # Fall back to procedural architecture on any error
                    logger.warning(f"Error loading Holodeck architecture GLB: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    pass

        # Prepare materials (needed for non-converted walls or as fallback)
        # Wall material with default gray color (#888899)
        wall_material = bpy.data.materials.new(name="wall_material")
        wall_material.use_nodes = True
        wall_nodes = wall_material.node_tree.nodes
        wall_nodes.clear()
        wall_bsdf = wall_nodes.new("ShaderNodeBsdfPrincipled")
        wall_output = wall_nodes.new("ShaderNodeOutputMaterial")
        wall_bsdf.inputs["Base Color"].default_value = (0.85, 0.82, 0.78, 1.0)  # Warm off-white wall color
        wall_material.node_tree.links.new(wall_bsdf.outputs["BSDF"], wall_output.inputs["Surface"])

        # Floor material with wood-like tan color
        floor_material = bpy.data.materials.new(name="floor_material")
        floor_material.use_nodes = True
        floor_nodes = floor_material.node_tree.nodes
        floor_nodes.clear()
        floor_bsdf = floor_nodes.new("ShaderNodeBsdfPrincipled")
        floor_output = floor_nodes.new("ShaderNodeOutputMaterial")
        floor_bsdf.inputs["Base Color"].default_value = (0.72, 0.58, 0.42, 1.0)  # Wood/tan floor color
        floor_material.node_tree.links.new(floor_bsdf.outputs["BSDF"], floor_output.inputs["Surface"])

        door_material = bpy.data.materials.new(name="door_material")
        door_material.use_nodes = True
        diffuse_node = door_material.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse_node.inputs["Color"].default_value = (0.31, 0.76, 0.40, 1)
        door_material.node_tree.links.new(door_material.node_tree.nodes["Material Output"].inputs["Surface"], diffuse_node.outputs["BSDF"])

        window_material = bpy.data.materials.new(name="window_material")
        window_material.use_nodes = True
        diffuse_node = window_material.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse_node.inputs["Color"].default_value = (0.37, 0.76, 0.90, 1)
        window_material.node_tree.links.new(window_material.node_tree.nodes["Material Output"].inputs["Surface"], diffuse_node.outputs["BSDF"])

        # Skip floor/wall generation if converted architecture was loaded
        if converted_floor_loaded and converted_walls_loaded:
            return

        # Build the architecture
        if use_simple or architecture is None:
            # Use a simple architecture - a rectangular floor

            # Create the floor plane
            bpy.ops.mesh.primitive_plane_add(size=1)
            floor = bpy.context.active_object
            floor.name = "floor"
            self.b_architecture[floor.name] = floor

            # Set the floor properties
            floor.scale = (length_x, length_y, 1)
            floor.location = (objs_bbox_min[0] + length_x / 2, objs_bbox_min[1] + length_y / 2, floor_height)

            # Set the floor material
            floor.data.materials.append(floor_material)

            # Prepare wall locations and material
            center_x, center_y, center_z = floor.location
            half_length_x, half_length_y = length_x / 2, length_y / 2
            wall_locations_xy = [  # Order: facing -y, x, y, -x
                (center_x, center_y + half_length_y),
                (center_x - half_length_x, center_y),
                (center_x, center_y - half_length_y),
                (center_x + half_length_x, center_y)
            ]
            
            # Create the walls
            for i, location_xy in enumerate(wall_locations_xy):
                wall = bpy.ops.mesh.primitive_plane_add(size=1)
                wall = bpy.context.active_object
                wall.name = f"wall_{i}"
                self.b_architecture[wall.name] = wall
                
                wall.rotation_euler = (np.pi / 2, 0, 0)
                wall.scale = (length_x if i % 2 == 0 else length_y, self.scene_cfg.simple_arch_wall_height, 1)
                bpy.ops.object.transform_apply()
                
                # Set the wall properties
                wall.location = location_xy[:2] + (center_z + self.scene_cfg.simple_arch_wall_height / 2,)
                wall.rotation_euler = (0, 0, i * np.pi / 2)

                # Set the wall material
                wall.visible_shadow = False
                wall.data.materials.append(wall_material)
            
        else:
            
            # Store floor polygon here for later checking wall facing direction
            floor_polygon = None

            # Track OBJ-loaded holes across all wall elements (to avoid duplicates)
            obj_loaded_hole_names = set()  # For visibility/material tracking
            obj_loaded_hole_ids = set()    # To prevent loading same OBJ twice

            # Use the provided architecture
            for element in architecture.elements:
                    
                # Check the element type
                match element.type:
                    case "Floor":
                        # Skip if floor already loaded from GLB
                        if converted_floor_loaded:
                            # Still need to compute floor polygon for wall facing direction
                            floor_points = np.asarray(element.points)
                            floor_polygon = shapely.geometry.Polygon(floor_points[..., :2])
                            continue

                        floor_name = f"floor_{element.roomId.replace(' ', '_')}"

                        # Create a empty mesh and object
                        floor_mesh = bpy.data.meshes.new(floor_name)
                        floor = bpy.data.objects.new(floor_name, floor_mesh)
                        bpy.context.collection.objects.link(floor)
                        floor_name = floor.name

                        # Select the floor object
                        bpy.ops.object.select_all(action="DESELECT")
                        floor.select_set(True)
                        bpy.context.view_layer.objects.active = floor

                        # Gather the vertices and edges for the floor
                        floor_points = np.asarray(element.points)
                        floor_edges = [(i, i + 1) for i in range(len(floor_points) - 1)] + [(len(floor_points) - 1, 0)]

                        # Add the vertices and edges to the mesh
                        floor_mesh.from_pydata(floor_points, floor_edges, [])
                        floor_mesh.update()

                        # Fill in the faces
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.fill()
                        bpy.ops.object.mode_set(mode="OBJECT")

                        # Translate the floor to the correct height
                        floor.location = (0, 0, floor_height)

                        # Set the floor material (clear existing materials first)
                        floor.data.materials.clear()
                        floor.data.materials.append(floor_material)

                        self.b_architecture[floor_name] = floor

                        # Extra, store the 2D floor polygon for later checking wall facing direction
                        floor_polygon = shapely.geometry.Polygon(floor_points[..., :2])
                        
                    case "Ceiling":
                        pass
                    
                    case "Wall":
                        # Skip if walls already loaded from GLB (includes doors/windows with materials)
                        if converted_walls_loaded:
                            continue

                        # Skip exterior walls (duplicates of interior walls, facing outward)
                        if "exterior" in str(element.id):
                            continue

                        wall_name = f"wall_{element.id}"

                        # Create a empty mesh and object for the wall
                        wall_mesh = bpy.data.meshes.new(wall_name)
                        wall = bpy.data.objects.new(wall_name, wall_mesh)
                        bpy.context.collection.objects.link(wall)
                        wall_name = wall.name
                        
                        # Select the wall object
                        bpy.ops.object.select_all(action="DESELECT")
                        wall.select_set(True)
                        bpy.context.view_layer.objects.active = wall
                        
                        # Gather the vertices and edges for the wall
                        wall_from_to_points = element.points
                        wall_width = np.linalg.norm(np.asarray(wall_from_to_points[0]) - np.asarray(wall_from_to_points[1]))
                        half_wall_width = wall_width / 2
                        wall_points = [
                            (-half_wall_width, 0, 0),
                            (half_wall_width, 0, 0),
                            (half_wall_width, 0, element.height),
                            (-half_wall_width, 0, element.height)
                        ]
                        wall_edges = [(i, i + 1) for i in range(len(wall_points) - 1)] + [(len(wall_points) - 1, 0)]
                        
                        # Gather the vertices and edges for the holes in the wall if any
                        holes = {}
                        if len(element.holes) > 0:
                            for hole_idx, hole in enumerate(element.holes):
                                hole_points = [
                                    (-half_wall_width + hole.box_min[0], 0, hole.box_min[1]),
                                    (-half_wall_width + hole.box_max[0], 0, hole.box_min[1]),
                                    (-half_wall_width + hole.box_max[0], 0, hole.box_max[1]),
                                    (-half_wall_width + hole.box_min[0], 0, hole.box_max[1])
                                ]
                                next_edge_index = len(wall_points)
                                hole_edges = [(i, i + 1) for i in range(next_edge_index, next_edge_index + len(hole_points) - 1)] + [(next_edge_index + len(hole_points) - 1, next_edge_index)]
                                
                                wall_points.extend(hole_points)
                                wall_edges.extend(hole_edges)

                                # If the hole is exactly at the wall's edge, remove the duplicate edges
                                # TODO: This is a hacky solution, need to find a better way
                                # at_max_edge = any([np.isclose(hole_point[0], half_wall_width) for hole_point in hole_points])
                                # at_min_edge = any([np.isclose(hole_point[0], -half_wall_width) for hole_point in hole_points])
                                # if at_max_edge:
                                #     wall_edges.remove((0, 1))
                                #     wall_edges.remove((1, 2))
                                #     wall_edges.remove((4, 5))
                                #     wall_edges.remove((5, 6))
                                #     wall_edges.append((0, hole_edges[0][0]))
                                #     wall_edges.append((hole_edges[1][1], 2))
                                # if at_min_edge:
                                #     wall_edges.remove((0, 1))
                                #     wall_edges.remove((3, 0))
                                #     wall_edges.remove((4, 5))
                                #     wall_edges.remove((7, 4))
                                #     wall_edges.append((3, hole_edges[3][0]))
                                #     wall_edges.append((hole_edges[0][1], 1))
                                
                                # --------------------------------------------------------------
                                # Additionally make the hole as a separate object
                                hole_type = hole.type.lower()
                                hole_name = f"{hole_type}_{hole_idx}_{element.id}"
                                hole_obj = None

                                # Check if we have an OBJ file for this hole
                                hole_id = hole.id if hasattr(hole, 'id') and hole.id else None
                                if hole_id and self.scene_state.source_path:
                                    # Skip if already loaded (same hole on interior+exterior wall)
                                    if hole_id in obj_loaded_hole_ids:
                                        obj_loaded_hole_names.add(hole_name)  # Still track name for visibility
                                        hole_obj = None  # Use placeholder for this duplicate
                                    else:
                                        # Load OBJ file positioned at hole center (from JSON)
                                        assets_dir = self.scene_state.source_path.parent / "assets"
                                        scene_name = self.scene_state.source_path.stem
                                        # Try scene-specific file first, then fall back to legacy filename
                                        obj_file = assets_dir / f"{scene_name}_{hole_id}.obj"
                                        if not obj_file.exists():
                                            obj_file = assets_dir / f"{hole_id}.obj"
                                        if obj_file.exists():
                                            try:
                                                hole_obj = _import_door_window_obj(
                                                    obj_file,
                                                    hole_name,
                                                    hole.box_min,
                                                    hole.box_max,
                                                    half_wall_width
                                                )
                                                obj_loaded_hole_ids.add(hole_id)  # Mark as loaded
                                                obj_loaded_hole_names.add(hole_name)  # Track for visibility
                                                logger.info(f"Loaded door/window OBJ: {hole_id}")
                                            except Exception as e:
                                                logger.warning(f"Failed to load door/window OBJ {hole_id}: {e}")
                                                hole_obj = None

                                # Fall back to placeholder mesh if OBJ not loaded
                                if hole_obj is None:
                                    # Create a empty mesh and object for the hole
                                    hole_mesh = bpy.data.meshes.new(hole_name)
                                    hole_obj = bpy.data.objects.new(hole_name, hole_mesh)
                                    bpy.context.collection.objects.link(hole_obj)

                                    # Select the hole object
                                    bpy.ops.object.select_all(action="DESELECT")
                                    hole_obj.select_set(True)
                                    bpy.context.view_layer.objects.active = hole_obj

                                    # Add the vertices and edges to the hole mesh
                                    local_hole_edges = [(i - next_edge_index, j - next_edge_index) for i, j in hole_edges]
                                    hole_mesh.from_pydata(hole_points, local_hole_edges, [])
                                    hole_mesh.update()

                                    # Fill in the faces of the hole
                                    bpy.ops.object.mode_set(mode="EDIT")
                                    bpy.ops.mesh.fill()
                                    bpy.ops.object.mode_set(mode="OBJECT")

                                hole_name = hole_obj.name

                                # Hide placeholder holes from rendering (NOT OBJ-loaded ones)
                                if hole_name not in obj_loaded_hole_names:
                                    hole_obj.hide_render = self.blender_cfg.hide_holes_in_render

                                # Set the hole of the wall
                                holes[hole_name] = hole_obj
                                
                        # Select the wall object
                        bpy.ops.object.select_all(action="DESELECT")
                        wall.select_set(True)
                        bpy.context.view_layer.objects.active = wall
                        
                        # Add the vertices and edges to the wall mesh
                        wall_mesh.from_pydata(wall_points, wall_edges, [])
                        wall_mesh.update()
                        
                        # Fill in the faces of the wall
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.fill()
                        bpy.ops.object.mode_set(mode="OBJECT")
                        
                        # Rotate the wall, rotation angle is based on the order of two points
                        if np.isclose(wall_from_to_points[1][0], wall_from_to_points[0][0]):
                            # Wall aligned with y-axis
                            if wall_from_to_points[1][1] > wall_from_to_points[0][1]:
                                rotation = (0, 0, np.pi / 2)
                                wall.rotation_euler = rotation
                                for hole_name, hole in holes.items():
                                    hole.rotation_euler = rotation
                            else:
                                rotation = (0, 0, -np.pi / 2)
                                wall.rotation_euler = rotation
                                for hole_name, hole in holes.items():
                                    hole.rotation_euler = rotation
                        else:
                            # Wall aligned with x-axis
                            if wall_from_to_points[1][0] < wall_from_to_points[0][0]:
                                rotation = (0, 0, np.pi)
                                wall.rotation_euler = rotation
                                for hole_name, hole in holes.items():
                                    hole.rotation_euler = rotation

                        # Translate the wall to the center of the two points and adjust the height
                        translation = np.mean([wall_from_to_points[0], wall_from_to_points[1]], axis=0)
                        translation += [0, 0, floor_height]
                        wall.location = translation
                        for hole_name, hole in holes.items():
                            hole.location = translation

                        # Set the origin of the wall to the center of the wall
                        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

                        # Repeat the same for the holes (skip OBJ-loaded - they already have origin set)
                        for hole_name, hole in holes.items():
                            if hole_name not in obj_loaded_hole_names:
                                bpy.ops.object.select_all(action="DESELECT")
                                hole.select_set(True)
                                bpy.context.view_layer.objects.active = hole
                                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
                        bpy.ops.object.select_all(action="DESELECT")
                        wall.select_set(True)
                        bpy.context.view_layer.objects.active = wall
                        
                        # Check if the wall is facing inwards or outwards of the room
                        if floor_polygon is not None:
                            
                            # Avoid stale data when checking wall facing direction
                            self.update()
                            
                            # Get a point slightly in front of the wall
                            wall_front_vector = np.asarray(wall.matrix_world)[:3, :3] @ np.asarray([0, -1, 0])
                            wall_front_point = (np.asarray(wall.location) + wall_front_vector * 0.1)[:2]
                            wall_front_point = shapely.geometry.Point(wall_front_point)
                            
                            # If the wall front point is not inside the floor polygon in 2D, flip the wall
                            if not floor_polygon.contains(wall_front_point):
                                wall.scale[0] *= -1
                                wall.rotation_euler[2] += np.pi
                                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True) # Apply the scale to the mesh

                                # Repeat the same for placeholder holes (OBJ holes handle this differently)
                                for hole_name, hole in holes.items():
                                    if hole_name not in obj_loaded_hole_names:
                                        bpy.ops.object.select_all(action="DESELECT")
                                        hole.select_set(True)
                                        bpy.context.view_layer.objects.active = hole
                                        hole.scale[0] *= -1
                                        hole.rotation_euler[2] += np.pi
                                        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                            
                        else:
                            warnings.warn("Floor polygon is not available for checking wall facing direction. Maybe the load order is incorrect?")
                        
                        # Set the wall material (clear existing materials first)
                        wall.visible_shadow = False
                        wall.data.materials.clear()
                        wall.data.materials.append(wall_material)
                        for hole_name, hole in holes.items():
                            hole.visible_shadow = False
                            # Only apply placeholder materials to non-OBJ holes
                            if hole_name not in obj_loaded_hole_names:
                                match hole_name.split("_")[0]:
                                    case "door":
                                        hole.data.materials.append(door_material)
                                    case "window":
                                        hole.data.materials.append(window_material)
                        
                        self.b_architecture[wall_name] = wall
                        for hole_name, hole in holes.items():
                            self.b_architecture[hole_name] = hole
                        
        self.update()

    def _add_default_camera(self, name: str = "camera") -> bpy.types.Object:
        """
        Add a default camera to the scene.

        Args:
            name: the name of the camera
        
        Returns:
            b_camera: the Blender camera object
        """

        bpy.ops.object.camera_add()
        
        # Add the camera to the li
        b_camera = bpy.context.active_object
        b_camera.name = name
        self.b_cameras[b_camera.name] = b_camera

        # Set the default camera properties
        b_camera.rotation_euler = self.blender_cfg.camera_rotation_euler
        b_camera.data.type = self.blender_cfg.camera_type
        b_camera.data.lens_unit = self.blender_cfg.camera_lens_unit
        b_camera.data.lens = self.blender_cfg.camera_lens
        b_camera.data.clip_start = 0.001  # 1mm near clip plane (default 0.1m clips small objects)

        return b_camera
    
    def _remove_camera(self, name: str) -> None:
        """
        Remove a camera from the scene.

        Args:
            name: the name of the camera
        """

        b_camera = self.b_cameras.get(name)
        if b_camera is not None:
            bpy.ops.object.select_all(action="DESELECT")
            b_camera.select_set(True)
            bpy.ops.object.delete()
            self.b_cameras.pop(name)
    
    def _set_elements_hide_render(self,
                                  b_objs: list[bpy.types.Object] | bpy.types.Object = None,
                                  b_archs: list[bpy.types.Object] | bpy.types.Object = None,
                                  reverse_selection: bool = False,
                                  hide: bool = True) -> None:
        """
        Set the hide_render state for the objects and architecture elements.

        Args:
            b_objs: the object(s) to hide
            b_archs: the architecture element(s) to hide
            reverse_selection: whether to reverse the selection
            hide: the value to set the hide_render state to
        """
        
        # Objects
        if b_objs is not None:

            # Input normalization
            if not isinstance(b_objs, list):
                b_objs = [b_objs]
                
            # Get the objects to hide
            if not reverse_selection:
                b_objs_to_hide = b_objs
            else:
                b_objs_to_hide = [b_obj for b_obj in self.b_objs.values() if b_obj not in b_objs]
            
            # Set the hide_render state
            for b_obj in b_objs_to_hide:
                b_obj.hide_render = hide
                for b_obj_child in b_obj.children_recursive:
                    b_obj_child.hide_render = hide
        
        # Architecture elements
        if b_archs is not None:
            
            # Input normalization
            if not isinstance(b_archs, list):
                b_archs = [b_archs]
            
            # Get the architecture elements to hide
            if not reverse_selection:
                b_archs_to_hide = b_archs
            else:
                b_archs_to_hide = [b_arch for b_arch in self.b_architecture.values() if b_arch not in b_archs]
            
            # Set the hide_render state
            for b_arch in b_archs_to_hide:
                # Skip windows and doors - their render visibility is controlled by config
                if self.blender_cfg.hide_holes_in_render and (b_arch.name.startswith("window") or b_arch.name.startswith("door")):
                    continue
                b_arch.hide_render = hide
    
    def _get_elements_blocking_camera_line_of_sight(self, b_camera: bpy.types.Object, target_obj: bpy.types.Object) -> list[bpy.types.Object]:
        """
        Get the elements that are blocking the line of sight between the camera and the target object via raycasting.

        Args:
            b_camera: the camera object
            target_obj: the target object
        
        Returns:
            blocking_objs: the objects blocking the line of sight
            blocking_archs: the architecture elements blocking the line of sight
        """
        
        self.update()
        
        # Get the camera and bounding box corners of the target object
        camera_location = b_camera.location
        target_bbox_corners = (np.asarray(target_obj.matrix_world)[:3, :3] @ np.asarray(target_obj.bound_box).T).T + np.asarray(target_obj.location)
        
        # Prepare the ray origin and directions to the target object's bounding box corners
        ray_origin = camera_location
        ray_directions = target_bbox_corners - camera_location
        to_target_ray_distances = np.linalg.norm(ray_directions, axis=1)
        ray_directions /= to_target_ray_distances[:, None]
        
        # Raycast against all other objects except the target object
        blocking_objs = []
        for b_obj_other in self.b_objs.values():
            if b_obj_other == target_obj:
                continue
            
            # Only consider mesh objects (including children)
            b_obj_other_parts = [b_obj_other] + list(b_obj_other.children_recursive)
            b_obj_other_parts = [b_obj_other_part for b_obj_other_part in b_obj_other_parts if b_obj_other_part.type == "MESH"]
            
            for b_obj_other_part in b_obj_other_parts:
            
                # Transform the ray origin and directions to the local space of the other object
                local_ray_origin = b_obj_other_part.matrix_world.inverted() @ ray_origin
                local_ray_directions = [np.asarray(b_obj_other_part.matrix_world.inverted())[:3, :3] @ ray_dir for ray_dir in ray_directions]
                
                # Perform the raycasting
                ray_results = [b_obj_other_part.ray_cast(local_ray_origin, local_ray_dir) for local_ray_dir in local_ray_directions]
                
                # For rays that hit the other object, check if the hit point is closer than the target object
                for i, result in enumerate(ray_results):
                    if not result[0]: # Skip rays that did not hit
                        continue
                    
                    # Check if the hit point is closer than the target object and the object is not hidden
                    # If so, this object is blocking the line of sight
                    hit_location = b_obj_other_part.matrix_world @ result[1]
                    hit_distance = np.linalg.norm(hit_location - camera_location)
                    
                    # Add the object to the blocking list if a part of it is blocking the line of sight
                    if hit_distance < to_target_ray_distances[i] and not b_obj_other.hide_render:
                        blocking_objs.append(b_obj_other)
                        break
                
                # No need to check further parts of the object if one part is blocking the line of sight
                if b_obj_other in blocking_objs:
                    break
        
        # Raycast against all architectural elements
        blocking_archs = []
        for b_arch_other in self.b_architecture.values():

            # Do not consider the floor for blocking the line of sight
            if "floor" in b_arch_other.name:
                continue

            # Skip objects without mesh data (e.g., empty objects)
            if b_arch_other.type != 'MESH' or b_arch_other.data is None:
                continue
            
            # Transform the ray origin and directions to the local space of the architectural element
            local_ray_origin = b_arch_other.matrix_world.inverted() @ ray_origin
            local_ray_directions = [np.asarray(b_arch_other.matrix_world.inverted())[:3, :3] @ ray_dir for ray_dir in ray_directions]
            
            # Perform the raycasting
            ray_results = [b_arch_other.ray_cast(local_ray_origin, local_ray_dir) for local_ray_dir in local_ray_directions]
            
            ray_block_count = 0
            for i, result in enumerate(ray_results):
                if not result[0]: # Skip rays that did not hit
                    continue
                
                # Check if the hit point is closer than the target object and the architectural element is not hidden
                # If so, this architectural element is blocking the line of sight
                hit_location = b_arch_other.matrix_world @ result[1]
                hit_distance = np.linalg.norm(hit_location - camera_location)
                if hit_distance < to_target_ray_distances[i] and not b_arch_other.hide_render:
                    ray_block_count += 1
            
            if ray_block_count > 4:
                blocking_archs.append(b_arch_other)
        
        return blocking_objs, blocking_archs
    
    def add_top_camera(self) -> None:
        """
        Add a top camera to the scene.
        """

        self.update()

        # Add a camera
        b_top_camera = self._add_default_camera("camera_top")
        bpy.context.scene.camera = b_top_camera

        # Select all objects and move the camera to view them
        bpy.ops.object.select_all(action="DESELECT")
        for b_obj in bpy.data.objects:
            b_obj.select_set(True)
            for b_obj_child in b_obj.children_recursive:
                b_obj_child.select_set(True)
        bpy.ops.view3d.camera_to_view_selected()

        bpy.ops.object.select_all(action="DESELECT")
    
    def render(self, b_camera: bpy.types.Object = None, relative_file_path: pathlib.Path = None) -> None:
        """
        Render the scene.

        Args:
            camera: the camera to use for rendering
            relative_file_path: the path to save the rendered image relative to the output directory
        """

        # Use the provided camera if available
        if b_camera is not None:
            bpy.context.scene.camera = b_camera
        
        # Set up output path
        if relative_file_path is None:
            relative_file_path = self.blender_cfg.default_render_filename
        if self.applied_semantic_colors:
            relative_file_path = pathlib.Path("semantic_renders") / relative_file_path
        bpy.context.scene.render.filepath = str(self.output_dir / relative_file_path)

        # Render using the current scene camera
        _log_memory(f"Blender before bpy.ops.render.render -> {relative_file_path}")
        bpy.ops.render.render(write_still=True)
        _log_memory(f"Blender after bpy.ops.render.render -> {relative_file_path}")
        
    def render_one_obj(self,
                       b_obj: bpy.types.Object,
                       relative_file_path: pathlib.Path = None,
                       hide_others: bool = True,
                       zoom_out: bool = False,
                       with_human_reference: bool = False,
                       bird_view_degree: float = None) -> None:
        """
        Render the scene with only one object.

        Args:
            b_obj: the object to render
            relative_file_path: the path to save the rendered image relative to the output directory
            hide_others: whether to hide other objects in the scene
            zoom_out: whether to zoom out a bit to include the surroundings
            with_human_reference: whether to include a human reference
            side_view: whether to render from the side view instead of the front view
            bird_view_degree: the degree of the camera looking at the object (0 straight down, 90 side view)
        """

        # Add a temporary camera that will be deleted after rendering
        b_camera = self._add_default_camera("camera_temp")
        bpy.context.scene.camera = b_camera
        b_camera.matrix_world = b_obj.matrix_world
        if bird_view_degree is not None:
            b_camera.rotation_euler[0] = np.deg2rad(bird_view_degree)
        else:
            b_camera.rotation_euler[0] = np.deg2rad(self.blender_cfg.camera_bird_view_degree)

        # Rotate camera based on object front vector convention.
        # If front is +Y (e.g., SceneAgent), rotate camera 180° to look at the actual front.
        if self.scene_state is not None and self.scene_state.objectFrontVector is not None:
            if self.scene_state.objectFrontVector[1] > 0:  # Front is +Y
                b_camera.rotation_euler[2] += np.pi
        
        # # Need this because models imported from .glb files can have multiple meshes within one root b_obj
        obj_bounds = np.asarray(self.get_obj_bounds(b_obj.name))
        obj_dimensions = obj_bounds[1] - obj_bounds[0]
        
        # Hide all objects except the target object and the architectural elements
        if hide_others:
            self._set_elements_hide_render(b_objs=b_obj, reverse_selection=True, hide=True)
            self._set_elements_hide_render(b_archs=[], reverse_selection=True, hide=True)

        # Move the human model to left of the object
        if with_human_reference and self.b_human is not None:
            bpy.ops.object.select_all(action="DESELECT")
            self.b_human.select_set(True)
            self.b_human.hide_render = False
            self.b_human.matrix_world = b_obj.matrix_world
            bpy.ops.transform.translate(value=(-(obj_dimensions[0] * 0.5 + self.b_human.dimensions[0] * 0.5), 0, 0), orient_type="LOCAL")
        
        # --------------------------------------------

        # Select objects that should be in view
        bpy.ops.object.select_all(action="DESELECT")
        
        # Select the target object and its children
        b_obj.select_set(True)
        for b_obj_child in b_obj.children_recursive:
            b_obj_child.select_set(True)
        
        # Select the human model if needed
        if with_human_reference and self.b_human is not None:
            self.b_human.select_set(True)
            
        # Select objects that are close to the target object
        if zoom_out:
            
            # Get the objects close to the target object
            close_to_obj_ids = set([obj_id for obj_id, distance in self.obj_pairwise_distances[b_obj.name].items() if distance < 0.2])
            close_to_objs = [self.b_objs[close_to_obj_id] for close_to_obj_id in close_to_obj_ids]
            
            obj_centroid = np.mean(np.asarray(self.get_obj_bounds(b_obj.name)), axis=0)
            for close_to_obj in close_to_objs:
                
                # Do not include objects that to the side of the target object
                close_to_obj_centroid = np.mean(np.asarray(self.get_obj_bounds(close_to_obj.name)), axis=0)
                from_obj_to_close_to_obj = close_to_obj_centroid - obj_centroid
                from_obj_to_close_to_obj /= np.linalg.norm(from_obj_to_close_to_obj)
                UP_VECTOR = np.array([0, 0, 1])
                if np.dot(from_obj_to_close_to_obj, UP_VECTOR) < 0.25:
                    close_to_objs.remove(close_to_obj)
                    continue
                
                # Select the object and its children
                close_to_obj.select_set(True)
                for close_to_obj_child in close_to_obj.children_recursive:
                    close_to_obj_child.select_set(True)
        
        # Move the camera to view the selected objects
        bpy.ops.view3d.camera_to_view_selected()
        
        # Back off the camera along its local z-axis to include the surroundings
        if zoom_out:
            bpy.ops.object.select_all(action="DESELECT")
            b_camera.select_set(True)
            bpy.ops.transform.translate(value=(0, 0, np.linalg.norm(obj_dimensions) * 0.25), orient_type="LOCAL")
        
        # Find things blocking the camera's line of sight and hide them
        if not hide_others:
            blocking_objs, blocking_archs = self._get_elements_blocking_camera_line_of_sight(b_camera, b_obj)
            if zoom_out:
                # Do not hide closeby objects that are supposed to be in view
                blocking_objs = [b_obj for b_obj in blocking_objs if b_obj not in close_to_objs]
            self._set_elements_hide_render(b_objs=blocking_objs, b_archs=blocking_archs, hide=True)
        
        # --------------------------------------------
        
        # Render the object
        if relative_file_path is None:
            relative_file_path = pathlib.Path(self.blender_cfg.object_render_subdir) / f"{b_obj.name}.{self.blender_cfg.render_file_format.lower()}"
        self.render(b_camera, relative_file_path)

        # --------------------------------------------

        # Delete the temporary camera
        self._remove_camera("camera_temp")
        
        # Unhide the previously hidden objects
        if hide_others:
            self._set_elements_hide_render(b_objs=b_obj, reverse_selection=True, hide=False)
            self._set_elements_hide_render(b_archs=[], reverse_selection=True, hide=False)
        else:
            self._set_elements_hide_render(b_objs=blocking_objs, b_archs=blocking_archs, hide=False)
        
        # Hide the human model
        if with_human_reference and self.b_human is not None:
            self.b_human.hide_render = True
    
    def render_scene_from_top(self, relative_file_path: pathlib.Path = pathlib.Path("scene.png")) -> None:
        """
        Render the scene from the top view.
        
        Args:
            relative_file_path: the path to save the rendered image relative to the output directory
        """
        
        if "camera_top" not in self.b_cameras:
            self.add_top_camera()
        
        self.render(self.b_cameras["camera_top"], relative_file_path)
        
    def render_all_objs_front_solo(self) -> None:
        """
        Render the front view of each of the objects without any surroundings.
        """

        relative_dir = pathlib.Path(self.blender_cfg.object_render_subdir)
        for obj_id, b_obj in self.b_objs.items():
            relative_file_path = relative_dir / f"{obj_id}.{self.blender_cfg.render_file_format.lower()}"
            self.render_one_obj(b_obj, relative_file_path, hide_others=True, zoom_out=False, with_human_reference=False)

    def render_all_objs_front_size_reference(self) -> None:
        """
        Render the front view of each of the objects with a human model for size comparison.
        """

        relative_dir = pathlib.Path(self.blender_cfg.object_render_subdir)
        for obj_id, b_obj in self.b_objs.items():
            relative_file_path = relative_dir / f"size_{obj_id}.{self.blender_cfg.render_file_format.lower()}"
            self.render_one_obj(b_obj, relative_file_path, hide_others=True, zoom_out=False, with_human_reference=True, bird_view_degree=80)

    def render_all_objs_front_surroundings(self) -> None:
        """
        Render the front view of each of the objects with surroundings.
        """

        relative_dir = pathlib.Path(self.blender_cfg.object_render_subdir)
        for obj_id, b_obj in self.b_objs.items():
            relative_file_path = relative_dir / f"surroundings_{obj_id}.{self.blender_cfg.render_file_format.lower()}"
            self.render_one_obj(b_obj, relative_file_path, hide_others=False, zoom_out=True, with_human_reference=False)
    
    def render_all_objs_global_top(self) -> None:
        """
        Render the global top view of each of the objects.
        """

        relative_dir = pathlib.Path(self.blender_cfg.object_render_subdir)
        for obj_id, b_obj in self.b_objs.items():
            relative_file_path = relative_dir / f"top_{obj_id}.{self.blender_cfg.render_file_format.lower()}"
            self._set_elements_hide_render(b_objs=b_obj, reverse_selection=True, hide=True)
            self.render_scene_from_top(relative_file_path)
            self._set_elements_hide_render(b_objs=b_obj, reverse_selection=True, hide=False)
    
    def render_selected_objs_global_top(self, selected_obj_ids: list[str], relative_file_path) -> None:
        """
        Render a top view image with only the selected objects.

        Args:
            obj_ids: the IDs of the objects to render
        """

        selected_objs = [self.b_objs[obj_id] for obj_id in selected_obj_ids]
        self._set_elements_hide_render(b_objs=selected_objs, reverse_selection=True, hide=True)
        self.render_scene_from_top(relative_file_path)
        self._set_elements_hide_render(b_objs=selected_objs, reverse_selection=True, hide=False)

    def save_blend(self, filename: str = None) -> None:
        """
        Save the scene as a Blender (.blend) file.

        Args:
            filename: the output filename
        """

        # Set up output path
        if filename is None:
            filename = self.blender_cfg.default_blend_filename
        
        # Hide the human model
        if self.b_human is not None:
            self.b_human.hide_set(True)
        
        # Save the scene as a Blender file
        bpy.ops.wm.save_as_mainfile(filepath=str(self.output_dir / filename))
        
        # Unhide the human model
        if self.b_human is not None:
            self.b_human.hide_set(False)

    def save_glb(self, filename: str = None) -> None:
        """
        Save the scene as a glTF (.glb) file.

        Args:
            filename: the output filename
        """

        # Set up output path
        if filename is None:
            filename = self.blender_cfg.default_glb_filename
        
        # Export the scene as a glTF file
        bpy.ops.export_scene.gltf(filepath=str(self.output_dir / filename), export_format="GLB")

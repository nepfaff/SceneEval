import pathlib
import bpy
import trimesh
import numpy as np
from mathutils import Matrix
from assets import Retriever
from .scene_state import SceneState
from .config import SceneConfig
from .blender_scene import BlenderScene, BlenderConfig
from .trimesh_scene import TrimeshScene, TrimeshConfig

# ----------------------------------------------------------------------------------------

class Scene:
    def __init__(self,
                 mesh_retriever: Retriever,
                 scene_state: SceneState,
                 scene_cfg: SceneConfig,
                 blender_cfg: BlenderConfig,
                 trimesh_cfg: TrimeshConfig,
                 output_dir: pathlib.Path) -> None:
        """
        Initialize a Scene object that contains both Blender and Trimesh scenes.

        Args:
            mesh_retriever: the mesh retriever to use for getting object files
            scene_state: the scene state to load
            scene_cfg: the scene configuration
            blender_cfg: the Blender configuration
            trimesh_cfg: the Trimesh configuration
            output_dir: the output directory for this scene
        """

        self.output_dir = output_dir
        self.blender_scene = BlenderScene(mesh_retriever, scene_state, scene_cfg, blender_cfg, output_dir)
        self.trimesh_scene = TrimeshScene(mesh_retriever, scene_state, scene_cfg, trimesh_cfg)
    
    @property
    def obj_descriptions(self) -> dict[str, str]:
        """
        Returns:
            The descriptions of the objects in the scene
        """

        return self.blender_scene.obj_descriptions

    @property
    def inverse_obj_descriptions(self) -> dict[str, str]:
        """
        Returns:
            The inverse mapping of object descriptions to object IDs
        """

        return self.blender_scene.inverse_obj_descriptions
    
    @property
    def b_objs(self) -> dict[str, bpy.types.Object]:
        """
        Returns:
            The objects in the Blender scene
        """

        return self.blender_scene.b_objs
    
    @property
    def b_architecture(self) -> dict[str, bpy.types.Object]:
        """
        Returns:
            The architecture objects in the Blender scene
        """

        return self.blender_scene.b_architecture
    
    @property
    def t_objs(self) -> dict[str, trimesh.Trimesh]:
        """
        Returns:
            The objects in the Trimesh scene
        """

        return self.trimesh_scene.t_objs

    @property
    def t_architecture(self) -> dict[str, trimesh.Trimesh]:
        """
        Returns:
            The architecture objects in the Trimesh scene
        """

        return self.trimesh_scene.t_architecture

    @property
    def carpet_obj_ids(self) -> set[str]:
        """
        Returns:
            Set of object IDs that are carpets/rugs (excluded from collision checks)
        """

        return self.trimesh_scene.carpet_obj_ids

    def is_carpet(self, obj_id: str) -> bool:
        """
        Check if an object is a carpet/rug.

        Args:
            obj_id: the object ID

        Returns:
            True if the object is a carpet/rug
        """

        return obj_id in self.carpet_obj_ids

    def get_obj_ids(self) -> list[str]:
        """
        Get the object IDs in the scene.

        Returns:
            obj_ids: the object IDs in the scene
        """

        return list(self.blender_scene.b_objs.keys())
    
    def get_arch_ids(self) -> list[str]:
        """
        Get the architecture element IDs in the scene.

        Uses Trimesh architecture IDs for consistent naming with metrics.
        This ensures names like "floor_kitchen", "wall_1" match the filtering
        logic in metrics (e.g., arch_id.startswith("floor")).

        Returns:
            arch_ids: the architecture element IDs in the scene
        """

        return list(self.trimesh_scene.t_architecture.keys())
    
    def get_obj_matrix(self, obj_id: str) -> Matrix:
        """
        Get the 4x4 transformation matrix of the object.

        Args:
            obj_id: the object ID

        Returns:
            matrix: The 4x4 transformation matrix of the object
        """

        return self.blender_scene.b_objs[obj_id].matrix_world
    
    def get_arch_matrix(self, arch_id: str) -> Matrix:
        """
        Get the 4x4 transformation matrix of the architecture element.

        Args:
            arch_id: the architecture element ID

        Returns:
            matrix: The 4x4 transformation matrix of the architecture element
        """

        # Check if arch exists in Blender scene
        if arch_id in self.blender_scene.b_architecture:
            return self.blender_scene.b_architecture[arch_id].matrix_world
        # For Trimesh-only arch (procedural), return identity matrix
        # These meshes are already in world coordinates
        return Matrix.Identity(4)

    def get_arch_orientation(self, arch_id: str) -> Matrix:
        """
        Get the 3x3 orientation matrix of an architectural element.

        For walls, this represents the rotation that transforms the default front
        vector [0, -1, 0] to the wall's actual front direction (pointing into the room).

        This is separate from get_arch_matrix() because:
        - get_arch_matrix() returns the full LOCAL→WORLD transform (for position calculations)
        - get_arch_orientation() returns just the rotation (for orientation-dependent calculations)

        For Trimesh-only elements, the mesh is already in world coordinates, so
        get_arch_matrix() returns Identity. But the orientation still needs to be
        computed from the mesh geometry.

        Args:
            arch_id: the architecture element ID

        Returns:
            matrix: The 3x3 orientation matrix of the architecture element
        """
        # For Blender elements, extract rotation from world matrix
        if arch_id in self.blender_scene.b_architecture:
            return self.blender_scene.b_architecture[arch_id].matrix_world.to_3x3()

        # For Trimesh-only walls, compute orientation from mesh geometry
        if arch_id.startswith("wall"):
            return self._compute_wall_orientation(arch_id)

        return Matrix.Identity(3)

    def _compute_wall_orientation(self, wall_id: str) -> Matrix:
        """
        Compute wall orientation from mesh bounds and position relative to floor.

        For multi-room scenes, parses room name from wall ID to find the correct floor.
        Falls back to geometric proximity if room name cannot be extracted.

        Holodeck wall ID format: wall_wall|<room>|<direction>|<idx>[|exterior]
        Example: wall_wall|bedroom|west|0 or wall_wall|bedroom|west|0|exterior

        The orientation determines which direction is "front" (pointing into the room
        for interior walls, or out of the room for exterior walls).

        Args:
            wall_id: the wall ID in t_architecture

        Returns:
            matrix: 3x3 rotation matrix that transforms [0, -1, 0] to wall's front direction
        """
        wall_mesh = self.t_architecture[wall_id]
        wall_bounds = wall_mesh.bounds
        wall_center = (wall_bounds[0] + wall_bounds[1]) / 2
        wall_extents = wall_bounds[1] - wall_bounds[0]

        # Check if this is an exterior wall (front should point OUT of room)
        is_exterior = "exterior" in wall_id.lower()

        # Find the floor this wall belongs to
        floor_keys = [k for k in self.t_architecture.keys() if k.startswith("floor")]
        if not floor_keys:
            return Matrix.Identity(3)

        best_floor = None

        # Strategy 1: Parse room name from wall ID (Holodeck convention)
        # Wall ID format: wall_wall|<room>|<direction>|<idx>[|exterior]
        # or: wall_<numeric_id> (IDesign)
        room_name = self._extract_room_from_wall_id(wall_id)
        if room_name:
            # Look for matching floor
            target_floor_key = f"floor_{room_name}"
            if target_floor_key in self.t_architecture:
                best_floor = self.t_architecture[target_floor_key]

        # Strategy 2: Geometric proximity (fallback)
        if best_floor is None:
            best_distance = float('inf')
            for floor_key in floor_keys:
                floor_mesh = self.t_architecture[floor_key]
                floor_bounds = floor_mesh.bounds
                floor_center_2d = (floor_bounds[0][:2] + floor_bounds[1][:2]) / 2

                # Check if wall overlaps with floor in XY plane
                x_overlap = (wall_bounds[0][0] <= floor_bounds[1][0]) and (wall_bounds[1][0] >= floor_bounds[0][0])
                y_overlap = (wall_bounds[0][1] <= floor_bounds[1][1]) and (wall_bounds[1][1] >= floor_bounds[0][1])

                if x_overlap and y_overlap:
                    # For overlapping floors, prefer the one where wall is on boundary
                    # (wall center is near floor min/max in one dimension)
                    wall_on_boundary = (
                        np.isclose(wall_center[0], floor_bounds[0][0], atol=0.1) or
                        np.isclose(wall_center[0], floor_bounds[1][0], atol=0.1) or
                        np.isclose(wall_center[1], floor_bounds[0][1], atol=0.1) or
                        np.isclose(wall_center[1], floor_bounds[1][1], atol=0.1)
                    )
                    if wall_on_boundary:
                        best_floor = floor_mesh
                        break
                    elif best_floor is None:
                        best_floor = floor_mesh

                # Fallback: use closest floor by center distance
                dist = np.linalg.norm(wall_center[:2] - floor_center_2d)
                if dist < best_distance:
                    best_distance = dist
                    if best_floor is None:
                        best_floor = floor_mesh

        if best_floor is None:
            return Matrix.Identity(3)

        floor_center = (best_floor.bounds[0] + best_floor.bounds[1]) / 2

        # Determine rotation angle based on wall alignment and position
        # Goal: transform front_vector [0,-1,0] to point INTO the room
        if wall_extents[0] > wall_extents[1]:  # X-aligned wall (runs along X axis)
            if wall_center[1] < floor_center[1]:  # South wall - front should be +Y
                angle = np.pi  # 180° rotation
            else:  # North wall - front should be -Y
                angle = 0  # No rotation needed
        else:  # Y-aligned wall (runs along Y axis)
            if wall_center[0] < floor_center[0]:  # West wall - front should be +X
                angle = np.pi / 2  # 90° rotation
            else:  # East wall - front should be -X
                angle = -np.pi / 2  # -90° rotation

        # For exterior walls, flip the direction (front points OUT of room)
        if is_exterior:
            angle += np.pi

        return Matrix.Rotation(angle, 3, 'Z')

    def _extract_room_from_wall_id(self, wall_id: str) -> str | None:
        """
        Extract room name from wall ID.

        Holodeck format: wall_wall|<room>|<direction>|<idx>[|exterior]
        Example: wall_wall|bedroom|west|0 -> "bedroom"

        Args:
            wall_id: the wall ID

        Returns:
            room_name: the room name, or None if cannot be extracted
        """
        # Try Holodeck format: wall_wall|<room>|<direction>|<idx>
        if wall_id.startswith("wall_wall|"):
            parts = wall_id[len("wall_"):].split("|")
            if len(parts) >= 2:
                return parts[1]  # parts[0] is "wall", parts[1] is room name

        # Try simpler format: wall_<room>_<direction>_<idx>
        if wall_id.startswith("wall_") and "_" in wall_id[5:]:
            parts = wall_id[5:].split("_")
            if len(parts) >= 2:
                # Check if first part looks like a room name (not a number)
                if not parts[0].isdigit():
                    return parts[0]

        return None

    def get_obj_z_rotation(self, obj_id: str) -> float:
        """
        Get the object's rotation around the z-axis (up axis).

        Args:
            obj_id: the object ID

        Returns:
            z_rotation: the object's rotation around the z-axis in radians
        """

        b_obj = self.b_objs[obj_id]
        b_obj.rotation_mode = "XYZ"
        return b_obj.rotation_euler.z

    def get_front_vector(self) -> np.ndarray:
        """
        Get the front vector for objects in this scene.

        The front vector defines the canonical front direction for objects in their
        default pose. This may vary by scene generation method (e.g., SceneAgent uses
        +Y as front, while LayoutVLM uses -Y).

        Returns:
            front_vector: the front direction vector [x, y, z]
        """

        scene_state = self.blender_scene.scene_state
        if scene_state and scene_state.objectFrontVector:
            return np.array(scene_state.objectFrontVector)
        return np.array([0, -1, 0])  # Default: -Y is front
    
    def get_default_pose_t_obj(self, obj_id: str) -> trimesh.Trimesh:
        """
        Get the object in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            new_t_obj: a copy of the object in its default pose
        """

        new_t_obj = self.t_objs[obj_id].copy()
        obj_world_matrix = self.get_obj_matrix(obj_id)
        obj_to_default_pose_matrix = obj_world_matrix.inverted()
        new_t_obj.apply_transform(obj_to_default_pose_matrix)
        
        return new_t_obj
    
    def get_default_pose_t_arch(self, arch_id: str) -> trimesh.Trimesh:
        """
        Get the architecture element in its default pose.

        Args:
            arch_id: the architecture element ID
        """

        new_t_arch = self.t_architecture[arch_id].copy()
        # Only apply inverse transform if arch exists in Blender
        # For Trimesh-only arch (procedural), mesh is already in default pose
        if arch_id in self.blender_scene.b_architecture:
            arch_world_matrix = self.get_arch_matrix(arch_id)
            arch_to_default_pose_matrix = arch_world_matrix.inverted()
            new_t_arch.apply_transform(arch_to_default_pose_matrix)

        return new_t_arch
    
    def get_default_pose_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """
        Get the center of the object's bounding box in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            center: the center of the object's bounding box in its default pose
        """

        t_obj_default_pose = self.get_default_pose_t_obj(obj_id)
        center = t_obj_default_pose.bounds[0] + t_obj_default_pose.extents / 2
        return center
    
    def get_default_pose_arch_bbox_center(self, arch_id: str) -> np.ndarray:
        """
        Get the center of the architecture element's bounding box in its default pose.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            center: the center of the architecture element's bounding box in its default pose
        """
            
        t_arch_default_pose = self.get_default_pose_t_arch(arch_id)
        center = t_arch_default_pose.bounds[0] + t_arch_default_pose.extents / 2
        return center
    
    def get_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """
        Get the center of the object's bounding box.

        Args:
            obj_id: the object ID
        
        Returns:
            center: the center of the object's bounding box
        """

        default_pose_obj_center = self.get_default_pose_obj_bbox_center(obj_id)
        obj_matrix = self.get_obj_matrix(obj_id)
        center = (np.asarray(obj_matrix) @ np.append(default_pose_obj_center, 1))[:3]
        return center
    
    def get_arch_bbox_center(self, arch_id: str) -> np.ndarray:
        """
        Get the center of the architecture element's bounding box.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            center: the center of the architecture element's bounding box
        """
            
        default_pose_arch_center = self.get_default_pose_arch_bbox_center(arch_id)
        arch_matrix = self.get_arch_matrix(arch_id)
        center = (np.asarray(arch_matrix) @ np.append(default_pose_arch_center, 1))[:3]
        return center

    def get_default_pose_obj_bbox_extents(self, obj_id: str) -> np.ndarray:
        """
        Get the extents of the object's bounding box in its default pose.

        Args:
            obj_id: the object ID
        
        Returns:
            extents: the extents of the object's bounding box in its default pose
        """
            
        t_obj_default_pose = self.get_default_pose_t_obj(obj_id)
        return t_obj_default_pose.extents
    
    def get_default_pose_arch_bbox_extents(self, arch_id: str) -> np.ndarray:
        """
        Get the extents of the architecture element's bounding box in its default pose.

        Args:
            arch_id: the architecture element ID
        
        Returns:
            extents: the extents of the architecture element's bounding box in its default pose
        """
        
        t_arch_default_pose = self.get_default_pose_t_arch(arch_id)
        return t_arch_default_pose.extents
    
    def get_obj_render_path(self, obj_id: str, view: str = "FRONT") -> pathlib.Path:
        """
        Get the path to the rendered image of the object.

        Args:
            obj_id: the object ID
            view: the view to render from (FRONT, SIZE_REFERENCE, SURROUNDINGS, TOP)
        
        Returns:
            render_path: the path to the rendered image of the object
        """
        
        assert view in ["FRONT", "SIZE_REFERENCE", "SURROUNDINGS", "TOP"], "View must be one of FRONT, SIZE_REF, SURROUNDINGS, TOP."
        obj_render_subdir = self.blender_scene.blender_cfg.object_render_subdir
        image_format = self.blender_scene.blender_cfg.render_file_format.lower()
        
        file_name = f"{obj_id}.{image_format}"
        
        match view:
            case "FRONT":
                pass
            case "SIZE_REFERENCE":
                file_name = "size_" + file_name
            case "SURROUNDINGS":
                file_name = "surroundings_" + file_name
            case "TOP":
                file_name = "top_" + file_name
        
        render_path = self.output_dir / obj_render_subdir / file_name
        
        return render_path
    
    def show(self) -> None:
        """
        Show the scene in Trimesh.
        """

        self.trimesh_scene.show()

    def export_trimesh(self, file_path: str) -> None:
        """
        Export the trimesh scene to a file (e.g., GLB for debugging).

        Args:
            file_path: the path to export to (e.g., "debug_trimesh.glb")
        """

        self.trimesh_scene.export(file_path)
        
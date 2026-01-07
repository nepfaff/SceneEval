"""Utilities for wall detection and visibility management in Blender scenes.

Used for dollhouse-style room renders where walls between camera and scene
center are hidden to reveal interior.
"""

import logging
import bpy
import numpy as np
from mathutils import Vector

logger = logging.getLogger(__name__)


def looks_like_wall(obj: bpy.types.Object) -> bool:
    """Detect if a Blender object looks like a wall using geometric heuristics.

    Walls are tall (vertical extent), thin in one horizontal dimension, and have
    high aspect ratio. Uses world-space bounding box to handle parent transforms.

    Args:
        obj: Blender object to check.

    Returns:
        True if object appears to be a wall based on dimensions and orientation.
    """
    if obj.type != "MESH":
        return False

    # Exclude special objects (including doors and windows)
    excluded_names = ["MetricOverlay", "Annotation", "floor", "ceiling", "Floor", "Ceiling",
                      "door", "Door", "window", "Window", "Door_", "Window_", "Doorway"]
    if any(name in obj.name for name in excluded_names):
        return False

    # Get world-space bounding box by transforming all 8 corners
    try:
        bbox_local = np.array(obj.bound_box)
        bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])
    except Exception:
        return False

    # Find world-space dimensions
    min_coords = bbox_world.min(axis=0)
    max_coords = bbox_world.max(axis=0)
    world_dimensions = max_coords - min_coords

    width_x = world_dimensions[0]
    depth_y = world_dimensions[1]
    height_z = world_dimensions[2]

    # Walls should be tall (significant vertical extent)
    MIN_WALL_HEIGHT = 1.5  # At least 1.5m tall
    if height_z < MIN_WALL_HEIGHT:
        return False

    # Walls should be thin in one horizontal dimension
    MAX_THICKNESS = 0.2  # Max 20cm thick
    horizontal_dims = [width_x, depth_y]
    min_horizontal = min(horizontal_dims)

    if min_horizontal > MAX_THICKNESS:
        return False

    # Check aspect ratio (height vs thin dimension)
    if min_horizontal < 0.001:  # Avoid division by zero
        return True
    aspect_ratio = height_z / min_horizontal
    MIN_ASPECT_RATIO = 5  # Height should be at least 5x the thickness

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    return True


def compute_wall_normal(obj: bpy.types.Object, scene_centroid: Vector) -> Vector:
    """Compute inward-pointing wall normal (toward scene center).

    For thin walls, the normal is perpendicular to the thinnest dimension
    and points toward the scene center.

    Args:
        obj: Wall object.
        scene_centroid: Center point of the scene in world coordinates.

    Returns:
        Normalized vector pointing from wall toward scene center (in XY plane).
    """
    # Get world-space bounding box
    bbox_local = np.array(obj.bound_box)
    bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])

    # Find dimensions
    min_coords = bbox_world.min(axis=0)
    max_coords = bbox_world.max(axis=0)
    world_dimensions = max_coords - min_coords
    wall_center = Vector((min_coords + max_coords) / 2)

    width_x = world_dimensions[0]
    depth_y = world_dimensions[1]

    # Determine wall orientation based on thin dimension
    if width_x < depth_y:
        # Wall is thin in X, extends in Y → normal is ±X
        candidate_normal = Vector((1.0, 0.0, 0.0))
    else:
        # Wall is thin in Y, extends in X → normal is ±Y
        candidate_normal = Vector((0.0, 1.0, 0.0))

    # Make normal point toward scene center
    to_center = scene_centroid - wall_center
    to_center.z = 0  # Keep in XY plane

    if candidate_normal.dot(to_center) < 0:
        candidate_normal = -candidate_normal

    return candidate_normal.normalized()


def should_hide_wall(
    obj: bpy.types.Object,
    camera_direction: Vector,
    is_top_view: bool,
    scene_centroid: Vector,
) -> bool:
    """Determine if wall should be hidden based on camera viewpoint.

    Walls between camera and room center should be hidden for dollhouse effect.

    Args:
        obj: Wall object to check.
        camera_direction: Direction camera is pointing (normalized, in XY plane).
        is_top_view: True if this is a top-down view.
        scene_centroid: Center point of the scene.

    Returns:
        True if wall should be hidden.
    """
    # Top view: show all walls (don't hide any)
    if is_top_view:
        return False

    # Side view: hide walls that block camera view into room
    wall_normal = compute_wall_normal(obj, scene_centroid)

    # Wall normal points from wall toward room center (inward)
    # Camera direction points from camera toward scene center
    # If they point in SAME direction (positive dot): wall is BETWEEN camera and room
    # If they point in OPPOSITE directions (negative dot): wall is on FAR side
    camera_dir_xy = Vector((camera_direction.x, camera_direction.y, 0.0)).normalized()
    dot_product = camera_dir_xy.dot(wall_normal)

    # Hide if wall is between camera and room (same direction, positive dot)
    return dot_product > 0.1


def get_all_walls() -> list[bpy.types.Object]:
    """Get all wall objects in the current scene.

    Returns:
        List of Blender objects that appear to be walls.
    """
    all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    return [obj for obj in all_meshes if looks_like_wall(obj)]


def compute_scene_centroid() -> Vector:
    """Compute the centroid of all mesh objects in the scene.

    Returns:
        Center point of all objects in world coordinates.
    """
    all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not all_meshes:
        return Vector((0, 0, 0))

    centers = []
    for obj in all_meshes:
        bbox_local = np.array(obj.bound_box)
        bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])
        center = (bbox_world.min(axis=0) + bbox_world.max(axis=0)) / 2
        centers.append(center)

    centroid = np.mean(centers, axis=0)
    return Vector(centroid)


def hide_walls_for_view(
    camera_position: Vector,
    scene_centroid: Vector,
    is_top_view: bool = False,
) -> list[bpy.types.Object]:
    """Hide walls that block the camera view.

    Args:
        camera_position: Camera position in world coordinates.
        scene_centroid: Center of the scene.
        is_top_view: True if rendering top-down view.

    Returns:
        List of wall objects that were hidden (for later restoration).
    """
    # Camera direction: from camera toward scene center
    camera_direction = (scene_centroid - camera_position).normalized()

    walls = get_all_walls()
    hidden_walls = []

    for wall in walls:
        if should_hide_wall(wall, camera_direction, is_top_view, scene_centroid):
            wall.hide_render = True
            wall.hide_viewport = True
            hidden_walls.append(wall)

    bpy.context.view_layer.update()
    logger.debug(f"Hidden {len(hidden_walls)} walls for camera view")
    return hidden_walls


def restore_all_walls() -> None:
    """Restore all walls to visible state.

    Call before rendering each new view to reset wall visibility.
    """
    walls = get_all_walls()
    for wall in walls:
        wall.hide_render = False
        wall.hide_viewport = False

    bpy.context.view_layer.update()


def _create_one_sided_wall_material(
    name: str = "TransparentWall",
    color: tuple = (0.6, 0.45, 0.2, 1.0),
) -> "bpy.types.Material":
    """Create a one-sided wall material (front opaque, back transparent).

    This creates the dollhouse effect where walls are visible from inside
    the room but transparent when viewed from outside.

    Args:
        name: Material name.
        color: RGBA color for the wall (front face).

    Returns:
        Blender material with one-sided transparency shader.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create shader nodes
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix_shader = nodes.new("ShaderNodeMixShader")
    geometry = nodes.new("ShaderNodeNewGeometry")
    output = nodes.new("ShaderNodeOutputMaterial")

    # Position nodes
    geometry.location = (-600, 0)
    bsdf.location = (-400, 200)
    transparent.location = (-400, -200)
    mix_shader.location = (-200, 0)
    output.location = (0, 0)

    # Link nodes: backfacing controls mix between opaque and transparent
    links.new(geometry.outputs["Backfacing"], mix_shader.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])  # Front face (opaque)
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])  # Back face (transparent)
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    # Configure material for one-sided transparency
    mat.use_backface_culling = True
    mat.blend_method = "CLIP"
    mat.alpha_threshold = 0.5

    # Set front face color
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Alpha"].default_value = 1.0

    return mat


def get_floor_bounds() -> tuple[Vector, Vector] | None:
    """Get the combined bounding box of all floor objects.

    Returns:
        Tuple of (min_coords, max_coords) as Vectors, or None if no floors found.
    """
    floor_objects = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and ("floor" in obj.name.lower() or "Floor" in obj.name)
    ]

    if not floor_objects:
        return None

    all_corners = []
    for obj in floor_objects:
        bbox_local = np.array(obj.bound_box)
        bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])
        all_corners.extend(bbox_world)

    all_corners = np.array(all_corners)
    min_coords = all_corners.min(axis=0)
    max_coords = all_corners.max(axis=0)

    return Vector(min_coords), Vector(max_coords)


def add_transparent_walls(
    wall_height: float = 2.8,
    wall_color: tuple = (0.6, 0.45, 0.2, 1.0),
) -> list[bpy.types.Object]:
    """Add 4 transparent walls around the floor bounding box.

    Creates dollhouse-style walls that are visible from inside but transparent
    from outside. Used for SceneWeaver/IDesign scenes that lack walls.

    Args:
        wall_height: Height of walls in meters (default 2.8m).
        wall_color: RGBA color for walls.

    Returns:
        List of created wall objects.
    """
    import math

    bounds = get_floor_bounds()
    if bounds is None:
        logger.warning("No floor objects found, cannot add transparent walls")
        return []

    min_coords, max_coords = bounds
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords

    # Wall center height (half wall height above floor)
    wall_center_z = min_z + wall_height / 2

    # Create one-sided material
    wall_material = _create_one_sided_wall_material(
        name="TransparentWallMaterial",
        color=wall_color,
    )

    created_walls = []

    # Wall definitions: (name, location, rotation_z_degrees, scale_x)
    wall_specs = [
        # Left wall (min Y side)
        ("transparent_wall_left",
         ((max_x + min_x) / 2, min_y, wall_center_z),
         180,  # Face into room
         max_x - min_x),
        # Right wall (max Y side)
        ("transparent_wall_right",
         ((max_x + min_x) / 2, max_y, wall_center_z),
         0,  # Face into room
         max_x - min_x),
        # Far wall (min X side)
        ("transparent_wall_far",
         (min_x, (max_y + min_y) / 2, wall_center_z),
         90,  # Face into room
         max_y - min_y),
        # Close wall (max X side)
        ("transparent_wall_close",
         (max_x, (max_y + min_y) / 2, wall_center_z),
         270,  # Face into room
         max_y - min_y),
    ]

    for name, location, rot_z_deg, width in wall_specs:
        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=1, location=location)
        wall = bpy.context.active_object
        wall.name = name
        wall.data.name = f"{name}_Mesh"

        # Rotate to vertical and face correct direction
        wall.rotation_euler = (
            math.radians(90),  # Rotate to vertical
            0,
            math.radians(rot_z_deg),
        )

        # Scale to wall dimensions
        wall.scale = (width, wall_height, 1)

        # Apply material
        wall.data.materials.append(wall_material)
        wall.visible_shadow = False

        # Apply transforms
        bpy.ops.object.select_all(action="DESELECT")
        wall.select_set(True)
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        created_walls.append(wall)
        logger.debug(f"Created transparent wall: {name}")

    logger.info(f"Added {len(created_walls)} transparent walls around floor bounds")
    return created_walls

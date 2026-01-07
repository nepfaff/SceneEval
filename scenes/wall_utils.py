"""Utilities for wall detection and visibility management in Blender scenes.

Used for dollhouse-style room renders where walls between camera and scene
center are hidden to reveal interior.
"""

import json
import logging
import pathlib
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


def _create_wall_boxes(wall_info: dict) -> dict[str, bpy.types.Object]:
    """Create temporary box meshes from wall bounding boxes for raycasting.

    This creates solid boxes that fill any openings (doors, windows) so rays
    can't pass through them.

    Args:
        wall_info: Dict mapping wall_name -> (center, thin_axis, thin_pos, min_c, max_c)

    Returns:
        Dict mapping wall_name -> temp_box_object
    """
    import bmesh

    temp_boxes = {}
    for wall_name, (center, thin_axis, thin_pos, min_c, max_c) in wall_info.items():
        # Create box mesh
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)

        # Scale and position to match wall bounding box
        dims = max_c - min_c
        for v in bm.verts:
            v.co.x = v.co.x * dims[0] + center[0]
            v.co.y = v.co.y * dims[1] + center[1]
            v.co.z = v.co.z * dims[2] + center[2]

        # Create Blender object
        mesh = bpy.data.meshes.new(f"_temp_wall_box_{wall_name}")
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(f"_temp_wall_box_{wall_name}", mesh)
        obj["_original_wall"] = wall_name  # Store reference to original wall
        bpy.context.scene.collection.objects.link(obj)
        temp_boxes[wall_name] = obj

    bpy.context.view_layer.update()
    return temp_boxes


def _remove_wall_boxes(temp_boxes: dict[str, bpy.types.Object]) -> None:
    """Remove temporary box meshes created for raycasting."""
    for obj in temp_boxes.values():
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)


def hide_walls_for_view(
    camera_position: Vector,
    scene_centroid: Vector,
    is_top_view: bool = False,
) -> list[bpy.types.Object]:
    """Hide walls blocking camera view using 360° raycasting at multiple heights.

    Shoots rays from camera XY position at 3 different heights (low, mid, high)
    to handle windows and other openings. Any wall hit by these rays is blocking
    the view and gets hidden.

    Args:
        camera_position: Camera position in world coordinates.
        scene_centroid: Center of the scene (unused, kept for API compat).
        is_top_view: True if rendering top-down view.

    Returns:
        List of wall objects that were hidden (for later restoration).
    """
    import math

    # Top view: show all walls
    if is_top_view:
        return []

    # Determine wall heights from actual walls in scene
    walls = get_all_walls()
    if walls:
        wall_heights = []
        for wall in walls:
            try:
                bbox = np.array([wall.matrix_world @ Vector(c) for c in wall.bound_box])
                height = bbox[:, 2].max() - bbox[:, 2].min()
                wall_heights.append(height)
            except Exception:
                pass
        if wall_heights:
            max_wall_height = max(wall_heights)
        else:
            max_wall_height = 2.8  # Default
    else:
        max_wall_height = 2.8  # Default

    # Cast rays at 3 heights: 10%, 50%, 90% of wall height (handles windows)
    RAY_HEIGHTS = [
        max_wall_height * 0.1,  # Low (below windows)
        max_wall_height * 0.5,  # Mid
        max_wall_height * 0.9,  # High (above windows)
    ]

    # Build wall info first (needed for flat-face detection and adjacency)
    all_walls = get_all_walls()
    wall_info = {}  # name -> (center, thin_axis, thin_pos, min_c, max_c)
    for wall in all_walls:
        try:
            bbox = np.array([wall.matrix_world @ Vector(c) for c in wall.bound_box])
            min_c = bbox.min(axis=0)
            max_c = bbox.max(axis=0)
            dims = max_c - min_c
            center = (min_c + max_c) / 2

            # Find thin axis (0=X, 1=Y)
            if dims[0] < dims[1]:
                thin_axis = 0
                thin_pos = center[0]
            else:
                thin_axis = 1
                thin_pos = center[1]

            wall_info[wall.name] = (center, thin_axis, thin_pos, min_c, max_c)
        except Exception:
            pass

    # Create temporary box meshes for raycasting (blocks doors/windows)
    temp_boxes = _create_wall_boxes(wall_info)

    # Shoot rays in 360 degrees at 1° resolution per height
    # Offset angles at each height for better coverage (effectively 0.33° resolution)
    NUM_RAYS = 360
    walls_to_hide = set()

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for height_idx, height in enumerate(RAY_HEIGHTS):
        ray_origin = Vector((camera_position.x, camera_position.y, height))
        # Offset angle by 1/3 degree for each height level
        angle_offset = height_idx / len(RAY_HEIGHTS)

        for i in range(NUM_RAYS):
            angle = math.radians(i + angle_offset)
            direction = Vector((math.cos(angle), math.sin(angle), 0))

            # Cast ray against temp boxes (solid, no door/window openings)
            result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
                depsgraph, ray_origin, direction
            )

            # Check if we hit a temp wall box OR an original wall
            # (Original walls still exist, so rays may hit them instead of temp boxes)
            if result and obj is not None:
                # Determine which wall was hit
                if obj.name.startswith("_temp_wall_box_"):
                    wall_name = obj.get("_original_wall")
                elif obj.name in wall_info:
                    wall_name = obj.name
                else:
                    wall_name = None

                if wall_name and wall_name in wall_info:
                    _, thin_axis, _, min_c, max_c = wall_info[wall_name]
                    hit_pos = np.array(location)

                    # Check if hit is on flat face (hide) or thin edge (don't hide)
                    # For wall thin in X: flat faces at x_min/x_max, thin edges at y_min/y_max
                    # For wall thin in Y: flat faces at y_min/y_max, thin edges at x_min/x_max
                    FACE_TOLERANCE = 0.15

                    hit_thin_edge = False
                    if thin_axis == 0:  # Wall thin in X, thin edges are at Y ends
                        if abs(hit_pos[1] - min_c[1]) < FACE_TOLERANCE or abs(hit_pos[1] - max_c[1]) < FACE_TOLERANCE:
                            hit_thin_edge = True
                    else:  # Wall thin in Y, thin edges are at X ends
                        if abs(hit_pos[0] - min_c[0]) < FACE_TOLERANCE or abs(hit_pos[0] - max_c[0]) < FACE_TOLERANCE:
                            hit_thin_edge = True

                    if not hit_thin_edge:
                        walls_to_hide.add(wall_name)

    # Clean up temporary boxes
    _remove_wall_boxes(temp_boxes)

    # Also hide adjacent parallel walls (for double-wall geometry like SceneAgent)
    # When outer wall is hit, also hide inner wall that's within 0.2m
    ADJACENT_WALL_TOLERANCE = 0.2

    # For each hit wall, find adjacent parallel walls
    expanded_walls_to_hide = set(walls_to_hide)
    for hit_wall_name in walls_to_hide:
        if hit_wall_name not in wall_info:
            continue
        hit_center, hit_thin_axis, hit_thin_pos, hit_min, hit_max = wall_info[hit_wall_name]

        for other_name, (other_center, other_thin_axis, other_thin_pos, other_min, other_max) in wall_info.items():
            if other_name == hit_wall_name:
                continue
            # Must have same thin axis (parallel walls)
            if other_thin_axis != hit_thin_axis:
                continue
            # Must be close in the thin dimension
            if abs(other_thin_pos - hit_thin_pos) > ADJACENT_WALL_TOLERANCE:
                continue
            # Must overlap in the long dimension (not just parallel but offset)
            long_axis = 1 - hit_thin_axis
            overlap_min = max(hit_min[long_axis], other_min[long_axis])
            overlap_max = min(hit_max[long_axis], other_max[long_axis])
            if overlap_max > overlap_min:  # They overlap
                expanded_walls_to_hide.add(other_name)

    # Hide the walls
    hidden_walls = []
    for obj in bpy.data.objects:
        if obj.name in expanded_walls_to_hide:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden_walls.append(obj)

    bpy.context.view_layer.update()
    logger.debug(f"Hidden {len(hidden_walls)} walls for camera view (360° raycast, {len(walls_to_hide)} hit + {len(expanded_walls_to_hide) - len(walls_to_hide)} adjacent)")
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

    Detects floors by:
    1. Objects with "floor" in the name
    2. Flat horizontal objects (large X/Y extent, minimal Z extent)

    Returns:
        Tuple of (min_coords, max_coords) as Vectors, or None if no floors found.
    """
    floor_objects = []

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        # Check by name
        if "floor" in obj.name.lower():
            floor_objects.append(obj)
            continue

        # Check by geometry: flat horizontal objects (floors have minimal Z height)
        try:
            dims = obj.dimensions
            # Floor criteria: Z dimension < 0.1m AND horizontal extent > 1m
            if dims.z < 0.1 and dims.x > 1.0 and dims.y > 1.0:
                floor_objects.append(obj)
        except Exception:
            pass

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


def _detect_outer_wall_geometric(obj: bpy.types.Object) -> bool:
    """Detect if wall is on house perimeter using geometry and raycasting.

    A wall is considered outer if casting a ray in its outward direction
    (away from scene center) does not hit any floor within the building.
    This handles L-shaped buildings where outer walls aren't on bbox edges.

    Args:
        obj: Wall object to check.

    Returns:
        True if wall appears to be on the house perimeter.
    """
    floor_bounds = get_floor_bounds()
    if floor_bounds is None:
        return True  # Conservative fallback

    min_coords, max_coords = floor_bounds

    # Get wall center
    try:
        bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
        wall_center = Vector((bbox_world.min(axis=0) + bbox_world.max(axis=0)) / 2)
    except Exception:
        return True

    # First check: is wall on the bounding box edge?
    TOLERANCE = 0.3
    on_min_x = abs(wall_center[0] - min_coords.x) < TOLERANCE
    on_max_x = abs(wall_center[0] - max_coords.x) < TOLERANCE
    on_min_y = abs(wall_center[1] - min_coords.y) < TOLERANCE
    on_max_y = abs(wall_center[1] - max_coords.y) < TOLERANCE

    if on_min_x or on_max_x or on_min_y or on_max_y:
        return True

    # Second check: raycast to detect L-corner outer walls
    # Cast ray from wall center outward (away from scene center)
    # If no floor is hit within building bounds, it's an outer wall
    scene_centroid = compute_scene_centroid()
    inward_normal = compute_wall_normal(obj, scene_centroid)
    outward_normal = -inward_normal

    # Cast ray from just outside the wall in the outward direction
    ray_origin = Vector((
        wall_center.x + outward_normal.x * 0.3,
        wall_center.y + outward_normal.y * 0.3,
        0.1  # Near floor level
    ))

    # Max distance to check (building diagonal)
    max_dist = ((max_coords.x - min_coords.x)**2 + (max_coords.y - min_coords.y)**2)**0.5

    depsgraph = bpy.context.evaluated_depsgraph_get()
    result, location, normal, index, hit_obj, matrix = bpy.context.scene.ray_cast(
        depsgraph, ray_origin, Vector((outward_normal.x, outward_normal.y, 0)), distance=max_dist
    )

    # If ray doesn't hit anything, or hits something far away, this is outer wall
    if not result:
        return True

    # If it hits a floor nearby, it's interior; if it hits far away floor, it's outer
    hit_dist = (location - ray_origin).length
    if hit_dist > 1.0:  # More than 1m means it went outside the room
        return True

    return False


def is_outer_wall(obj: bpy.types.Object) -> bool:
    """Check if a wall is an outer (perimeter) wall.

    Uses custom property if set during scene loading, otherwise falls back
    to geometric detection based on floor bounding box.

    Args:
        obj: Wall object to check.

    Returns:
        True if wall is on the house perimeter, False if interior partition.
    """
    # Check explicit marking first (set during scene load)
    if "is_outer_wall" in obj:
        return obj["is_outer_wall"]

    # Fallback: geometric detection
    return _detect_outer_wall_geometric(obj)


def mark_outer_walls_from_json(scene_dir: pathlib.Path) -> int:
    """Mark walls as outer/inner based on house_layout.json (SceneAgent format).

    Reads the house_layout.json file and marks Blender wall objects based on
    whether they are on a room boundary that faces the exterior (no adjacent room).

    For L-shaped houses, this correctly identifies outer walls that are not on
    the overall house bounding box but still face outside.

    Args:
        scene_dir: Path to scene directory containing house_layout.json.

    Returns:
        Number of walls marked, or 0 if JSON not found.
    """
    # Try to find house_layout.json
    json_path = scene_dir / "house_layout.json"
    if not json_path.exists():
        # Try in parent directory (for assets subdirectory structure)
        json_path = scene_dir.parent / "house_layout.json"
        if not json_path.exists():
            logger.debug(f"house_layout.json not found in {scene_dir}")
            return 0

    try:
        with open(json_path, "r") as f:
            layout_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load house_layout.json: {e}")
        return 0

    # Build list of all exterior edges from room boundaries
    # An edge is exterior if no other room shares that boundary
    placed_rooms = layout_data.get("placed_rooms", [])
    if not placed_rooms:
        return 0

    # Collect all room boundaries as (edge_type, coord, range_min, range_max)
    # edge_type: 'x' for vertical walls (constant x), 'y' for horizontal walls (constant y)
    room_edges = []
    for room in placed_rooms:
        pos = room.get("position", [0, 0])
        width = room.get("width", 0)
        depth = room.get("depth", 0)

        room_min_x, room_min_y = pos[0], pos[1]
        room_max_x, room_max_y = pos[0] + width, pos[1] + depth

        # Four edges of this room
        room_edges.append(('x', room_min_x, room_min_y, room_max_y, room.get("room_id")))  # west
        room_edges.append(('x', room_max_x, room_min_y, room_max_y, room.get("room_id")))  # east
        room_edges.append(('y', room_min_y, room_min_x, room_max_x, room.get("room_id")))  # south
        room_edges.append(('y', room_max_y, room_min_x, room_max_x, room.get("room_id")))  # north

    # Find exterior edges (edges not shared with another room)
    EDGE_TOLERANCE = 0.2
    exterior_edges = []
    for edge in room_edges:
        edge_type, coord, range_min, range_max, room_id = edge
        is_shared = False

        # Check if another room shares this edge
        for other_edge in room_edges:
            if other_edge[4] == room_id:  # Same room
                continue
            other_type, other_coord, other_min, other_max, _ = other_edge
            if edge_type != other_type:
                continue
            # Check if coordinates match and ranges overlap
            if abs(coord - other_coord) < EDGE_TOLERANCE:
                # Check for range overlap
                overlap = min(range_max, other_max) - max(range_min, other_min)
                if overlap > EDGE_TOLERANCE:
                    is_shared = True
                    break

        if not is_shared:
            exterior_edges.append((edge_type, coord, range_min, range_max))

    logger.debug(f"Found {len(exterior_edges)} exterior edges from {len(placed_rooms)} rooms")

    # Mark Blender wall objects based on position relative to exterior edges
    marked_count = 0
    walls = get_all_walls()
    TOLERANCE = 0.2  # 20cm tolerance for wall-to-edge matching

    for wall_obj in walls:
        # Get wall bounding box in world space
        try:
            bbox_local = np.array(wall_obj.bound_box)
            bbox_world = np.array([wall_obj.matrix_world @ Vector(corner) for corner in bbox_local])
            wall_min = bbox_world.min(axis=0)
            wall_max = bbox_world.max(axis=0)
            wall_center = (wall_min + wall_max) / 2

            # Determine wall orientation (thin dimension)
            dims = wall_max - wall_min
            is_x_wall = dims[0] < dims[1]  # Thin in X = wall runs along Y
        except Exception:
            wall_obj["is_outer_wall"] = True
            marked_count += 1
            continue

        # Check if wall matches any exterior edge
        is_outer = False
        for edge_type, coord, range_min, range_max in exterior_edges:
            if edge_type == 'x' and is_x_wall:
                # Vertical wall (constant x) - check if x matches and y range overlaps
                if abs(wall_center[0] - coord) < TOLERANCE:
                    wall_y_min, wall_y_max = wall_min[1], wall_max[1]
                    overlap = min(wall_y_max, range_max) - max(wall_y_min, range_min)
                    if overlap > -TOLERANCE:
                        is_outer = True
                        break
            elif edge_type == 'y' and not is_x_wall:
                # Horizontal wall (constant y) - check if y matches and x range overlaps
                if abs(wall_center[1] - coord) < TOLERANCE:
                    wall_x_min, wall_x_max = wall_min[0], wall_max[0]
                    overlap = min(wall_x_max, range_max) - max(wall_x_min, range_min)
                    if overlap > -TOLERANCE:
                        is_outer = True
                        break

        wall_obj["is_outer_wall"] = is_outer
        marked_count += 1
        logger.debug(f"Marked wall '{wall_obj.name}' at ({float(wall_center[0]):.2f}, {float(wall_center[1]):.2f}) as outer={is_outer}")

    logger.info(f"Marked {marked_count} walls using exterior edges from house_layout.json")
    return marked_count


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

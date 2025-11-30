#!/usr/bin/env python3
"""
Convert SceneWeaver output to SceneEval format.

This script:
1. Reads SceneWeaver output directory (layout JSON, room info, Blender file)
2. Opens the Blender file and exports each object as a separate GLB
3. Creates a SceneEval scene state JSON with proper transforms and architecture

Usage:
    # Run with Blender's Python
    blender --background --python convert_SceneEval.py -- \
        --input_dir /path/to/SceneWeaver/output/Design_me_a_messy_kids_bedroom_0 \
        --output_dir /path/to/SceneEval/input/SceneWeaver \
        --scene_id 0

    # Or with uv run (if bpy is installed)
    uv run python convert_SceneEval.py \
        --input_dir /path/to/SceneWeaver/output/Design_me_a_messy_kids_bedroom_0 \
        --output_dir /path/to/SceneEval/input/SceneWeaver \
        --scene_id 0
"""

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


# =============================================================================
# Scene State Template
# =============================================================================

SCENE_STATE_JSON_BASE = {
    "format": "sceneState",
    "scene": {
        "version": "scene@1.0.2",
        "id": "",
        "unit": 1.0,
        "up": [0, 0, 1],
        "front": [0, 1, 0],
        "assetSource": ["sceneweaver"],
        "modifications": [],
        "arch": {
            "version": "arch@1.0.2",
            "id": "",
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "coords2d": [0, 1],
            "scaleToMeters": 1,
            "defaults": {
                "Floor": {"depth": 0.05},
                "Ceiling": {"depth": 0.05},
                "Wall": {"depth": 0.1, "extraHeight": 0.035}
            },
            "elements": [],
            "regions": [],
            "holes": [],
            "images": [],
            "materials": [],
            "textures": []
        },
        "object": []
    },
    "selected": []
}

DEFAULT_WALL_HEIGHT = 2.8


# =============================================================================
# Helper Functions
# =============================================================================

def get_final_iteration(scene_dir: Path) -> int:
    """Get the final iteration number from args.json."""
    args_file = scene_dir / "args.json"
    if not args_file.exists():
        raise FileNotFoundError(f"args.json not found in {scene_dir}")

    with open(args_file) as f:
        args = json.load(f)

    return args.get("iter", 0)


def euler_to_quaternion(euler_xyz: list[float]) -> list[float]:
    """Convert Euler angles (XYZ radians) to quaternion [x, y, z, w]."""
    euler = Euler(euler_xyz, 'XYZ')
    quat = euler.to_quaternion()
    return [quat.x, quat.y, quat.z, quat.w]


def build_transform_matrix(location: list[float], rotation: list[float], scale: list[float] = None) -> list[float]:
    """
    Build a 4x4 transformation matrix from location, rotation (Euler XYZ), and scale.

    Returns:
        Flattened 4x4 matrix in column-major order (as expected by SceneEval).
    """
    loc = Vector(location)
    rot = Euler(rotation, 'XYZ')
    scl = Vector(scale) if scale else Vector((1, 1, 1))

    # Build transformation matrix
    mat_loc = Matrix.Translation(loc)
    mat_rot = rot.to_matrix().to_4x4()
    mat_scl = Matrix.Diagonal(scl).to_4x4()

    # Combined transform: location * rotation * scale
    mat = mat_loc @ mat_rot @ mat_scl

    # Flatten in column-major order (transpose then flatten row by row)
    # SceneEval expects data as a flat list
    flat = []
    for col in range(4):
        for row in range(4):
            flat.append(mat[row][col])

    return flat


def extract_transform_from_blender_matrix(matrix: Matrix) -> dict:
    """
    Extract transform dict from Blender 4x4 world matrix.

    Args:
        matrix: Blender 4x4 transformation matrix (matrix_world)

    Returns:
        Dictionary with transform data in SceneEval format.
    """
    loc, rot, scale = matrix.decompose()

    # Column-major flattening for SceneEval
    flat_data = []
    for col in range(4):
        for row in range(4):
            flat_data.append(matrix[row][col])

    return {
        "rows": 4,
        "cols": 4,
        "data": flat_data,
        "rotation": [rot.x, rot.y, rot.z, rot.w],
        "translation": [loc.x, loc.y, loc.z],
        "scale": [scale.x, scale.y, scale.z]
    }


def create_identity_transform() -> dict:
    """
    Create an Identity transform dict.

    Used when the world transform is already baked into the GLB vertex positions.

    Returns:
        Dictionary with Identity transform in SceneEval format.
    """
    return {
        "rows": 4,
        "cols": 4,
        "data": [
            1.0, 0.0, 0.0, 0.0,  # Column 0
            0.0, 1.0, 0.0, 0.0,  # Column 1
            0.0, 0.0, 1.0, 0.0,  # Column 2
            0.0, 0.0, 0.0, 1.0   # Column 3
        ],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0]
    }


def build_architecture(roomsize: list[float], wall_height: float = DEFAULT_WALL_HEIGHT) -> dict:
    """
    Build architecture elements (floor and walls) from room size.

    Args:
        roomsize: [width, depth] of the room
        wall_height: Height of walls (default 2.8m)

    Returns:
        Dictionary containing arch elements in SceneEval format.
    """
    width, depth = roomsize

    elements = []
    wall_ids = []

    # Floor element
    floor = {
        "id": "floor|room0",
        "type": "Floor",
        "roomId": "room0",
        "depth": 0.05,
        "points": [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, depth, 0.0],
            [0.0, depth, 0.0]
        ],
        "materials": [{"name": "surface", "diffuse": "#888888"}]
    }
    elements.append(floor)

    # Wall elements (4 walls forming a box)
    walls = [
        {
            "id": "wall|room0|south",
            "type": "Wall",
            "roomId": "room0",
            "height": wall_height,
            "depth": 0.1,
            "points": [[0.0, 0.0, 0.0], [width, 0.0, 0.0]],
            "holes": [],
            "materials": [{"name": "surface", "diffuse": "#cccccc"}]
        },
        {
            "id": "wall|room0|east",
            "type": "Wall",
            "roomId": "room0",
            "height": wall_height,
            "depth": 0.1,
            "points": [[width, 0.0, 0.0], [width, depth, 0.0]],
            "holes": [],
            "materials": [{"name": "surface", "diffuse": "#cccccc"}]
        },
        {
            "id": "wall|room0|north",
            "type": "Wall",
            "roomId": "room0",
            "height": wall_height,
            "depth": 0.1,
            "points": [[width, depth, 0.0], [0.0, depth, 0.0]],
            "holes": [],
            "materials": [{"name": "surface", "diffuse": "#cccccc"}]
        },
        {
            "id": "wall|room0|west",
            "type": "Wall",
            "roomId": "room0",
            "height": wall_height,
            "depth": 0.1,
            "points": [[0.0, depth, 0.0], [0.0, 0.0, 0.0]],
            "holes": [],
            "materials": [{"name": "surface", "diffuse": "#cccccc"}]
        }
    ]

    for i, wall in enumerate(walls):
        elements.append(wall)
        wall_ids.append(i + 1)  # Wall indices (floor is 0)

    regions = [{
        "id": "room0",
        "type": "Other",
        "walls": wall_ids
    }]

    return {
        "elements": elements,
        "regions": regions
    }


def extract_category(obj_name: str) -> str:
    """Extract category from SceneWeaver object name (e.g., '5348940_BedFactory' -> 'bed')."""
    parts = obj_name.split("_")
    if len(parts) > 1:
        factory = parts[-1].replace("Factory", "").lower()
        return factory
    return "object"


def generate_object_id_from_blender(obj: bpy.types.Object) -> str:
    """
    Generate a unique ID from Blender object name.

    Input pattern: {FactoryType}({seed}).spawn_asset({seed2})
    Output pattern: {seed}_{FactoryType}_{seed2} or {seed}_{FactoryType}_{seed2}.001

    Args:
        obj: Blender object

    Returns:
        Human-readable object ID like "3164690_BedFactory_7191960" or "3164690_PillowFactory_123.001"
    """
    name = obj.name

    # Extract factory type (before first parenthesis)
    paren_idx = name.find("(")
    factory_type = name[:paren_idx] if paren_idx > 0 else "Unknown"

    # Extract seed (number between first parentheses)
    try:
        seed_start = name.find("(") + 1
        seed_end = name.find(")")
        seed = name[seed_start:seed_end]
    except:
        seed = "0"

    # Extract spawn_asset seed (number between parentheses after .spawn_asset)
    spawn_seed = "0"
    spawn_marker = ".spawn_asset("
    spawn_idx = name.find(spawn_marker)
    if spawn_idx >= 0:
        spawn_start = spawn_idx + len(spawn_marker)
        spawn_end = name.find(")", spawn_start)
        if spawn_end > spawn_start:
            spawn_seed = name[spawn_start:spawn_end]

    # Handle Blender duplicate suffix (.001, .002, etc.)
    # These appear after the last closing parenthesis
    suffix = ""
    last_paren = name.rfind(")")
    if last_paren >= 0 and last_paren < len(name) - 1:
        suffix = name[last_paren + 1:]  # e.g., ".001"

    return f"{seed}_{factory_type}_{spawn_seed}{suffix}"


def is_in_hidden_or_asset_collection(obj: bpy.types.Object) -> bool:
    """
    Check if an object is in a hidden or raw asset collection.

    SceneWeaver stores raw asset templates in collections prefixed with:
    - 'assets:' - raw asset collections (e.g., fruits for scatter)
    - 'placeholders:' - placeholder geometry

    These should NOT be exported as placed scene objects.

    Args:
        obj: Blender object to check

    Returns:
        True if object should be excluded from export
    """
    # Check all collections this object belongs to
    for col in obj.users_collection:
        # Skip objects in raw asset collections
        if col.name.startswith('assets:'):
            return True
        if col.name.startswith('placeholders:'):
            return True
        # Skip objects in collections with hide_render=True
        if col.hide_render:
            return True
    return False


def find_all_spawn_asset_objects() -> list[bpy.types.Object]:
    """
    Find all Blender objects matching the spawn_asset pattern.

    These are individual exportable mesh objects from Infinigen factories.
    Excludes objects in hidden/asset collections (raw templates not placed in scene).

    Pattern: {FactoryType}({seed}).spawn_asset({seed2})
    Examples:
        - BedFactory(3164690).spawn_asset(7191960)
        - MattressFactory(3164690).spawn_asset(7191960)
        - ObjaverseCategoryFactory(1210932).spawn_asset(7874330)

    Returns:
        List of Blender objects that are individual exportable meshes
    """
    spawn_asset_objects = []

    for obj in bpy.data.objects:
        # Skip non-mesh objects (cameras, lights, empties, etc.)
        if obj.type != 'MESH':
            continue

        # Match the spawn_asset pattern
        if ".spawn_asset(" in obj.name:
            # Skip objects in hidden/asset collections (raw templates, not placed)
            if is_in_hidden_or_asset_collection(obj):
                continue

            # Validate object has actual geometry
            if obj.data is not None and len(obj.data.vertices) > 0:
                spawn_asset_objects.append(obj)

    return spawn_asset_objects


# =============================================================================
# Texture Baking Functions
# =============================================================================

def material_needs_baking(mat) -> bool:
    """
    Check if a material needs baking for glTF export.

    Returns True if the material doesn't have an image texture that will export properly.
    This catches:
    - Procedural textures (noise, voronoi, etc.)
    - GROUP nodes (Infinigen shader groups)
    - RGB nodes (solid colors)
    - MIX nodes (color mixing)
    - Color Ramp nodes
    - Any non-image input to Principled BSDF

    Args:
        mat: Blender material to check

    Returns:
        True if material needs baking
    """
    if not mat or not mat.use_nodes:
        return False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Find Principled BSDF or Material Output
    principled = None
    material_output = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
        elif node.type == 'OUTPUT_MATERIAL':
            material_output = node


    # If there's a Principled BSDF, check if Base Color has image texture
    if principled:
        base_color_input = principled.inputs.get('Base Color')
        if base_color_input and base_color_input.is_linked:
            for link in links:
                if link.to_socket == base_color_input:
                    if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                        return False  # Already has image texture - no baking needed

    # If no Principled BSDF (e.g., GROUP-based shader), check what's connected to Material Output
    # GROUP nodes need baking since they contain procedural shaders internally
    if not principled and material_output:
        surface_input = material_output.inputs.get('Surface')
        if surface_input and surface_input.is_linked:
            for link in links:
                if link.to_socket == surface_input:
                    # GROUP node - needs baking
                    if link.from_node.type == 'GROUP':
                        return True

    # If we have a Principled BSDF but no image texture connected, we need baking
    if principled:
        return True

    return False


def has_procedural_materials(obj: bpy.types.Object) -> bool:
    """
    Check if object has any materials that need baking.

    Args:
        obj: Blender object to check

    Returns:
        True if any material needs baking
    """
    if not obj.data or not hasattr(obj.data, 'materials') or not obj.data.materials:
        return False

    for mat in obj.data.materials:
        if material_needs_baking(mat):
            return True

    return False


def _rewire_node_tree(node_tree, processed_groups=None) -> int:
    """
    Recursively rewire procedural textures in a node tree to use UV coordinates.

    Returns number of nodes rewired.
    """
    if processed_groups is None:
        processed_groups = set()

    if node_tree is None:
        return 0

    nodes = node_tree.nodes
    links = node_tree.links
    rewired_count = 0

    # Find or create Texture Coordinate node in this tree
    tex_coord = None
    for node in nodes:
        if node.type == 'TEX_COORD':
            tex_coord = node
            break
    if tex_coord is None:
        tex_coord = nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-600, 0)

    # Procedural texture types that need Vector input rewiring
    procedural_types = ['TEX_NOISE', 'TEX_VORONOI', 'TEX_MUSGRAVE', 'TEX_WAVE',
                        'TEX_BRICK', 'TEX_CHECKER', 'TEX_GRADIENT', 'TEX_MAGIC']

    for node in nodes:
        # Recursively handle GROUP nodes
        if node.type == 'GROUP' and node.node_tree:
            group_name = node.node_tree.name
            if group_name not in processed_groups:
                processed_groups.add(group_name)
                rewired_count += _rewire_node_tree(node.node_tree, processed_groups)

        # Handle procedural texture nodes
        if node.type in procedural_types:
            vector_input = node.inputs.get('Vector')
            if vector_input:
                if not vector_input.is_linked:
                    # Unlinked = using Generated coords by default, connect UV instead
                    links.new(tex_coord.outputs['UV'], vector_input)
                    rewired_count += 1
                else:
                    # Check if linked to Object/Generated output or via Mapping node
                    for link in list(links):
                        if link.to_socket == vector_input:
                            if link.from_node.type == 'TEX_COORD':
                                if link.from_socket.name in ['Object', 'Generated']:
                                    links.remove(link)
                                    links.new(tex_coord.outputs['UV'], vector_input)
                                    rewired_count += 1
                            elif link.from_node.type == 'MAPPING':
                                # Mapping node - rewire its input instead
                                mapping_node = link.from_node
                                mapping_vec_input = mapping_node.inputs.get('Vector')
                                if mapping_vec_input:
                                    if not mapping_vec_input.is_linked:
                                        # Unlinked mapping uses Generated by default
                                        links.new(tex_coord.outputs['UV'], mapping_vec_input)
                                        rewired_count += 1
                                    else:
                                        for mlink in list(links):
                                            if mlink.to_socket == mapping_vec_input:
                                                if mlink.from_node.type == 'TEX_COORD':
                                                    if mlink.from_socket.name in ['Object', 'Generated']:
                                                        links.remove(mlink)
                                                        links.new(tex_coord.outputs['UV'], mapping_vec_input)
                                                        rewired_count += 1

        # Handle standalone Mapping nodes connected to Object/Generated
        if node.type == 'MAPPING':
            vector_input = node.inputs.get('Vector')
            if vector_input:
                if not vector_input.is_linked:
                    # Unlinked = using Generated, connect UV
                    try:
                        links.new(tex_coord.outputs['UV'], vector_input)
                        rewired_count += 1
                    except Exception:
                        pass  # Link may already exist or node tree is modified
                elif vector_input.is_linked:
                    # Get the actual connected link safely
                    connected_link = None
                    for link in links:
                        try:
                            if link.to_socket == vector_input:
                                connected_link = link
                                break
                        except ReferenceError:
                            continue  # Link was removed

                    if connected_link:
                        try:
                            from_node_type = connected_link.from_node.type
                            from_socket_name = connected_link.from_socket.name
                            if from_node_type == 'TEX_COORD' and from_socket_name in ['Object', 'Generated']:
                                links.remove(connected_link)
                                links.new(tex_coord.outputs['UV'], vector_input)
                                rewired_count += 1
                        except (ReferenceError, RuntimeError):
                            pass  # Link was already modified

    return rewired_count


def rewire_procedural_textures_to_uv(mat) -> None:
    """
    Replace Object/Generated texture coordinates with UV coordinates.

    This is CRITICAL for baking procedural textures. Infinigen shaders use
    Object or Generated (3D) coordinates for procedural textures, but baking
    samples at UV positions (2D). Without this rewiring, baked textures are black.

    This function recursively processes GROUP nodes to handle nested shaders.

    Args:
        mat: Blender material to rewire
    """
    if not mat or not mat.use_nodes:
        return

    print(f"      Rewiring procedural textures for: {mat.name}")
    rewired = _rewire_node_tree(mat.node_tree)
    if rewired > 0:
        print(f"      Rewired {rewired} texture coordinate connections")


def ensure_uv_layer(obj: bpy.types.Object, force_recreate: bool = False) -> bool:
    """
    Ensure object has a UV layer, creating one via smart projection if needed.
    After projection, UVs are scaled to fill the 0-1 range to maximize texture usage.

    Args:
        obj: Blender mesh object
        force_recreate: If True, recreate UV layer even if one exists (for baking)

    Returns:
        True if UV layer exists or was created successfully
    """
    if obj.data.uv_layers and not force_recreate:
        return True

    try:
        # Remove existing UV layers if force_recreate
        if force_recreate and obj.data.uv_layers:
            while obj.data.uv_layers:
                obj.data.uv_layers.remove(obj.data.uv_layers[0])

        # Store current selection state
        prev_active = bpy.context.view_layer.objects.active
        prev_selected = [o for o in bpy.context.selected_objects]

        # Select only this object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Enter edit mode and create UV projection
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.01)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Scale UVs to fill the 0-1 range
        _normalize_uvs(obj)

        # Restore selection state
        bpy.ops.object.select_all(action='DESELECT')
        for o in prev_selected:
            o.select_set(True)
        bpy.context.view_layer.objects.active = prev_active

        return True
    except Exception as e:
        print(f"    Warning: Failed to create UV layer for {obj.name}: {e}")
        return False


def _normalize_uvs(obj: bpy.types.Object) -> None:
    """
    Scale and translate UVs to fill the 0-1 range.
    This ensures baked textures use the full texture space.
    """
    if not obj.data.uv_layers:
        return

    mesh = obj.data
    uv_layer = mesh.uv_layers.active

    # Find UV bounds
    min_u, max_u = float('inf'), float('-inf')
    min_v, max_v = float('inf'), float('-inf')

    for loop in mesh.loops:
        uv = uv_layer.data[loop.index].uv
        min_u = min(min_u, uv[0])
        max_u = max(max_u, uv[0])
        min_v = min(min_v, uv[1])
        max_v = max(max_v, uv[1])

    # Calculate scale and offset to fill 0-1 range with small margin
    margin = 0.02  # Small margin to avoid edge artifacts
    u_range = max_u - min_u
    v_range = max_v - min_v

    if u_range <= 0 or v_range <= 0:
        return

    # Scale to fill (1 - 2*margin) of the texture
    target_range = 1.0 - 2 * margin
    scale_u = target_range / u_range
    scale_v = target_range / v_range

    # Use uniform scale to maintain aspect ratio
    scale = min(scale_u, scale_v)

    # Apply transformation
    for loop in mesh.loops:
        uv = uv_layer.data[loop.index].uv
        # Center and scale
        new_u = (uv[0] - min_u) * scale + margin
        new_v = (uv[1] - min_v) * scale + margin
        uv_layer.data[loop.index].uv = (new_u, new_v)


def bake_procedural_materials(obj: bpy.types.Object, output_dir: Path, resolution: int = 1024) -> bool:
    """
    Bake procedural materials to image textures for glTF export.

    This function:
    1. Checks if the object has procedural materials (noise, voronoi, etc.)
    2. Creates UV mapping if needed
    3. Bakes the diffuse color to an image texture
    4. Replaces the procedural shader connection with the baked image

    Args:
        obj: Blender object with procedural materials
        output_dir: Directory to save baked textures
        resolution: Texture resolution (default 1024x1024)

    Returns:
        True if baking succeeded or was not needed, False on error
    """
    if not has_procedural_materials(obj):
        return True

    print(f"    Baking procedural materials for: {obj.name}")

    # Recreate UVs with smart projection for optimal baking coverage
    if not ensure_uv_layer(obj, force_recreate=True):
        print(f"    Warning: Could not create UVs for {obj.name}, skipping bake")
        return False

    # Store original render engine
    original_engine = bpy.context.scene.render.engine

    try:
        # Set up Cycles for baking
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'CPU'  # CPU is more reliable for baking
        bpy.context.scene.cycles.samples = 16  # Low samples for diffuse bake

        # Select and activate object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Process each material that needs baking
        for mat_idx, mat in enumerate(obj.data.materials):
            if not mat or not mat.use_nodes:
                continue

            # Skip materials that don't need baking
            if not material_needs_baking(mat):
                continue

            # CRITICAL: Rewire 3D procedural coordinates (Object/Generated) to UV
            # before baking. Without this, procedural textures like Musgrave/Noise
            # that use 3D coords will bake as black (UV coords ≠ 3D world coords)
            rewire_procedural_textures_to_uv(mat)

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Find Principled BSDF node (may not exist for GROUP-based materials)
            principled = None
            material_output = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    principled = node
                elif node.type == 'OUTPUT_MATERIAL':
                    material_output = node

            # Create bake image
            safe_obj_name = obj.name.replace("(", "_").replace(")", "_").replace(".", "_")
            safe_mat_name = mat.name.replace("(", "_").replace(")", "_").replace(".", "_")
            img_name = f"{safe_obj_name}_{safe_mat_name}_baked"
            bake_img = bpy.data.images.new(img_name, resolution, resolution, alpha=False)
            bake_img.colorspace_settings.name = 'sRGB'

            # Create image texture node for baking target
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.image = bake_img
            if principled:
                tex_node.location = (principled.location.x - 400, principled.location.y)
            else:
                tex_node.location = (-400, 0)

            # Set as active node for baking
            nodes.active = tex_node

            # Configure bake settings
            print(f"      Baking material: {mat.name}")

            # Try EMIT bake first - this bypasses lighting and captures color directly
            # For procedural textures that use 3D coordinates, we need to temporarily
            # connect the color output to an emission shader

            # Find what's connected to Principled Base Color or GROUP Surface output
            base_color_source = None
            base_color_default = None
            group_surface_source = None

            if principled:
                base_color_input = principled.inputs.get('Base Color')
                if base_color_input:
                    if base_color_input.is_linked:
                        for link in links:
                            if link.to_socket == base_color_input:
                                base_color_source = link.from_socket
                                break
                    else:
                        # Use default value (solid color)
                        base_color_default = base_color_input.default_value[:]
            else:
                # For GROUP-based materials, find what's connected to Material Output Surface
                # We'll use the GROUP's BSDF output for emission baking
                if material_output:
                    surface_input = material_output.inputs.get('Surface')
                    if surface_input and surface_input.is_linked:
                        for link in links:
                            if link.to_socket == surface_input:
                                group_surface_source = link.from_socket
                                break

            # Bake the material using the appropriate method
            if base_color_source or base_color_default:
                # Principled BSDF with known Base Color - use emission baking
                emission_node = nodes.new('ShaderNodeEmission')
                emission_node.location = (principled.location.x + 200, principled.location.y - 200) if principled else (200, -200)

                # Connect base color source to emission, or set default color
                if base_color_source:
                    links.new(base_color_source, emission_node.inputs['Color'])
                else:
                    emission_node.inputs['Color'].default_value = base_color_default

                # Save and replace material output connection
                original_surface_link = None
                if material_output:
                    surface_input = material_output.inputs.get('Surface')
                    if surface_input and surface_input.is_linked:
                        for link in links:
                            if link.to_socket == surface_input:
                                original_surface_link = link.from_socket
                                links.remove(link)
                                break
                    links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])

                # Bake EMIT
                bpy.ops.object.bake(type='EMIT')

                # Restore original connection
                if original_surface_link and material_output:
                    for link in list(links):
                        if link.to_socket == material_output.inputs.get('Surface'):
                            links.remove(link)
                            break
                    links.new(original_surface_link, material_output.inputs['Surface'])

                # Remove emission node
                nodes.remove(emission_node)

            elif group_surface_source:
                # GROUP-based material - extract Base Color from Principled BSDF inside GROUP
                # Then use emission baking
                group_node = group_surface_source.node

                # Find Principled BSDF inside the GROUP and get its Base Color source
                inner_base_color_source = None
                inner_principled = None
                if group_node.type == 'GROUP' and group_node.node_tree:
                    inner_nodes = group_node.node_tree.nodes
                    inner_links = group_node.node_tree.links
                    for inner_node in inner_nodes:
                        if inner_node.type == 'BSDF_PRINCIPLED':
                            inner_principled = inner_node
                            bc_input = inner_node.inputs.get('Base Color')
                            if bc_input and bc_input.is_linked:
                                for ilink in inner_links:
                                    if ilink.to_socket == bc_input:
                                        inner_base_color_source = ilink.from_socket
                                        break
                            break

                if inner_base_color_source and inner_principled:
                    # Create emission node INSIDE the GROUP
                    inner_emission = inner_nodes.new('ShaderNodeEmission')
                    inner_emission.location = (inner_principled.location.x + 200, inner_principled.location.y - 200)

                    # Connect the Base Color source to emission
                    inner_links.new(inner_base_color_source, inner_emission.inputs['Color'])

                    # Find GROUP OUTPUT node and temporarily replace the BSDF output
                    group_output = None
                    original_bsdf_link = None
                    for inner_node in inner_nodes:
                        if inner_node.type == 'GROUP_OUTPUT':
                            group_output = inner_node
                            break

                    if group_output:
                        # Find the BSDF input on group output
                        bsdf_input = None
                        for inp in group_output.inputs:
                            if inp.name == 'BSDF' or 'bsdf' in inp.name.lower() or inp.type == 'SHADER':
                                bsdf_input = inp
                                break

                        if bsdf_input and bsdf_input.is_linked:
                            for ilink in list(inner_links):
                                if ilink.to_socket == bsdf_input:
                                    original_bsdf_link = ilink.from_socket
                                    inner_links.remove(ilink)
                                    break
                            inner_links.new(inner_emission.outputs['Emission'], bsdf_input)

                    # Bake EMIT
                    bpy.ops.object.bake(type='EMIT')

                    # Restore original GROUP connection
                    if group_output and bsdf_input and original_bsdf_link:
                        for ilink in list(inner_links):
                            if ilink.to_socket == bsdf_input:
                                inner_links.remove(ilink)
                                break
                        inner_links.new(original_bsdf_link, bsdf_input)

                    # Remove temporary emission node
                    inner_nodes.remove(inner_emission)
                else:
                    # Fallback: Use DIFFUSE bake with color pass only (no direct/indirect lighting)
                    bpy.context.scene.render.bake.use_pass_direct = False
                    bpy.context.scene.render.bake.use_pass_indirect = False
                    bpy.context.scene.render.bake.use_pass_color = True
                    bpy.ops.object.bake(type='DIFFUSE')

            else:
                # Fall back to COMBINED bake
                bpy.context.scene.render.bake.use_pass_direct = True
                bpy.context.scene.render.bake.use_pass_indirect = True
                bpy.context.scene.render.bake.use_pass_diffuse = True
                bpy.context.scene.render.bake.use_pass_glossy = False
                bpy.context.scene.render.bake.use_pass_transmission = False
                bpy.context.scene.render.bake.use_pass_emit = True
                bpy.ops.object.bake(type='COMBINED')

            # Save baked image and ensure it's properly linked
            img_path = output_dir / f"{img_name}.png"
            bake_img.filepath_raw = str(img_path)
            bake_img.file_format = 'PNG'
            bake_img.save()
            print(f"      Saved: {img_path.name}")

            # Reload the saved image to ensure it's properly linked for GLB export
            bake_img.pack()  # Pack the image into the blend file so it exports with GLB

            # Connect baked image to shader
            if principled:
                # Remove existing links to Base Color input
                for link in list(links):
                    if link.to_node == principled and link.to_socket.name == 'Base Color':
                        links.remove(link)
                # Connect baked image to Principled BSDF Base Color
                links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
            else:
                # For GROUP-based materials, create a new Principled BSDF and connect
                # This replaces the GROUP shader with a simple image-based material
                new_principled = nodes.new('ShaderNodeBsdfPrincipled')
                new_principled.location = (0, 0)

                # Connect baked image to new Principled BSDF
                links.new(tex_node.outputs['Color'], new_principled.inputs['Base Color'])

                # Connect new Principled BSDF to Material Output
                if material_output:
                    # Remove existing link to Surface
                    for link in list(links):
                        if link.to_node == material_output and link.to_socket.name == 'Surface':
                            links.remove(link)
                    links.new(new_principled.outputs['BSDF'], material_output.inputs['Surface'])

        return True

    except Exception as e:
        print(f"    Error baking materials for {obj.name}: {e}")
        return False

    finally:
        # Restore original render engine
        bpy.context.scene.render.engine = original_engine


# =============================================================================
# Blend File Cleanup Functions
# =============================================================================

def _save_cleaned_blend_file(output_path: Path) -> None:
    """
    Save a cleaned version of the current blend file.

    Removes objects in hidden/asset collections (raw asset templates) that would
    cause camera framing issues when rendering. This fixes the "zoomed out" render
    problem where giant template objects at origin expand the scene bounds.

    Args:
        output_path: Path to save the cleaned blend file
    """
    # Collect objects to delete (can't delete while iterating)
    objects_to_delete = []

    for obj in bpy.data.objects:
        if is_in_hidden_or_asset_collection(obj):
            objects_to_delete.append(obj)

    # Delete the objects
    if objects_to_delete:
        print(f"    Removing {len(objects_to_delete)} raw asset objects from blend file")
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Also delete the empty asset collections
    collections_to_delete = []
    for col in bpy.data.collections:
        if col.name.startswith('assets:') or col.name.startswith('placeholders:'):
            collections_to_delete.append(col)

    for col in collections_to_delete:
        bpy.data.collections.remove(col)

    # Clean up orphan data (meshes, materials, etc. no longer used)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    # Save the cleaned blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


# =============================================================================
# Blender Export Functions
# =============================================================================

def export_object_as_glb(obj: bpy.types.Object, output_path: Path) -> bool:
    """
    Export a single object (NOT its children) as a GLB file.

    The object's world transform is baked into the vertex positions via export_apply=True,
    so the scene state should use an Identity transform.

    Args:
        obj: Blender object to export (single mesh, no hierarchy)
        output_path: Path to save the GLB file

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        # Select ONLY this object (no children)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Export as GLB with export_apply=True
        # This bakes the world transform into the vertex positions
        # Use standard Y-up glTF coordinate system - SceneEval/Blender will convert back to Z-up on import
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            use_selection=True,
            export_apply=True,
            export_format='GLB',
            export_yup=True  # Standard glTF Y-up, converted back to Z-up on import
        )

        return True
    except Exception as e:
        print(f"Error exporting {obj.name}: {e}")
        return False


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_sceneweaver_scene(
    input_dir: Path,
    output_dir: Path,
    scene_id: int,
    wall_height: float = DEFAULT_WALL_HEIGHT,
    export_architecture: bool = False
) -> Path:
    """
    Convert a SceneWeaver scene to SceneEval format.

    Args:
        input_dir: Path to SceneWeaver output directory
        output_dir: Path to SceneEval input directory
        scene_id: Scene ID for output filename
        wall_height: Height of walls
        export_architecture: If True, export floor/wall meshes from SceneWeaver with baked textures

    Returns:
        Path to the created scene state JSON file.
    """
    print(f"Converting SceneWeaver scene: {input_dir}")

    # Get final iteration
    iter_num = get_final_iteration(input_dir)
    print(f"  Final iteration: {iter_num}")

    # Load required files
    blend_file = input_dir / "record_files" / f"scene_{iter_num}.blend"
    layout_file = input_dir / "record_scene" / f"layout_{iter_num}.json"
    roominfo_file = input_dir / "roominfo.json"

    if not blend_file.exists():
        raise FileNotFoundError(f"Blender file not found: {blend_file}")
    if not layout_file.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_file}")
    if not roominfo_file.exists():
        raise FileNotFoundError(f"Room info file not found: {roominfo_file}")

    # Load layout and room info
    with open(layout_file) as f:
        layout = json.load(f)
    with open(roominfo_file) as f:
        roominfo = json.load(f)

    roomsize = layout.get("roomsize", roominfo.get("roomsize", [4.0, 4.0]))
    print(f"  Room size: {roomsize}")

    # Create output directories
    # Assets go in scene_N/assets/ subdirectory
    # Scene JSON goes directly in output_dir as scene_N.json (SceneEval convention)
    scene_assets_dir = output_dir / f"scene_{scene_id}" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_assets_dir.mkdir(parents=True, exist_ok=True)

    # Open Blender file
    print(f"  Opening Blender file: {blend_file}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    # DIRECT BLENDER TRAVERSAL: Find all spawn_asset objects
    # This matches SceneWeaver's own object counting methodology (Nobj_unique)
    spawn_objects = find_all_spawn_asset_objects()
    print(f"  Found {len(spawn_objects)} spawn_asset objects in Blender")

    # Process objects
    objects_data = []

    for obj in spawn_objects:
        # Generate unique object ID from Blender object name
        obj_id = generate_object_id_from_blender(obj)
        print(f"    Processing: {obj.name} -> {obj_id}")

        # Bake procedural materials to textures before export
        bake_procedural_materials(obj, scene_assets_dir, resolution=1024)

        # Export single object (NO hierarchy)
        glb_path = scene_assets_dir / f"{obj_id}.glb"
        if not export_object_as_glb(obj, glb_path):
            print(f"      Failed to export: {obj_id}")
            continue

        print(f"      Exported: {glb_path.name}")

        # Use Identity transform since world position is baked into the GLB via export_apply=True
        transform_data = create_identity_transform()

        # Build object entry
        obj_entry = {
            "id": obj_id,
            "modelId": f"sw.s{scene_id}__{obj_id}",
            "index": len(objects_data),
            "parentId": "",
            "parentIndex": -1,
            "transform": transform_data
        }
        objects_data.append(obj_entry)

    # Export architecture with baked textures if requested
    architecture_objects = []
    if export_architecture:
        print("  Exporting SceneWeaver architecture with baked textures...")

        # Find and export floor mesh
        floor_obj = None
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and 'floor' in obj.name.lower():
                floor_obj = obj
                break

        if floor_obj:
            print(f"    Found floor: {floor_obj.name}")
            bake_procedural_materials(floor_obj, scene_assets_dir, resolution=1024)
            floor_glb = scene_assets_dir / "architecture_floor.glb"
            if export_object_as_glb(floor_obj, floor_glb):
                print(f"      Exported: {floor_glb.name}")
                architecture_objects.append({
                    "id": "architecture_floor",
                    "modelId": f"sw.s{scene_id}__architecture_floor",
                    "index": len(objects_data) + len(architecture_objects),
                    "parentId": "",
                    "parentIndex": -1,
                    "transform": create_identity_transform(),
                    "isArchitecture": True
                })

        # Find and export wall mesh
        wall_obj = None
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and 'wall' in obj.name.lower():
                wall_obj = obj
                break

        if wall_obj:
            print(f"    Found wall: {wall_obj.name}")
            bake_procedural_materials(wall_obj, scene_assets_dir, resolution=1024)
            wall_glb = scene_assets_dir / "architecture_walls.glb"
            if export_object_as_glb(wall_obj, wall_glb):
                print(f"      Exported: {wall_glb.name}")
                architecture_objects.append({
                    "id": "architecture_walls",
                    "modelId": f"sw.s{scene_id}__architecture_walls",
                    "index": len(objects_data) + len(architecture_objects),
                    "parentId": "",
                    "parentIndex": -1,
                    "transform": create_identity_transform(),
                    "isArchitecture": True
                })

    # Build architecture (default simple geometry if not exporting SceneWeaver architecture)
    arch_data = build_architecture(roomsize, wall_height)

    # Create scene state
    scene_state = SCENE_STATE_JSON_BASE.copy()
    scene_state = json.loads(json.dumps(scene_state))  # Deep copy

    scene_uuid = str(uuid4())
    scene_state["scene"]["id"] = scene_uuid
    scene_state["scene"]["arch"]["id"] = scene_uuid
    scene_state["scene"]["arch"]["elements"] = arch_data["elements"]
    scene_state["scene"]["arch"]["regions"] = arch_data["regions"]
    # Only include regular objects (architecture GLBs are exported for visual rendering but not
    # included as objects - they would be empty meshes that cause evaluation issues)
    scene_state["scene"]["object"] = objects_data

    # Save scene state (directly in output_dir, not in subdirectory)
    output_json = output_dir / f"scene_{scene_id}.json"
    with open(output_json, "w") as f:
        json.dump(scene_state, f, indent=2)

    print(f"  Saved scene state: {output_json}")
    print(f"  Exported {len(objects_data)} objects")

    # Save cleaned blend file for high-quality rendering
    # (The GLB exports have texture baking issues, original blend has perfect materials)
    # Remove objects in hidden/asset collections to fix camera framing and reduce file size
    original_blend_dest = scene_assets_dir / "original_sceneweaver.blend"
    _save_cleaned_blend_file(original_blend_dest)
    print(f"  Saved cleaned blend file: {original_blend_dest}")

    return output_json


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    # Handle Blender's argument passing (-- separates Blender args from script args)
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Convert SceneWeaver output to SceneEval format"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to SceneWeaver output directory (e.g., output/Design_me_a_messy_kids_bedroom_0)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./input/SceneWeaver"),
        help="Path to SceneEval input directory (default: ./input/SceneWeaver)"
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=0,
        help="Scene ID for output filename (default: 0)"
    )
    parser.add_argument(
        "--wall_height",
        type=float,
        default=DEFAULT_WALL_HEIGHT,
        help=f"Wall height in meters (default: {DEFAULT_WALL_HEIGHT})"
    )
    parser.add_argument(
        "--export_architecture",
        action="store_true",
        help="Export floor/wall meshes from SceneWeaver with baked textures (default: use SceneEval's simple architecture)"
    )

    args = parser.parse_args(argv)

    convert_sceneweaver_scene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        wall_height=args.wall_height,
        export_architecture=args.export_architecture
    )


if __name__ == "__main__":
    main()

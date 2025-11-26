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
    Output pattern: {seed}_{FactoryType} or {seed}_{FactoryType}.001

    Args:
        obj: Blender object

    Returns:
        Human-readable object ID like "3164690_BedFactory" or "3164690_PillowFactory.001"
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

    # Handle Blender duplicate suffix (.001, .002, etc.)
    # These appear after the last closing parenthesis
    suffix = ""
    last_paren = name.rfind(")")
    if last_paren >= 0 and last_paren < len(name) - 1:
        suffix = name[last_paren + 1:]  # e.g., ".001"

    return f"{seed}_{factory_type}{suffix}"


def find_all_spawn_asset_objects() -> list[bpy.types.Object]:
    """
    Find all Blender objects matching the spawn_asset pattern.

    These are individual exportable mesh objects from Infinigen factories.

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
            # Validate object has actual geometry
            if obj.data is not None and len(obj.data.vertices) > 0:
                spawn_asset_objects.append(obj)

    return spawn_asset_objects


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
    wall_height: float = DEFAULT_WALL_HEIGHT
) -> Path:
    """
    Convert a SceneWeaver scene to SceneEval format.

    Args:
        input_dir: Path to SceneWeaver output directory
        output_dir: Path to SceneEval input directory
        scene_id: Scene ID for output filename
        wall_height: Height of walls

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
            "modelId": f"sceneweaver.scene_{scene_id}__{obj_id}",
            "index": len(objects_data),
            "parentId": "",
            "parentIndex": -1,
            "transform": transform_data
        }
        objects_data.append(obj_entry)

    # Build architecture
    arch_data = build_architecture(roomsize, wall_height)

    # Create scene state
    scene_state = SCENE_STATE_JSON_BASE.copy()
    scene_state = json.loads(json.dumps(scene_state))  # Deep copy

    scene_uuid = str(uuid4())
    scene_state["scene"]["id"] = scene_uuid
    scene_state["scene"]["arch"]["id"] = scene_uuid
    scene_state["scene"]["arch"]["elements"] = arch_data["elements"]
    scene_state["scene"]["arch"]["regions"] = arch_data["regions"]
    scene_state["scene"]["object"] = objects_data

    # Save scene state (directly in output_dir, not in subdirectory)
    output_json = output_dir / f"scene_{scene_id}.json"
    with open(output_json, "w") as f:
        json.dump(scene_state, f, indent=2)

    print(f"  Saved scene state: {output_json}")
    print(f"  Exported {len(objects_data)} objects")

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

    args = parser.parse_args(argv)

    convert_sceneweaver_scene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        wall_height=args.wall_height
    )


if __name__ == "__main__":
    main()

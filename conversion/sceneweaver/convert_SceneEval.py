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


def extract_factory_type(layout_name: str) -> str:
    """
    Extract factory type from layout JSON object name.

    Examples:
        '5348940_BedFactory' -> 'BedFactory'
        '9391942_wardrobe' -> 'wardrobe'
        '4061705_toy' -> 'toy'
    """
    parts = layout_name.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return layout_name


def find_blender_object_by_factory(factory_type: str, used_objects: set) -> bpy.types.Object | None:
    """
    Find a Blender object matching the factory type.

    Args:
        factory_type: The factory type to search for (e.g., 'BedFactory', 'wardrobe')
        used_objects: Set of already matched object names to avoid duplicates

    Returns:
        Matching Blender object or None
    """
    # Blender naming patterns:
    # - Procedural: 'BedFactory(3164690).spawn_asset(7191960)'
    # - Objaverse: 'ObjaverseCategoryFactory(1210932).spawn_asset(7874330)'

    # First try exact factory match with .spawn_asset
    for obj in bpy.data.objects:
        if obj.name in used_objects:
            continue
        if obj.type != 'MESH':
            continue

        # Match pattern: {FactoryType}({seed}).spawn_asset({seed})
        if obj.name.startswith(f"{factory_type}(") and ".spawn_asset(" in obj.name:
            return obj

    # For generic types like 'wardrobe', 'toy', 'laundrybasket' - these are ObjaverseCategoryFactory
    # We need a different matching strategy - perhaps by position or by keeping track

    return None


def build_factory_to_blender_map() -> dict[str, list[bpy.types.Object]]:
    """
    Build a mapping from factory types to all matching Blender objects.

    Returns:
        Dict mapping factory type to list of Blender objects
    """
    factory_map = {}

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if ".spawn_asset(" not in obj.name:
            continue

        # Extract factory type from Blender object name
        # Pattern: {FactoryType}({seed}).spawn_asset({seed})
        paren_idx = obj.name.find("(")
        if paren_idx > 0:
            factory_type = obj.name[:paren_idx]
            if factory_type not in factory_map:
                factory_map[factory_type] = []
            factory_map[factory_type].append(obj)

    return factory_map


def find_closest_objaverse_object(
    target_location: list[float],
    objaverse_objects: list[bpy.types.Object],
    used_objects: set,
    tolerance: float = 1.0
) -> bpy.types.Object | None:
    """
    Find the closest ObjaverseCategoryFactory object by position.

    Args:
        target_location: Target position [x, y, z]
        objaverse_objects: List of ObjaverseCategoryFactory Blender objects
        used_objects: Set of already matched object names
        tolerance: Maximum distance threshold

    Returns:
        Closest matching Blender object or None
    """
    best_obj = None
    best_distance = float('inf')

    target = Vector(target_location)

    for obj in objaverse_objects:
        if obj.name in used_objects:
            continue

        # Get object's world location
        obj_location = obj.location
        distance = (target - obj_location).length

        if distance < best_distance and distance < tolerance:
            best_distance = distance
            best_obj = obj

    return best_obj


# =============================================================================
# Blender Export Functions
# =============================================================================

def select_object_hierarchy(obj: bpy.types.Object) -> None:
    """Select an object and all its children recursively."""
    bpy.ops.object.select_all(action='DESELECT')

    def select_recursive(o):
        o.select_set(True)
        for child in o.children:
            select_recursive(child)

    select_recursive(obj)
    bpy.context.view_layer.objects.active = obj


def export_object_as_glb(obj: bpy.types.Object, output_path: Path) -> bool:
    """
    Export a single object (and its children) as a GLB file.

    Args:
        obj: Blender object to export
        output_path: Path to save the GLB file

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        # Select object and children
        select_object_hierarchy(obj)

        # Store original transform
        orig_location = obj.location.copy()
        orig_rotation = obj.rotation_euler.copy()

        # Reset transform for export (we'll apply transform in scene state)
        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)

        # Export as GLB
        # Keep Z-up coordinate system (don't convert to glTF's Y-up)
        # This ensures transforms from Blender match the exported geometry
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            use_selection=True,
            export_apply=True,
            export_format='GLB',
            export_yup=False  # Keep Blender's Z-up coordinate system
        )

        # Restore original transform
        obj.location = orig_location
        obj.rotation_euler = orig_rotation

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
    scene_output_dir = output_dir / f"scene_{scene_id}"
    assets_dir = scene_output_dir / "assets"
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Open Blender file
    print(f"  Opening Blender file: {blend_file}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    # Process objects
    objects_data = []
    objects_info = layout.get("objects", {})

    print(f"  Found {len(objects_info)} objects in layout")

    # Build mapping of factory types to Blender objects
    factory_map = build_factory_to_blender_map()
    print(f"  Blender factory types: {list(factory_map.keys())}")

    # Track used objects to avoid duplicates
    used_blender_objects = set()

    for obj_name, obj_info in objects_info.items():
        print(f"    Processing: {obj_name}")

        # Extract factory type from layout name
        factory_type = extract_factory_type(obj_name)
        print(f"      Factory type: {factory_type}")

        # Find matching Blender object
        blender_obj = None

        # Try direct factory match first
        if factory_type in factory_map:
            for candidate in factory_map[factory_type]:
                if candidate.name not in used_blender_objects:
                    blender_obj = candidate
                    break

        # If not found, try case-insensitive match or partial match
        if blender_obj is None:
            for ftype, candidates in factory_map.items():
                if factory_type.lower() in ftype.lower() or ftype.lower() in factory_type.lower():
                    for candidate in candidates:
                        if candidate.name not in used_blender_objects:
                            blender_obj = candidate
                            break
                    if blender_obj:
                        break

        # If still not found, try position-based matching with ObjaverseCategoryFactory
        if blender_obj is None and "ObjaverseCategoryFactory" in factory_map:
            location = obj_info.get("location", [0, 0, 0])
            blender_obj = find_closest_objaverse_object(
                location,
                factory_map["ObjaverseCategoryFactory"],
                used_blender_objects,
                tolerance=1.5  # Allow up to 1.5 meter distance
            )
            if blender_obj:
                print(f"      Position-matched to ObjaverseCategoryFactory: {blender_obj.name}")

        if blender_obj is None:
            print(f"      Warning: No Blender object found for factory '{factory_type}', skipping")
            continue

        used_blender_objects.add(blender_obj.name)
        print(f"      Matched Blender object: {blender_obj.name}")

        # Export object as GLB
        glb_path = assets_dir / f"{obj_name}.glb"
        if export_object_as_glb(blender_obj, glb_path):
            print(f"      Exported: {glb_path.name}")
        else:
            print(f"      Failed to export: {obj_name}")
            continue

        # Get transform info from layout
        location = obj_info.get("location", [0, 0, 0])
        rotation = obj_info.get("rotation", [0, 0, 0])
        size = obj_info.get("size", [1, 1, 1])

        # Build transform matrix
        transform_data = build_transform_matrix(location, rotation)

        # Build object entry
        # Include scene_id in modelId to support per-scene assets
        obj_entry = {
            "id": obj_name,
            "modelId": f"sceneweaver.scene_{scene_id}__{obj_name}",
            "index": len(objects_data),
            "parentId": "",
            "parentIndex": -1,
            "transform": {
                "rows": 4,
                "cols": 4,
                "data": transform_data,
                "rotation": euler_to_quaternion(rotation),
                "translation": location,
                "scale": [1.0, 1.0, 1.0]  # Scale is baked into the exported GLB
            }
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

    # Save scene state
    output_json = scene_output_dir / f"scene_{scene_id}.json"
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

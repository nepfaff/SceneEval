#!/usr/bin/env python3
"""
Blender-based conversion script for IDesign output to SceneEval input format.

This script replicates IDesign's place_in_blender.py logic exactly, then exports
each object with transforms baked into vertices (like SceneWeaver approach).

IMPORTANT: This script MUST be run with Blender 4.2 (not 5.0+).
IDesign was developed with Blender 4.2, and there is an undocumented breaking
change in bpy.ops.transform.rotate between Blender 4.2 and 5.0 that causes
objects to be mirrored incorrectly.

Usage:
    /home/ubuntu/blender-4.2.0-linux-x64/blender --background --python convert_SceneEval_blender.py -- \\
        --source_dir /home/ubuntu/IDesign/data/scenes_batch \\
        --output_dir /path/to/SceneEval/input/IDesign \\
        --mapping '{"0": 106, "1": 56, "2": 39, "3": 74, "4": 94}'

This converts IDesign output to SceneEval format:
    - Loads GLBs and positions them exactly like IDesign's place_in_blender.py
    - Exports each object with world transform baked into vertices
    - Stores identity transforms in JSON (like SceneWeaver)
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import bpy
from mathutils import Matrix, Vector, Euler


# =============================================================================
# Scene State Template
# =============================================================================

SCENE_STATE_JSON_BASE = {
    "format": "sceneState",
    "scene": {
        "arch": {
            "coords2d": [0, 1],
            "defaults": {
                "Ceiling": {"depth": 0.05},
                "Floor": {"depth": 0.05},
                "Ground": {"depth": 0.08},
                "Wall": {"depth": 0.1, "extraHeight": 0.035},
                "textureSource": "smtTexture"
            },
            "elements": [],
            "front": [0, 1, 0],
            "holes": [],
            "id": "generated-arch",
            "images": [],
            "materials": [],
            "regions": [{"id": "0", "type": "Other", "walls": []}],
            "scaleToMeters": 1,
            "textures": [],
            "up": [0, 0, 1],
            "version": "arch@1.0.2"
        },
        "assetSource": ["idesign"],
        "front": [0, 1, 0],
        "id": "converted-scene",
        "modifications": [],
        "object": [],
        "objectFrontVector": [0, 1, 0],
        "unit": 1.0,
        "up": [0, 0, 1],
        "version": "scene@1.0.2"
    },
    "selected": []
}


# =============================================================================
# Helper Functions (from IDesign's place_in_blender.py)
# =============================================================================

def import_glb(file_path: str, object_name: str):
    """Import a GLB file and rename the imported object."""
    bpy.ops.import_scene.gltf(filepath=file_path)
    imported_object = bpy.context.view_layer.objects.active
    if imported_object is not None:
        imported_object.name = object_name


def find_glb_files(directory: str) -> dict:
    """Find all GLB files in a directory tree."""
    glb_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".glb"):
                key = file.split(".")[0]
                if key not in glb_files:
                    glb_files[key] = os.path.join(root, file)
    return glb_files


def get_highest_parent_objects() -> list:
    """Get all objects with no parent."""
    return [obj for obj in bpy.data.objects if obj.parent is None]


def delete_empty_objects():
    """Delete all empty objects in the scene."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'EMPTY':
            bpy.data.objects.remove(obj)


def select_meshes_under_empty(empty_object_name: str):
    """Recursively select all meshes under an empty object."""
    empty_object = bpy.data.objects.get(empty_object_name)
    if empty_object is not None and empty_object.type == 'EMPTY':
        for child in empty_object.children:
            if child.type == 'MESH':
                child.select_set(True)
                bpy.context.view_layer.objects.active = child
            else:
                select_meshes_under_empty(child.name)


def rescale_object(obj, scale: dict):
    """Rescale object based on target dimensions."""
    if obj.type == 'MESH':
        bbox_dimensions = obj.dimensions
        if bbox_dimensions.x > 0 and bbox_dimensions.y > 0 and bbox_dimensions.z > 0:
            scale_factors = (
                scale["length"] / bbox_dimensions.x,
                scale["width"] / bbox_dimensions.y,
                scale["height"] / bbox_dimensions.z
            )
            obj.scale = scale_factors


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


# =============================================================================
# Transform Helpers
# =============================================================================

def create_identity_transform() -> dict:
    """Create an identity transform dict for SceneEval format."""
    return {
        "rows": 4,
        "cols": 4,
        "data": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0]
    }


def humanize_object_id(obj_id: str) -> str:
    """Convert object ID to human-readable description."""
    return obj_id.replace('_', ' ').replace('-', ' ')


# =============================================================================
# Architecture Building
# =============================================================================

def create_arch_elements(room_elements: list) -> tuple:
    """Create architecture elements from room layout."""
    elements = []
    wall_ids = []

    # Find floor element
    floor_elem = None
    for elem in room_elements:
        if elem.get('itemType') == 'floor' or elem.get('new_object_id') == 'middle of the room':
            floor_elem = elem
            break

    if not floor_elem:
        print("  Warning: No floor element found in room layout")
        return [], []

    # Get room dimensions from floor
    room_width = floor_elem['size_in_meters']['length']
    room_depth = floor_elem['size_in_meters']['width']

    # Find ceiling for height, or use wall height
    room_height = 2.5
    for elem in room_elements:
        if elem.get('itemType') == 'ceiling' or elem.get('new_object_id') == 'ceiling':
            room_height = elem['position']['z']
            break
        if elem.get('itemType') == 'wall':
            room_height = elem['size_in_meters']['height']

    # Create floor element
    floor_points = [
        [0, 0, 0],
        [room_width, 0, 0],
        [room_width, room_depth, 0],
        [0, room_depth, 0]
    ]
    floor = {
        "depth": 0.05,
        "id": 0,
        "materials": [{"diffuse": "#888899", "name": "surface"}],
        "points": floor_points,
        "roomId": "0",
        "type": "Floor"
    }
    elements.append(floor)

    # Create wall elements
    wall_defs = [
        ([0, 0, 0], [room_width, 0, 0]),
        ([room_width, 0, 0], [room_width, room_depth, 0]),
        ([room_width, room_depth, 0], [0, room_depth, 0]),
        ([0, room_depth, 0], [0, 0, 0]),
    ]

    for i, (start, end) in enumerate(wall_defs):
        wall = {
            "depth": 0.1,
            "height": room_height,
            "holes": [],
            "id": i + 1,
            "materials": [{"diffuse": "#888899", "name": "surface"}],
            "points": [start, end],
            "roomId": "0",
            "type": "Wall"
        }
        elements.append(wall)
        wall_ids.append(i + 1)

    return elements, wall_ids


# =============================================================================
# Export Function
# =============================================================================

def export_object_as_glb(obj, output_path: Path) -> bool:
    """Export a single object with world transform baked into vertices."""
    try:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            use_selection=True,
            export_apply=True,  # Bake world transform into vertices
            export_format='GLB',
            export_yup=True
        )
        return True
    except Exception as e:
        print(f"    Error exporting {obj.name}: {e}")
        return False


# =============================================================================
# Main Conversion Function (replicates IDesign's place_in_blender.py)
# =============================================================================

def convert_idesign_scene(
    scene_dir: Path,
    output_dir: Path,
    scene_id: int
) -> Path:
    """
    Convert an IDesign scene to SceneEval format.

    This replicates IDesign's place_in_blender.py logic exactly:
    1. Import GLBs
    2. Join meshes under empty parents
    3. Normalize objects (move to origin, apply transforms, center origin)
    4. Position, rotate (+180°), and scale objects
    5. Export each object with baked transforms
    """
    print(f"\nConverting {scene_dir.name} -> scene_{scene_id}")

    # Load scene graph
    scene_graph_path = scene_dir / "scene_graph.json"
    if not scene_graph_path.exists():
        raise FileNotFoundError(f"scene_graph.json not found in {scene_dir}")

    with open(scene_graph_path) as f:
        data = json.load(f)

    # Separate room layout elements from furniture objects
    room_elements = []
    furniture_objects = []
    objects_in_room = {}

    layout_ids = ["south_wall", "north_wall", "east_wall", "west_wall", "middle of the room", "ceiling"]

    for item in data:
        obj_id = item.get("new_object_id", "")
        if obj_id in layout_ids:
            room_elements.append(item)
        else:
            furniture_objects.append(item)
            objects_in_room[obj_id] = item

    # Create output directories
    scene_output_dir = output_dir / f"scene_{scene_id}"
    assets_output_dir = scene_output_dir / "assets"
    assets_output_dir.mkdir(parents=True, exist_ok=True)

    # Clear Blender scene
    clear_scene()

    # Find GLB files
    assets_dir = scene_dir / "Assets"
    glb_file_paths = find_glb_files(str(assets_dir))

    # =======================================================================
    # STEP 1: Import all GLBs (from IDesign's place_in_blender.py lines 100-102)
    # =======================================================================
    print(f"  Importing {len(objects_in_room)} objects...")
    for item_id in objects_in_room.keys():
        if item_id in glb_file_paths:
            glb_path = glb_file_paths[item_id]
            import_glb(glb_path, item_id)

    # =======================================================================
    # STEP 2: Join meshes under empty parents (lines 104-117)
    # =======================================================================
    parents = get_highest_parent_objects()
    empty_parents = [p for p in parents if p.type == "EMPTY"]

    for empty_parent in empty_parents:
        bpy.ops.object.select_all(action='DESELECT')
        select_meshes_under_empty(empty_parent.name)

        if bpy.context.selected_objects:
            bpy.ops.object.join()
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

            joined_object = bpy.context.view_layer.objects.active
            if joined_object is not None:
                joined_object.name = empty_parent.name + "-joined"

    bpy.context.view_layer.objects.active = None

    # =======================================================================
    # STEP 3: Normalize all mesh objects (lines 121-129)
    # Move to origin, apply transforms, center origin
    # =======================================================================
    MSH_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
    for obj in MSH_OBJS:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        obj.location = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # =======================================================================
    # STEP 4: Position, rotate, and scale objects (lines 131-141)
    # This is where IDesign adds +180° to all z_angles
    # =======================================================================
    MSH_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
    for obj in MSH_OBJS:
        # Get object ID (remove "-joined" suffix if present)
        obj_id = obj.name.split("-")[0]

        if obj_id not in objects_in_room:
            continue

        item = objects_in_room[obj_id]

        # Position
        pos = item.get("position", {"x": 0, "y": 0, "z": 0})
        object_position = (pos["x"], pos["y"], pos["z"])

        # Rotation: z_angle + 180° (THIS IS THE KEY!)
        z_angle = item.get("rotation", {}).get("z_angle", 0)
        object_rotation_z = (z_angle / 180.0) * math.pi + math.pi

        # Apply transforms exactly like IDesign
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        obj.location = object_position
        bpy.ops.transform.rotate(value=object_rotation_z, orient_axis='Z')

        # Scale
        size = item.get("size_in_meters", {"length": 1, "width": 1, "height": 1})
        rescale_object(obj, size)

    # Clean up empty objects
    bpy.ops.object.select_all(action='DESELECT')
    delete_empty_objects()

    # =======================================================================
    # STEP 5: Export each object with baked transforms (like SceneWeaver)
    # =======================================================================
    print(f"  Exporting objects with baked transforms...")
    objects_data = []

    MSH_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
    for idx, obj in enumerate(MSH_OBJS):
        obj_id = obj.name.split("-")[0]

        if obj_id not in objects_in_room:
            continue

        item = objects_in_room[obj_id]

        # Export GLB with baked transforms
        glb_path = assets_output_dir / f"{obj_id}.glb"
        if not export_object_as_glb(obj, glb_path):
            print(f"    Failed to export: {obj_id}")
            continue

        # Build description
        style = item.get('style', '')
        material = item.get('material', '')
        base_name = humanize_object_id(obj_id)
        if style and material:
            description = f"{style} {material} {base_name}"
        elif style:
            description = f"{style} {base_name}"
        else:
            description = base_name

        # Create object entry with IDENTITY transform (baked into GLB)
        obj_entry = {
            "id": obj_id,
            "index": len(objects_data),
            "modelId": f"idesign.scene_{scene_id}__{obj_id}",
            "parentId": "",
            "parentIndex": -1,
            "transform": create_identity_transform(),
            "description": description
        }
        objects_data.append(obj_entry)
        print(f"    Exported: {obj_id}")

    # =======================================================================
    # Build scene state JSON
    # =======================================================================
    output = json.loads(json.dumps(SCENE_STATE_JSON_BASE))
    output["scene"]["id"] = str(uuid4())
    output["scene"]["arch"]["id"] = str(uuid4())

    # Create architecture
    elements, wall_ids = create_arch_elements(room_elements)
    output["scene"]["arch"]["elements"] = elements
    output["scene"]["arch"]["regions"][0]["walls"] = wall_ids

    # Add objects
    output["scene"]["object"] = objects_data

    # Save JSON
    output_json = output_dir / f"scene_{scene_id}.json"
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    # Copy original blend file if it exists
    original_blend = scene_dir / "scene.blend"
    if original_blend.exists():
        shutil.copy2(original_blend, assets_output_dir / "original_idesign.blend")

    print(f"  Saved: {output_json} ({len(objects_data)} objects)")

    return output_json


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    # Handle Blender's argument passing
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Convert IDesign scenes to SceneEval format using Blender"
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Path to IDesign scenes_batch directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to SceneEval input directory"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help='JSON mapping from source index to target ID, e.g., \'{"0": 106, "1": 56}\''
    )

    args = parser.parse_args(argv)

    # Parse mapping
    mapping = {}
    if args.mapping:
        mapping = json.loads(args.mapping)
        mapping = {str(k): v for k, v in mapping.items()}

    # Find all scene directories
    scene_dirs = sorted([
        d for d in args.source_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])

    print(f"Found {len(scene_dirs)} scenes to convert")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.output_dir}")
    if mapping:
        print(f"Mapping: {mapping}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scene_dir in scene_dirs:
        # Extract source index
        match = scene_dir.name.replace("scene_", "")
        source_idx = str(int(match))  # "000" -> "0"

        # Determine target scene ID
        if source_idx in mapping:
            target_id = mapping[source_idx]
        else:
            target_id = int(source_idx)

        convert_idesign_scene(scene_dir, args.output_dir, target_id)

    print(f"\nConversion complete! {len(scene_dirs)} scenes saved to {args.output_dir}")


if __name__ == "__main__":
    main()

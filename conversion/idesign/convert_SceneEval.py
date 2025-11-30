"""
Conversion script for IDesign output to SceneEval input format.

Usage:
    python conversion/idesign/convert_SceneEval.py \
        /home/ubuntu/IDesign/data/scenes_batch \
        input/IDesign

With custom ID mapping:
    python conversion/idesign/convert_SceneEval.py \
        /home/ubuntu/IDesign/data/scenes_batch \
        input/IDesign \
        --mapping '{"0": 106, "1": 56}'

This converts IDesign output to SceneEval format:
    - scene_graph.json -> scene_X.json
    - Assets/*.glb -> scene_X/assets/*.glb
    - Extracts positions, rotations (+180°), and scales from GLB bounding boxes
    - Creates architecture (floor, walls) from room layout elements
"""

import json
import math
import shutil
import argparse
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np

# Try to import trimesh for GLB bounding box calculation
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Scale calculation will use identity scale.")


# SceneEval scene state template
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
        "objectFrontVector": [0, 1, 0],  # Same as SceneAgent
        "unit": 1.0,
        "up": [0, 0, 1],
        "version": "scene@1.0.2"
    },
    "selected": []
}


def apply_transforms_and_export_glb(src_glb_path: Path, dest_glb_path: Path) -> tuple:
    """Load GLB, apply internal transforms, center at origin, export clean GLB in Y-up.

    IMPORTANT: We keep the GLB in Y-up format (glTF convention) so that Blender's
    automatic Y-up to Z-up conversion works correctly. If we baked Z-up into vertices,
    Blender would double-rotate the mesh.

    Args:
        src_glb_path: Source GLB file
        dest_glb_path: Destination path for cleaned GLB

    Returns:
        Tuple of (x_extent, y_extent, z_extent) bounding box in Y-up coordinates.
        In Y-up: X=width, Y=height(up), Z=depth
    """
    if not TRIMESH_AVAILABLE:
        return (1.0, 1.0, 1.0)

    try:
        # Load as scene to preserve hierarchy and transforms
        scene = trimesh.load(src_glb_path)

        # Get the combined mesh with all internal transforms applied
        # This puts the mesh in glTF's Y-up world coordinates
        mesh = None
        if hasattr(scene, 'to_geometry'):
            mesh = scene.to_geometry()
        elif hasattr(scene, 'dump'):
            mesh = scene.dump(concatenate=True)

        if mesh is None:
            mesh = trimesh.load(src_glb_path, force='mesh')

        if mesh is not None and hasattr(mesh, 'bounding_box'):
            # DO NOT apply Y-up to Z-up conversion here!
            # Blender will do this automatically when importing GLB.
            # If we bake it here, Blender double-rotates the mesh, causing
            # incorrect orientations that depend on the object's Z rotation.

            # Center at origin (like IDesign's ORIGIN_GEOMETRY, center='BOUNDS')
            centroid = mesh.bounding_box.centroid
            mesh.vertices -= centroid

            # Get bbox extents in Y-up coordinates
            # Y-up: X=width, Y=height(up), Z=depth
            extents = tuple(mesh.bounding_box.extents)

            # Export the cleaned mesh (still in Y-up)
            mesh.export(dest_glb_path)
            return extents

        return (1.0, 1.0, 1.0)

    except Exception as e:
        print(f"  Warning: Could not process GLB {src_glb_path.name}: {e}")
        return (1.0, 1.0, 1.0)


def build_transform(
    position: dict,
    z_angle_degrees: float,
    size_in_meters: dict,
    bbox_yup: tuple = None
) -> dict:
    """Build 4x4 transform matrix with rotation, scale, and translation.

    Args:
        position: {"x": float, "y": float, "z": float} in Z-up coords
        z_angle_degrees: Rotation around Z-axis in degrees
        size_in_meters: Target size {"length": X, "width": Y, "height": Z} in Z-up coords
        bbox_yup: Bounding box extents (X, Y, Z) in Y-up coordinates from GLB
                  Y-up: X=width, Y=height(up), Z=depth

    Returns:
        SceneEval transform dict with data, rotation, scale, translation
    """
    if bbox_yup is None:
        bbox_yup = (1.0, 1.0, 1.0)

    # Map Y-up bbox to Z-up size_in_meters:
    # Y-up: X=width, Y=height, Z=depth
    # Z-up size_in_meters: length=X, width=Y, height=Z
    # After Blender's Y-up to Z-up conversion: old_Y->new_Z, old_Z->new_-Y (but same magnitude)
    # So in Z-up after import: X stays X, old_Y(height) becomes Z(height), old_Z(depth) becomes Y(width)
    scale_x = size_in_meters['length'] / bbox_yup[0] if bbox_yup[0] > 1e-6 else 1.0  # X stays X
    scale_y = size_in_meters['width'] / bbox_yup[2] if bbox_yup[2] > 1e-6 else 1.0   # Y-up Z(depth) -> Z-up Y(width)
    scale_z = size_in_meters['height'] / bbox_yup[1] if bbox_yup[1] > 1e-6 else 1.0  # Y-up Y(height) -> Z-up Z(height)

    # Rotation: add +180° to match IDesign's place_in_blender.py
    # IDesign adds +π to all z_angles when placing objects in Blender
    z_rad = math.radians(z_angle_degrees) + math.pi
    cos_z = math.cos(z_rad)
    sin_z = math.sin(z_rad)

    # Build rotation matrix (Z-axis rotation)
    rot = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ])

    # Build scale matrix
    scale_mat = np.diag([scale_x, scale_y, scale_z])

    # Combined: rotation @ scale (scale first, then rotate)
    # This matches how the transform is applied: mesh -> scale -> rotate -> translate
    rs = rot @ scale_mat

    # Build 4x4 matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rs
    matrix[:3, 3] = [position['x'], position['y'], position['z']]

    # Flatten column-major for SceneEval
    data = matrix.T.flatten().tolist()

    # Quaternion for pure Z rotation: q = [0, 0, sin(θ/2), cos(θ/2)]
    qz = math.sin(z_rad / 2)
    qw = math.cos(z_rad / 2)
    quaternion = [0.0, 0.0, qz, qw]

    return {
        "rows": 4,
        "cols": 4,
        "data": data,
        "rotation": quaternion,
        "scale": [scale_x, scale_y, scale_z],
        "translation": [position['x'], position['y'], position['z']]
    }


def create_arch_elements(room_elements: list, room_height: float = 2.8) -> tuple:
    """Create floor and wall architecture elements from IDesign room elements.

    Args:
        room_elements: List of IDesign room layout elements (walls, floor, ceiling)
        room_height: Default room height if not found

    Returns:
        Tuple of (elements list, wall_ids list)
    """
    elements = []

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
    ceiling_elem = None
    wall_elem = None
    for elem in room_elements:
        if elem.get('itemType') == 'ceiling' or elem.get('new_object_id') == 'ceiling':
            ceiling_elem = elem
        if elem.get('itemType') == 'wall':
            wall_elem = elem

    if ceiling_elem:
        room_height = ceiling_elem['position']['z']
    elif wall_elem:
        room_height = wall_elem['size_in_meters']['height']

    # Create floor element (rectangle from origin)
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
    wall_ids = []
    wall_defs = [
        # (start, end, name)
        ([0, 0, 0], [room_width, 0, 0], "south"),
        ([room_width, 0, 0], [room_width, room_depth, 0], "east"),
        ([room_width, room_depth, 0], [0, room_depth, 0], "north"),
        ([0, room_depth, 0], [0, 0, 0], "west"),
    ]

    for i, (p1, p2, name) in enumerate(wall_defs):
        wall = {
            "depth": 0.1,
            "height": room_height,
            "holes": [],
            "id": i + 1,
            "materials": [{"diffuse": "#888899", "name": "surface"}],
            "points": [p1, p2],
            "roomId": "0",
            "type": "Wall"
        }
        elements.append(wall)
        wall_ids.append(i + 1)

    return elements, wall_ids


def humanize_object_id(obj_id: str) -> str:
    """Convert object ID to human-readable description.

    "twin_bed_1" -> "twin bed"
    "bean_bag_chair_1" -> "bean bag chair"
    """
    # Remove trailing numbers
    cleaned = re.sub(r'_\d+$', '', obj_id)
    # Replace underscores with spaces
    return cleaned.replace('_', ' ')


def convert_single_scene(
    source_dir: Path,
    target_dir: Path,
    scene_id: int
) -> bool:
    """Convert a single IDesign scene to SceneEval format.

    Args:
        source_dir: Path to IDesign scene directory (e.g., scene_000/)
        target_dir: Path to SceneEval input directory (e.g., input/IDesign/)
        scene_id: Scene ID for output naming

    Returns:
        True if conversion successful, False otherwise
    """
    scene_graph_path = source_dir / "scene_graph.json"
    assets_dir = source_dir / "Assets"
    scene_blend_path = source_dir / "scene.blend"

    # Check required files exist
    if not scene_graph_path.exists():
        print(f"  Warning: {scene_graph_path} not found, skipping scene")
        return False

    # Load scene_graph.json
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)

    # Separate furniture objects from room layout elements
    furniture_objects = []
    room_elements = []

    # Room element IDs (fixed infrastructure)
    room_element_ids = {'south_wall', 'north_wall', 'east_wall', 'west_wall',
                        'middle of the room', 'ceiling'}

    for item in scene_graph:
        obj_id = item.get('new_object_id', '')
        if obj_id in room_element_ids or item.get('itemType') in ('wall', 'floor', 'ceiling'):
            room_elements.append(item)
        else:
            # It's a furniture object if it has style/material
            if 'style' in item or 'material' in item:
                furniture_objects.append(item)

    # Create output directories
    scene_output_dir = target_dir / f"scene_{scene_id}"
    assets_output_dir = scene_output_dir / "assets"
    assets_output_dir.mkdir(parents=True, exist_ok=True)

    # Create the output scene state
    output = json.loads(json.dumps(SCENE_STATE_JSON_BASE))
    output["scene"]["id"] = str(uuid4())
    output["scene"]["arch"]["id"] = str(uuid4())

    # Create architecture from room elements
    elements, wall_ids = create_arch_elements(room_elements)
    output["scene"]["arch"]["elements"] = elements
    output["scene"]["arch"]["regions"][0]["walls"] = wall_ids

    # Process furniture objects
    objects = []
    for i, item in enumerate(furniture_objects):
        obj_id = item['new_object_id']
        position = item.get('position', {'x': 0, 'y': 0, 'z': 0})
        z_angle = item.get('rotation', {}).get('z_angle', 0)
        size = item.get('size_in_meters', {'length': 1, 'width': 1, 'height': 1})

        # Check for source GLB file
        src_glb_path = assets_dir / f"{obj_id}.glb" if assets_dir.exists() else None
        if src_glb_path and not src_glb_path.exists():
            src_glb_path = None

        # Process GLB: center at origin and get Y-up bbox
        # GLB stays in Y-up so Blender's automatic Y-up to Z-up conversion works correctly
        bbox_yup = (1.0, 1.0, 1.0)
        dest_glb = assets_output_dir / f"{obj_id}.glb"
        if src_glb_path and src_glb_path.exists():
            bbox_yup = apply_transforms_and_export_glb(src_glb_path, dest_glb)

        # Build transform (rotation + scale + translation)
        # bbox_yup is in Y-up coords, build_transform maps to Z-up size
        transform = build_transform(position, z_angle, size, bbox_yup)

        # Build description from style and material
        style = item.get('style', '')
        material = item.get('material', '')
        base_name = humanize_object_id(obj_id)
        if style and material:
            description = f"{style} {material} {base_name}"
        elif style:
            description = f"{style} {base_name}"
        elif material:
            description = f"{material} {base_name}"
        else:
            description = base_name

        obj = {
            "id": obj_id,
            "index": i,
            "modelId": f"idesign.scene_{scene_id}__{obj_id}",
            "parentId": "",
            "parentIndex": -1,
            "transform": transform,
            "description": description
        }
        objects.append(obj)

    output["scene"]["object"] = objects

    # Copy scene.blend if it exists (for high-quality rendering)
    if scene_blend_path.exists():
        dest_blend = assets_output_dir / "original_idesign.blend"
        shutil.copy2(scene_blend_path, dest_blend)
        print(f"  Copied scene.blend -> assets/original_idesign.blend")

    # Save output JSON
    output_json_path = target_dir / f"scene_{scene_id}.json"
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_json_path} ({len(objects)} objects)")
    return True


def convert_idesign_batch(
    source_batch_dir: Path,
    target_dir: Path,
    mapping: Optional[dict] = None
) -> None:
    """Convert all IDesign scenes from a batch directory to SceneEval format.

    Args:
        source_batch_dir: Path to IDesign scenes_batch directory
        target_dir: Path to SceneEval input directory (e.g., input/IDesign/)
        mapping: Optional dict mapping source scene index to target scene ID
                 e.g., {"0": 106, "1": 56} means scene_000 -> scene_106
    """
    source_batch_dir = Path(source_batch_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()

    if not source_batch_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_batch_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene directories
    scene_dirs = sorted([
        d for d in source_batch_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])

    if not scene_dirs:
        raise ValueError(f"No scene directories found in {source_batch_dir}")

    print(f"Found {len(scene_dirs)} scenes to convert")
    print(f"Source: {source_batch_dir}")
    print(f"Target: {target_dir}")
    if mapping:
        print(f"Mapping: {mapping}")
    print()

    converted = 0
    for scene_dir in scene_dirs:
        # Extract source index from directory name (e.g., "scene_000" -> 0)
        src_idx = int(scene_dir.name.split("_")[1])

        # Determine target scene ID
        if mapping:
            src_key = str(src_idx)
            if src_key in mapping:
                target_id = mapping[src_key]
            else:
                print(f"Skipping {scene_dir.name}: not in mapping")
                continue
        else:
            target_id = src_idx

        print(f"Converting {scene_dir.name} -> scene_{target_id}...")
        if convert_single_scene(scene_dir, target_dir, target_id):
            converted += 1

    print()
    print(f"Conversion complete! {converted} scenes saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IDesign output to SceneEval input format"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to IDesign scenes_batch directory"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to SceneEval input directory (e.g., input/IDesign)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help='JSON mapping from source index to target ID, e.g., \'{"0": 106, "1": 56}\''
    )
    args = parser.parse_args()

    mapping = None
    if args.mapping:
        mapping = json.loads(args.mapping)

    convert_idesign_batch(args.source_dir, args.target_dir, mapping)


if __name__ == "__main__":
    main()

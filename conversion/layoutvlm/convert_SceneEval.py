"""
Conversion script for LayoutVLM output to SceneEval input format.

Usage:
    python conversion/layoutvlm/convert_SceneEval.py \
        /home/ubuntu/LayoutVLM/results \
        input/LayoutVLM \
        --mapping '{"0": 106, "1": 56, "2": 39, "3": 74, "4": 94}'

This converts LayoutVLM output to SceneEval format:
    - scene.json + layout.json + complete_sandbox_program.py -> scene_X.json
    - Extracts asset IDs, positions, rotations, and bounding boxes
    - Creates architecture (floor, walls) from boundary
"""

import json
import re
import math
import argparse
from pathlib import Path
from typing import Optional


# SceneEval scene state template (based on existing converted files)
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
        "assetSource": ["objaverse"],
        "front": [0, 1, 0],
        "id": "converted-scene",
        "modifications": [],
        "object": [],
        "unit": 1.0,
        "up": [0, 0, 1],
        "version": "scene@1.0.2"
    },
    "selected": []
}


def euler_to_quaternion(rx: float, ry: float, rz: float) -> list:
    """Convert Euler angles (degrees) to quaternion [x, y, z, w]."""
    # Convert to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    # Calculate quaternion components
    cx = math.cos(rx / 2)
    sx = math.sin(rx / 2)
    cy = math.cos(ry / 2)
    sy = math.sin(ry / 2)
    cz = math.cos(rz / 2)
    sz = math.sin(rz / 2)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    return [qx, qy, qz, qw]


def quaternion_to_matrix(q: list) -> list:
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix (column-major)."""
    x, y, z, w = q

    # Rotation matrix components
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)

    # 4x4 matrix in column-major order (OpenGL convention)
    return [
        r00, r10, r20, 0,
        r01, r11, r21, 0,
        r02, r12, r22, 0,
        0, 0, 0, 1
    ]


def multiply_quaternions(q1: list, q2: list) -> list:
    """Multiply two quaternions q1 * q2. Quaternions are [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def create_transform(position: list, rotation: list) -> dict:
    """Create a SceneEval transform from position and Euler rotation.

    SceneEval's blender_scene.py applies a -90° X rotation as post-multiplication
    for non-SceneWeaver objects (line 389-391). To counteract this, we must
    pre-compose a +90° X rotation into the scene transform.

    Formula: scene_quaternion = q_yaw * q_x90

    Args:
        position: [x, y, z] position
        rotation: [rx, ry, rz] Euler angles in degrees (typically [0, 0, yaw])

    Returns:
        SceneEval transform dict with matrix, quaternion, scale, translation
    """
    rx, ry, rz = rotation

    # Yaw rotation from LayoutVLM
    q_yaw = euler_to_quaternion(rx, ry, rz)

    # +90° X rotation to counteract Blender's -90° X post-rotation
    # q_x90 = [sin(45°), 0, 0, cos(45°)]
    sqrt2_2 = math.sqrt(2) / 2
    q_x90 = [sqrt2_2, 0, 0, sqrt2_2]

    # Compose: scene_quaternion = q_yaw * q_x90
    final_quat = multiply_quaternions(q_yaw, q_x90)

    # Build the rotation matrix
    mat = quaternion_to_matrix(final_quat)

    # Set translation (column 4)
    mat[12] = position[0]
    mat[13] = position[1]
    mat[14] = position[2]
    mat[15] = 1.0

    return {
        "rows": 4,
        "cols": 4,
        "data": mat,
        "rotation": final_quat,
        "scale": [1.0, 1.0, 1.0],
        "translation": position
    }


def parse_sandbox_program(program_path: Path) -> dict:
    """Parse the complete_sandbox_program.py to extract asset metadata.

    Returns a dict mapping asset_var_name to:
        - size: [x, y, z] bounding box
        - description: str
    """
    assets = {}

    with open(program_path, 'r') as f:
        content = f.read()

    # Match patterns like: drawer = Assets(description="...", size=[0.87, 1.52, 0.48], ...)
    pattern = r'(\w+)\s*=\s*Assets\s*\(\s*description\s*=\s*["\']([^"\']+)["\']\s*,\s*size\s*=\s*\[([^\]]+)\]'

    for match in re.finditer(pattern, content):
        var_name = match.group(1)
        description = match.group(2)
        size_str = match.group(3)
        size = [float(x.strip()) for x in size_str.split(',')]

        assets[var_name] = {
            'size': size,
            'description': description
        }

    return assets


def create_arch_elements(boundary: dict, wall_height: float = 3.0) -> list:
    """Create floor and wall architecture elements from boundary."""
    elements = []
    floor_vertices = boundary.get("floor_vertices", [])
    wall_height = boundary.get("wall_height", wall_height)

    if not floor_vertices:
        return elements

    # Floor element
    floor = {
        "depth": 0.05,
        "id": 0,
        "materials": [{"diffuse": "#888899", "name": "surface", "texture": "wood_cream_plane_1375"}],
        "points": [[v[0], v[1], 0] for v in floor_vertices],
        "roomId": "0",
        "type": "Floor"
    }
    elements.append(floor)

    # Wall elements
    wall_ids = []
    for i in range(len(floor_vertices)):
        p1 = floor_vertices[i]
        p2 = floor_vertices[(i + 1) % len(floor_vertices)]

        wall = {
            "depth": 0.025,
            "height": wall_height,
            "holes": [],
            "id": i + 1,
            "materials": [{"diffuse": "#888899", "name": "surface", "texture": "wood_cream_plane_1375"}],
            "points": [[p1[0], p1[1], 0], [p2[0], p2[1], 0]],
            "roomId": "0",
            "type": "Wall"
        }
        elements.append(wall)
        wall_ids.append(i + 1)

    return elements, wall_ids


def convert_single_scene(
    scene_dir: Path,
    target_dir: Path,
    scene_id: int
) -> bool:
    """Convert a single LayoutVLM scene to SceneEval format.

    Args:
        scene_dir: Path to LayoutVLM scene directory (e.g., scene_000/)
        target_dir: Path to SceneEval input directory (e.g., input/LayoutVLM/)
        scene_id: Scene ID for output naming

    Returns:
        True if conversion successful, False otherwise
    """
    scene_json_path = scene_dir / "scene.json"
    layout_json_path = scene_dir / "layout.json"
    program_path = scene_dir / "complete_sandbox_program.py"

    # Check required files exist
    if not scene_json_path.exists():
        print(f"  Warning: {scene_json_path} not found, skipping scene")
        return False

    if not layout_json_path.exists():
        print(f"  Warning: {layout_json_path} not found, skipping scene")
        return False

    # Load scene.json (asset IDs and boundary)
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)

    # Load layout.json (positions and rotations)
    with open(layout_json_path, 'r') as f:
        layout_data = json.load(f)

    # Parse sandbox program for bounding box info
    asset_metadata = {}
    if program_path.exists():
        asset_metadata = parse_sandbox_program(program_path)

    # Build mapping from asset_var_name-idx to objaverse_id
    # scene.json assets are keyed like "25134a84c07e420b93cae181731fd7a0-0"
    # where the number after dash is the index
    objaverse_id_map = {}  # Maps "drawer-0" style to objaverse ID

    # Group assets by their base variable name
    asset_groups = {}  # var_name -> list of (idx, objaverse_id)

    for asset_key in scene_data.get("assets", {}).keys():
        # Split "25134a84c07e420b93cae181731fd7a0-0" into id and index
        parts = asset_key.rsplit("-", 1)
        if len(parts) == 2:
            objaverse_id = parts[0]
            idx = int(parts[1])

            # Store with index
            if objaverse_id not in asset_groups:
                asset_groups[objaverse_id] = []
            asset_groups[objaverse_id].append(idx)

    # Now match layout.json keys to objaverse IDs
    # layout.json has keys like "drawer-0", "desk-0", etc.
    # We need to match these to the objaverse IDs in order

    layout_keys = list(layout_data.keys())
    objaverse_ids = list(scene_data.get("assets", {}).keys())

    # Create the output scene state
    output = json.loads(json.dumps(SCENE_STATE_JSON_BASE))

    # Create architecture from boundary
    boundary = scene_data.get("boundary", {})
    wall_height = boundary.get("wall_height", 3.0)
    elements, wall_ids = create_arch_elements(boundary, wall_height)
    output["scene"]["arch"]["elements"] = elements
    output["scene"]["arch"]["regions"][0]["walls"] = wall_ids

    # Process objects
    # Match layout_keys to objaverse_ids in order
    objects = []

    for i, (layout_key, objaverse_key) in enumerate(zip(layout_keys, objaverse_ids)):
        layout_info = layout_data[layout_key]
        position = layout_info.get("position", [0, 0, 0])
        rotation = layout_info.get("rotation", [0, 0, 0])

        # Extract objaverse ID (remove instance suffix)
        objaverse_id = objaverse_key.rsplit("-", 1)[0]

        # Get var_name from layout_key (e.g., "drawer-0" -> "drawer")
        var_name = layout_key.rsplit("-", 1)[0] if "-" in layout_key else layout_key

        # Create transform
        transform = create_transform(position, rotation)

        obj = {
            "id": str(i + 1),
            "index": i,
            "modelId": f"layoutvlm-objaverse.{objaverse_id}",
            "parentId": "",
            "parentIndex": i,
            "transform": transform
        }
        objects.append(obj)

    output["scene"]["object"] = objects

    # Save output
    output_path = target_dir / f"scene_{scene_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_path}")
    return True


def convert_layoutvlm_run(
    source_dir: Path,
    target_dir: Path,
    mapping: Optional[dict] = None
) -> None:
    """Convert LayoutVLM output to SceneEval format.

    Args:
        source_dir: Path to LayoutVLM results directory
        target_dir: Path to SceneEval input directory (e.g., input/LayoutVLM/)
        mapping: Optional dict mapping source scene index to target scene ID
                 e.g., {"0": 106, "1": 56} means scene_000 -> scene_106
    """
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene directories
    scene_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])

    if not scene_dirs:
        raise ValueError(f"No scene directories found in {source_dir}")

    print(f"Found {len(scene_dirs)} scenes to convert")
    print(f"Source: {source_dir}")
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
        description="Convert LayoutVLM output to SceneEval input format"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to LayoutVLM results directory"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to SceneEval input directory (e.g., input/LayoutVLM)"
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

    convert_layoutvlm_run(args.source_dir, args.target_dir, mapping)


if __name__ == "__main__":
    main()

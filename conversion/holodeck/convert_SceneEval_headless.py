#!/usr/bin/env python3
"""
Headless Holodeck to SceneEval conversion script.

This script converts Holodeck scene JSON files to SceneEval scene state format
WITHOUT requiring Unity. It directly reads position/rotation from the Holodeck
JSON and applies the same coordinate system transformation that Unity does.

Coordinate System Conversion:
- Holodeck/Unity uses Y-up coordinate system
- SceneEval uses Z-up coordinate system
- Position: (x, y, z) → (x, z, y)
- Rotation: Euler(90,0,0) * Euler(x, -(y-180), -z)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from uuid import uuid4
from scipy.spatial.transform import Rotation


# Scene state JSON template
SCENE_STATE_JSON_BASE = {
    "format": "sceneState",
    "scene": {
        "arch": {
            "coords2d": [0, 1],
            "defaults": {
                "Ceiling": {"depth": 0.05},
                "Floor": {"depth": 0.05},
                "Wall": {"depth": 0.1, "extraHeight": 0.035}
            },
            "elements": [],
            "front": [0, 1, 0],
            "holes": [],
            "id": "",
            "images": [],
            "materials": [],
            "regions": [{"id": "bedroom", "type": "Other", "walls": []}],
            "scaleToMeters": 1,
            "textures": [],
            "up": [0, 0, 1],
            "version": "arch@1.0.2"
        },
        "assetSource": ["objaverse"],
        "front": [0, 1, 0],
        "id": "",
        "modifications": [],
        "object": [],
        "unit": 1.0,
        "up": [0, 0, 1],
        "version": "scene@1.0.2"
    },
    "selected": []
}

SCENE_STATE_ARCH_ELEMENT_TEMPLATE = {
    "id": 0,
    "type": "",
    "roomId": "",
    "height": -1,
    "depth": 0.05,
    "points": [[], []],
    "holes": [],
    "materials": [{"diffuse": "#888899", "name": "Walldrywall4Tiled"}]
}


def euler_to_quaternion_unity(euler_x: float, euler_y: float, euler_z: float) -> np.ndarray:
    """
    Convert Euler angles (in degrees) to quaternion using Unity's convention.
    Unity uses ZXY rotation order (intrinsic).

    Returns quaternion as [x, y, z, w].
    """
    # Convert degrees to radians
    euler_rad = np.radians([euler_x, euler_y, euler_z])

    # Unity uses intrinsic ZXY rotation order
    # scipy uses extrinsic by default, so we use 'ZXY' with capital letters for intrinsic
    r = Rotation.from_euler('ZXY', [euler_rad[2], euler_rad[0], euler_rad[1]])

    # scipy returns [x, y, z, w]
    return r.as_quat()


def transform_holodeck_to_sceneeval(position: dict, rotation: dict) -> tuple:
    """
    Transform position and rotation from Holodeck/Unity coordinate system
    to SceneEval coordinate system.

    Based on Unity C# code:
        Vector3 b_pos = new Vector3(child.position.x, child.position.z, child.position.y);
        Quaternion b_rot = Quaternion.Euler(90, 0, 0) * Quaternion.Euler(x, -(y-180), -z);
        Vector3 b_scale = new Vector3(child.localScale.x, child.localScale.z, child.localScale.y);

    Args:
        position: dict with x, y, z keys (Unity Y-up)
        rotation: dict with x, y, z Euler angles in degrees

    Returns:
        (transformed_position, transformed_quaternion, scale)
    """
    # Position: swap Y and Z
    b_pos = np.array([position['x'], position['z'], position['y']])

    # Rotation: Euler(90, 0, 0) * Euler(x, -(y-180), -z)
    # First rotation: 90 degrees around X axis
    r1 = Rotation.from_euler('ZXY', [0, np.radians(90), 0])  # 90 around X

    # Second rotation: Euler(x, -(y-180), -z)
    euler_x = rotation['x']
    euler_y = -(rotation['y'] - 180.0)
    euler_z = -rotation['z']
    r2 = Rotation.from_euler('ZXY', [np.radians(euler_z), np.radians(euler_x), np.radians(euler_y)])

    # Combined rotation (r1 * r2 in Unity is r2 applied first, then r1)
    # In scipy, to match Unity's left multiplication, we do r1 * r2
    b_rot = r1 * r2
    b_quat = b_rot.as_quat()  # [x, y, z, w]

    # Scale: default to 1,1,1 (Holodeck doesn't specify scale)
    b_scale = np.array([1.0, 1.0, 1.0])

    return b_pos, b_quat, b_scale


def build_transform_matrix(position: np.ndarray, quaternion: np.ndarray, scale: np.ndarray) -> list:
    """
    Build a 4x4 transformation matrix from position, quaternion, and scale.
    Returns the matrix data in column-major order (as expected by SceneEval).
    """
    # Build rotation matrix from quaternion
    r = Rotation.from_quat(quaternion)
    rot_matrix = r.as_matrix()

    # Apply scale to rotation matrix columns
    rot_matrix[:, 0] *= scale[0]
    rot_matrix[:, 1] *= scale[1]
    rot_matrix[:, 2] *= scale[2]

    # Build 4x4 matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position

    # Return in column-major order
    return transform.T.flatten().tolist()


def get_arch_info(holodeck_scene: dict) -> list:
    """
    Extract arch element information from a Holodeck scene dict into scene state format.
    """
    windows_info = holodeck_scene.get("windows", [])
    doors_info = holodeck_scene.get("doors", [])

    arch_info = []

    # Process floors
    for floor_info in holodeck_scene.get("rooms", []):
        arch_element = {
            "id": f"floor|{floor_info['id']}",
            "type": "Floor",
            "roomId": floor_info["id"],
            "depth": 0.05,
            "points": [
                [fp["x"], fp.get("z", 0), fp["y"]] for fp in floor_info["floorPolygon"]
            ],
            "materials": [{"diffuse": "#888899", "name": "Walldrywall4Tiled"}]
        }
        arch_info.append(arch_element)

    # Process walls
    for wall_info in holodeck_scene.get("walls", []):
        arch_element = {
            "id": wall_info["id"],
            "type": "Wall",
            "roomId": wall_info.get("roomId", ""),
            "height": wall_info.get("height", 2.8),
            "depth": 0.025,
            "points": [point + [0.0] for point in wall_info["segment"]],
            "materials": [{"diffuse": "#888899", "name": "Walldrywall4Tiled"}]
        }

        # Process holes (doors/windows)
        holes = []
        for opening_info in windows_info + doors_info:
            wall0_id = opening_info.get("wall0", "")
            wall1_id = opening_info.get("wall1", "")

            if arch_element["id"] not in [wall0_id, wall1_id]:
                continue

            # Determine if we need to mirror the coordinates
            mirror_x = (arch_element["id"] == wall1_id)

            p0 = opening_info["holePolygon"][0]
            p1 = opening_info["holePolygon"][1]
            width = wall_info.get("width")

            if mirror_x and width is not None:
                x0 = width - p0["x"]
                x1 = width - p1["x"]
            else:
                x0 = p0["x"]
                x1 = p1["x"]

            y0, y1 = p0["y"], p1["y"]
            min_x, max_x = (x0, x1) if x0 <= x1 else (x1, x0)
            min_y, max_y = (y0, y1) if y0 <= y1 else (y1, y0)

            holes.append({
                "box": {"min": [min_x, min_y], "max": [max_x, max_y]},
                "id": opening_info["id"],
                "type": "Window" if opening_info in windows_info else "Door"
            })

        arch_element["holes"] = holes
        arch_info.append(arch_element)

    return arch_info


def get_object_info(holodeck_scene: dict) -> list:
    """
    Extract and convert object information from a Holodeck scene dict.

    Holodeck stores object positions as the center of the bounding box.
    When Unity loads the scene, it places floor objects so their base sits on y=0.
    We replicate this by setting z=0 for floor objects.

    Object categories:
    - floor_objects: Furniture on the floor -> z=0
    - wall_objects: Items mounted on walls -> keep original z (height)
    - small_objects: Items on furniture -> keep original z (relative to parent)
    """
    objects = holodeck_scene.get("objects", [])

    # Build sets of object IDs by category
    floor_obj_ids = {o['id'] for o in holodeck_scene.get('floor_objects', [])}
    wall_obj_ids = {o['id'] for o in holodeck_scene.get('wall_objects', [])}
    small_obj_ids = {o['id'] for o in holodeck_scene.get('small_objects', [])}

    object_info = []

    for idx, obj in enumerate(objects):
        asset_id = obj.get("assetId", "")
        obj_id = obj.get("id", "")

        # Skip AI2-THOR assets (short asset IDs)
        if len(asset_id) < 32:  # Objaverse UIDs are 32 chars
            print(f"  Skipping AI2-THOR asset: {obj_id}")
            continue

        position = obj.get("position", {"x": 0, "y": 0, "z": 0})
        rotation = obj.get("rotation", {"x": 0, "y": 0, "z": 0})

        # For floor objects, set y=0 (which becomes z=0 after coordinate swap)
        # This places furniture on the floor instead of floating at center height
        if obj_id in floor_obj_ids:
            position = {"x": position["x"], "y": 0, "z": position["z"]}

        # Transform coordinates
        b_pos, b_quat, b_scale = transform_holodeck_to_sceneeval(position, rotation)

        # Build transform matrix
        matrix_data = build_transform_matrix(b_pos, b_quat, b_scale)

        obj_entry = {
            "id": str(idx + 1),
            "modelId": f"objaverse.{asset_id}",
            "index": idx,
            "parentId": "-1",
            "parentIndex": -1,
            "transform": {
                "rows": 4,
                "cols": 4,
                "data": matrix_data
            }
        }

        object_info.append(obj_entry)

    return object_info


def convert_scene(input_json: Path, output_json: Path) -> None:
    """
    Convert a single Holodeck scene to SceneEval format.
    """
    print(f"Converting: {input_json}")

    with open(input_json, 'r') as f:
        holodeck_scene = json.load(f)

    # Extract architecture and object info
    arch_info = get_arch_info(holodeck_scene)
    object_info = get_object_info(holodeck_scene)

    print(f"  Architecture elements: {len(arch_info)}")
    print(f"  Objects: {len(object_info)}")

    # Build scene state
    scene_state = json.loads(json.dumps(SCENE_STATE_JSON_BASE))  # Deep copy
    scene_state["scene"]["arch"]["elements"] = arch_info
    scene_state["scene"]["object"] = object_info

    # Generate unique ID
    uuid = str(uuid4())
    scene_state["scene"]["arch"]["id"] = uuid
    scene_state["scene"]["id"] = uuid

    # Save
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(scene_state, f, indent=2)

    print(f"  Saved to: {output_json}")


def gather_holodeck_scene_jsons(holodeck_dir: Path) -> list:
    """
    Gather all scene JSON files from a Holodeck output directory.
    """
    scene_jsons = []
    for scene_dir in sorted(holodeck_dir.iterdir()):
        if scene_dir.is_dir():
            json_files = list(scene_dir.rglob("*.json"))
            scene_jsons.extend(json_files)
    return scene_jsons


def main():
    parser = argparse.ArgumentParser(description="Convert Holodeck scenes to SceneEval format (no Unity required)")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing Holodeck scene folders")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for SceneEval scene state JSONs")
    parser.add_argument("--mapping", type=str, default=None,
                        help="JSON mapping of source scene index to output scene ID, e.g. '{\"0\": 106, \"1\": 56}'")
    args = parser.parse_args()

    # Parse mapping
    if args.mapping:
        mapping = json.loads(args.mapping)
        mapping = {int(k): int(v) for k, v in mapping.items()}
    else:
        mapping = None

    # Gather all scene JSONs
    scene_jsons = gather_holodeck_scene_jsons(args.input_dir)
    print(f"Found {len(scene_jsons)} scenes to convert")

    # Convert each scene
    for idx, scene_json in enumerate(scene_jsons):
        # Determine output scene ID
        if mapping and idx in mapping:
            scene_id = mapping[idx]
        else:
            # Extract from directory name (e.g., scene_000 -> 0)
            scene_dir_name = scene_json.parent.parent.name
            if scene_dir_name.startswith("scene_"):
                scene_id = int(scene_dir_name.split("_")[1])
            else:
                scene_id = idx

        output_path = args.output_dir / f"scene_{scene_id}.json"
        convert_scene(scene_json, output_path)

    print(f"\nConversion complete! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

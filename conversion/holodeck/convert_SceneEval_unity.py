#!/usr/bin/env python3
"""
Convert Holodeck scenes to SceneEval format using ai2thor with CloudRendering.

This script:
1. Loads each Holodeck scene into ai2thor (headless Unity)
2. Extracts actual object transforms from Unity after scene loading
3. Converts to SceneEval scene state format

Usage:
    cd /home/ubuntu/Holodeck
    source .venv/bin/activate
    python /home/ubuntu/SceneEval/conversion/holodeck/convert_SceneEval_unity.py \
        --input_dir /home/ubuntu/Holodeck/data/scenes_batch \
        --output_dir /home/ubuntu/SceneEval/input/Holodeck \
        --mapping '{"0": 106, "1": 56, "2": 39, "3": 74, "4": 94}'
"""

import json
import argparse
import numpy as np
from pathlib import Path
from uuid import uuid4
from scipy.spatial.transform import Rotation

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR, THOR_COMMIT_ID


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


def transform_unity_to_sceneeval(position: dict, rotation: dict, scale: dict = None) -> tuple:
    """
    Transform position and rotation from Unity coordinate system to SceneEval.

    Based on Unity C# code:
        Vector3 b_pos = new Vector3(child.position.x, child.position.z, child.position.y);
        Quaternion b_rot = Quaternion.Euler(90, 0, 0) * Quaternion.Euler(x, -(y-180), -z);
        Vector3 b_scale = new Vector3(child.localScale.x, child.localScale.z, child.localScale.y);
    """
    # Position: swap Y and Z
    b_pos = np.array([position['x'], position['z'], position['y']])

    # Get rotation as Euler angles (Unity returns rotation as quaternion or Euler)
    if 'x' in rotation and 'y' in rotation and 'z' in rotation:
        # Euler angles case
        euler_x = rotation['x']
        euler_y = rotation['y']
        euler_z = rotation['z']
    else:
        euler_x = euler_y = euler_z = 0

    # Rotation: Euler(90, 0, 0) * Euler(x, -(y-180), -z)
    r1 = Rotation.from_euler('ZXY', [0, np.radians(90), 0])
    euler_y_adjusted = -(euler_y - 180.0)
    euler_z_adjusted = -euler_z
    r2 = Rotation.from_euler('ZXY', [np.radians(euler_z_adjusted), np.radians(euler_x), np.radians(euler_y_adjusted)])
    b_rot = r1 * r2
    b_quat = b_rot.as_quat()  # [x, y, z, w]

    # Scale: swap Y and Z (default to 1,1,1 if not provided)
    if scale:
        b_scale = np.array([scale.get('x', 1.0), scale.get('z', 1.0), scale.get('y', 1.0)])
    else:
        b_scale = np.array([1.0, 1.0, 1.0])

    return b_pos, b_quat, b_scale


def build_transform_matrix(position: np.ndarray, quaternion: np.ndarray, scale: np.ndarray) -> list:
    """Build a 4x4 transformation matrix from position, quaternion, and scale."""
    r = Rotation.from_quat(quaternion)
    rot_matrix = r.as_matrix()

    rot_matrix[:, 0] *= scale[0]
    rot_matrix[:, 1] *= scale[1]
    rot_matrix[:, 2] *= scale[2]

    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position

    return transform.T.flatten().tolist()


def get_arch_info(holodeck_scene: dict) -> list:
    """Extract arch element information from a Holodeck scene dict."""
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


def get_object_info_from_unity(unity_objects: list, holodeck_scene: dict) -> list:
    """
    Extract object info from Unity metadata and convert to SceneEval format.

    Uses ai2thor metadata for transforms. The key insight is:
    - ai2thor's 'position' is the pivot/origin of the object (after physics placement)
    - SceneEval applies the transform matrix to the object's pivot point
    - Therefore we use ai2thor's 'position' (pivot) directly

    This matches the C# PythonUnityBridge.cs which reads child.position (the pivot).

    Args:
        unity_objects: List of object metadata from ai2thor
        holodeck_scene: Original Holodeck scene dict (for reference)
    """
    object_info = []
    idx = 0

    for unity_obj in unity_objects:
        name = unity_obj.get("name", "")
        asset_id = unity_obj.get("assetId", "")

        # Skip non-object entries (doors, windows, room, floor, structural)
        if name.startswith("door") or name.startswith("window"):
            continue
        if name in ["Floor", "Ceiling"]:
            continue
        # Skip room objects (like "kids bedroom")
        if not asset_id:
            continue

        # Skip AI2-THOR assets (short asset IDs) - only keep Objaverse
        if len(asset_id) < 32:
            continue

        # Use ai2thor position (pivot point) - this is what the C# script uses
        position = unity_obj.get("position", {"x": 0, "y": 0, "z": 0})
        rotation = unity_obj.get("rotation", {"x": 0, "y": 0, "z": 0})

        # Transform coordinates from Unity to SceneEval
        b_pos, b_quat, b_scale = transform_unity_to_sceneeval(position, rotation)
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
        idx += 1

    return object_info


def convert_scene(controller: Controller, scene_json_path: Path, output_path: Path) -> bool:
    """Convert a single Holodeck scene using Unity."""
    print(f"Converting: {scene_json_path}")

    # Load Holodeck scene JSON
    with open(scene_json_path, 'r') as f:
        holodeck_scene = json.load(f)

    # Reset to clean Procedural scene before loading new house
    controller.reset(scene="Procedural")

    # Load scene into Unity
    print("  Loading scene into Unity...")
    event = controller.step(action="CreateHouse", house=holodeck_scene)

    if not event.metadata["lastActionSuccess"]:
        print(f"  ERROR: Failed to load scene: {event.metadata.get('errorMessage', 'Unknown error')}")
        return False

    # Get object count from Unity for verification
    unity_objects = event.metadata.get("objects", [])
    print(f"  Unity loaded {len(unity_objects)} objects")

    # Extract architecture info (from original JSON)
    arch_info = get_arch_info(holodeck_scene)

    # Extract object info from ai2thor metadata
    # ai2thor provides the actual Unity transform positions after physics placement
    object_info = get_object_info_from_unity(unity_objects, holodeck_scene)

    print(f"  Architecture elements: {len(arch_info)}")
    print(f"  Converted objects: {len(object_info)}")

    # Build scene state
    scene_state = json.loads(json.dumps(SCENE_STATE_JSON_BASE))
    scene_state["scene"]["arch"]["elements"] = arch_info
    scene_state["scene"]["object"] = object_info

    # Generate unique ID
    uuid = str(uuid4())
    scene_state["scene"]["arch"]["id"] = uuid
    scene_state["scene"]["id"] = uuid

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scene_state, f, indent=2)

    print(f"  Saved to: {output_path}")
    return True


def gather_holodeck_scene_jsons(holodeck_dir: Path) -> list:
    """Gather all scene JSON files from a Holodeck output directory."""
    scene_jsons = []
    for scene_dir in sorted(holodeck_dir.iterdir()):
        if scene_dir.is_dir():
            json_files = list(scene_dir.rglob("*.json"))
            scene_jsons.extend(json_files)
    return scene_jsons


def main():
    parser = argparse.ArgumentParser(description="Convert Holodeck scenes using ai2thor CloudRendering")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing Holodeck scene folders")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for SceneEval scene state JSONs")
    parser.add_argument("--mapping", type=str, default=None,
                        help="JSON mapping of source scene index to output scene ID")
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

    # Create ai2thor controller
    print("Creating ai2thor controller with CloudRendering...")
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        platform=CloudRendering,
        scene="Procedural",
        gridSize=0.25,
        width=300,
        height=300,
        makeAgentsVisible=False,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=OBJATHOR_ASSETS_DIR,
            asset_symlink=True,
            verbose=False,
        ),
    )
    print("Controller created!")

    try:
        # Convert each scene
        for idx, scene_json in enumerate(scene_jsons):
            # Determine output scene ID
            if mapping and idx in mapping:
                scene_id = mapping[idx]
            else:
                scene_dir_name = scene_json.parent.parent.name
                if scene_dir_name.startswith("scene_"):
                    scene_id = int(scene_dir_name.split("_")[1])
                else:
                    scene_id = idx

            output_path = args.output_dir / f"scene_{scene_id}.json"
            convert_scene(controller, scene_json, output_path)

    finally:
        print("Stopping controller...")
        controller.stop()

    print(f"\nConversion complete! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

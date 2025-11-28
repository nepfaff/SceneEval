"""
Asset dataset for scene-agent exported assets.

Scene-agent exports objects with pre-computed SDFs containing physics properties.
Each scene has its own assets directory structure after conversion:
    input/SceneAgent/scene_X/assets/furniture/sdf/{obj_name}_*/obj_name.gltf

Asset IDs follow the format: scene_{scene_id}__{obj_id}
Example: scene_0__workstation_desk_0
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from functools import lru_cache
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset


@register_dataset("scene_agent")
class SceneAgentAssetDataset(BaseAssetDataset):
    """
    Dataset for scene-agent exported assets.

    Scene-agent stores objects as SDFs with GLTF visual geometry.
    The SDF path is stored in sceneeval_state.json for each object.

    Asset ID format: scene_{scene_id}__{obj_id}
    The double underscore (__) separates scene identifier from object name.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the asset dataset.

        Args:
            dataset_config: the configuration for the dataset
        """
        self.asset_id_prefix = dataset_config.asset_id_prefix
        self.root_dir = Path(dataset_config.dataset_root_path).expanduser().resolve()

    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset in format "scene_{scene_id}__{obj_id}"

        Returns:
            AssetInfo object containing the asset's information
        """
        # Parse asset_id: scene_0__workstation_desk_0
        if "__" not in asset_id:
            raise ValueError(
                f"Invalid SceneAgent asset ID format: {asset_id}. "
                f"Expected format: scene_{{scene_id}}__{{obj_id}}"
            )

        scene_part, obj_id = asset_id.split("__", 1)

        # Extract scene_id from scene_X
        if not scene_part.startswith("scene_"):
            raise ValueError(
                f"Invalid scene identifier: {scene_part}. "
                f"Expected format: scene_{{scene_id}}"
            )

        scene_id = scene_part[6:]  # Remove "scene_" prefix

        # Load scene JSON to get sdfPath for this object
        scene_json_path = self.root_dir / f"scene_{scene_id}.json"
        scene_data = self._load_scene_json(scene_json_path)

        # Find object and get GLTF path from SDF
        for obj in scene_data.get("scene", {}).get("object", []):
            if obj.get("id") == obj_id:
                sdf_relative_path = obj.get("sdfPath")
                if not sdf_relative_path:
                    raise FileNotFoundError(
                        f"Object {obj_id} has no sdfPath in scene JSON"
                    )

                # Build full SDF path: root_dir/scene_X/assets/{sdf_relative_path}
                sdf_path = self.root_dir / f"scene_{scene_id}" / "assets" / sdf_relative_path

                if not sdf_path.exists():
                    raise FileNotFoundError(
                        f"SDF file not found: {sdf_path}. "
                        f"Make sure the scene-agent output has been converted."
                    )

                # Extract GLTF path from SDF
                gltf_path = self._extract_gltf_from_sdf(sdf_path)

                return AssetInfo(
                    asset_id=asset_id,
                    file_path=gltf_path,
                    description=f"{obj_id} (scene-agent)",
                    extra_rotation_transform=None  # No extra rotation needed
                )

        raise FileNotFoundError(
            f"Object {obj_id} not found in scene_{scene_id}.json"
        )

    @lru_cache(maxsize=32)
    def _load_scene_json(self, scene_json_path: Path) -> dict:
        """
        Load and cache scene JSON file.

        Args:
            scene_json_path: Path to scene JSON file

        Returns:
            Parsed JSON data
        """
        if not scene_json_path.exists():
            raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

        with open(scene_json_path, "r") as f:
            return json.load(f)

    def _extract_gltf_from_sdf(self, sdf_path: Path) -> Path:
        """
        Parse SDF XML to get GLTF path from <visual><geometry><mesh><uri>.

        Args:
            sdf_path: Path to SDF file

        Returns:
            Absolute path to GLTF file
        """
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        # Find the visual mesh URI
        # Structure: <sdf><model><link><visual><geometry><mesh><uri>
        for visual in root.iter("visual"):
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    uri = mesh.find("uri")
                    if uri is not None and uri.text:
                        # URI is relative to SDF location
                        gltf_path = sdf_path.parent / uri.text

                        if not gltf_path.exists():
                            raise FileNotFoundError(
                                f"GLTF file not found: {gltf_path}"
                            )

                        return gltf_path

        raise ValueError(
            f"Could not find visual mesh URI in SDF: {sdf_path}"
        )

"""
Asset dataset for IDesign exported assets.

IDesign stores objects as GLB files retrieved from Objaverse.
Each scene has its own assets directory after conversion:
    input/IDesign/scene_X/assets/{obj_id}.glb

Asset IDs follow the format: scene_{scene_id}__{obj_id}
Example: scene_0__twin_bed_1
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from .base import AssetInfo, BaseAssetDataset, DatasetConfig
from .registry import register_dataset


@register_dataset("idesign")
class IDesignAssetDataset(BaseAssetDataset):
    """
    Dataset for IDesign exported assets.

    IDesign stores objects as GLB files in per-scene directories.
    The assets are retrieved from Objaverse based on text descriptions.

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
        # Parse asset_id: scene_0__twin_bed_1 -> scene_id=0, obj_id=twin_bed_1
        if "__" not in asset_id:
            raise ValueError(
                f"Invalid IDesign asset ID format: {asset_id}. "
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

        # GLB path: root_dir/scene_N/assets/{obj_id}.glb
        glb_path = self.root_dir / f"scene_{scene_id}" / "assets" / f"{obj_id}.glb"

        if not glb_path.exists():
            raise FileNotFoundError(
                f"Asset not found: {glb_path}. "
                f"Make sure the IDesign scene has been converted."
            )

        # Get description from scene JSON
        description = self._get_description(scene_id, obj_id)

        return AssetInfo(
            asset_id=asset_id,
            file_path=glb_path,
            description=description,
            extra_rotation_transform=None  # IDesign uses Z-up like SceneEval
        )

    @lru_cache(maxsize=32)
    def _load_scene_json(self, scene_id: str) -> dict:
        """
        Load and cache scene JSON file.

        Args:
            scene_id: Scene ID (e.g., "0", "1")

        Returns:
            Parsed JSON data
        """
        scene_json_path = self.root_dir / f"scene_{scene_id}.json"

        if not scene_json_path.exists():
            return {}

        with open(scene_json_path, "r") as f:
            return json.load(f)

    def _get_description(self, scene_id: str, obj_id: str) -> str:
        """
        Get description from scene JSON metadata.

        Args:
            scene_id: Scene ID
            obj_id: Object ID (e.g., "twin_bed_1")

        Returns:
            Description string for VLM matching
        """
        scene_data = self._load_scene_json(scene_id)

        # Find object in scene JSON
        for obj in scene_data.get("scene", {}).get("object", []):
            if obj.get("id") == obj_id:
                desc = obj.get("description")
                if desc:
                    return desc

        # Fallback: humanize obj_id
        return self._humanize_object_id(obj_id)

    @staticmethod
    def _humanize_object_id(obj_id: str) -> str:
        """
        Convert object ID to human-readable description.

        "twin_bed_1" -> "twin bed"
        "bean_bag_chair_1" -> "bean bag chair"
        """
        # Remove trailing numbers
        cleaned = re.sub(r'_\d+$', '', obj_id)
        # Replace underscores with spaces
        return cleaned.replace('_', ' ')

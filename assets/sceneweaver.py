"""
Asset dataset for SceneWeaver exported assets.

SceneWeaver exports procedurally generated objects as GLB files during conversion.
Each scene has its own assets directory structure:
    input/SceneWeaver/scene_X/assets/{obj_name}.glb

Asset IDs follow the format: scene_{scene_id}__{obj_name}
Example: scene_0__5348940_BedFactory
"""

from pathlib import Path
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset


@register_dataset("sceneweaver")
@register_dataset("sw")  # Short alias to avoid Blender's 63-char name limit
class SceneWeaverAssetDataset(BaseAssetDataset):
    """
    Dataset for SceneWeaver exported assets.

    SceneWeaver uses procedural generation (Infinigen) for objects like beds, wardrobes, etc.
    During conversion, objects are exported from Blender as individual GLB files.

    Asset ID format: scene_{scene_id}__{obj_name}
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
            asset_id: the ID of the asset in format "scene_{scene_id}__{obj_name}"

        Returns:
            AssetInfo object containing the asset's information
        """
        # Parse asset_id: scene_0__5348940_BedFactory or s0__5348940_BedFactory (short form)
        if "__" not in asset_id:
            raise ValueError(
                f"Invalid SceneWeaver asset ID format: {asset_id}. "
                f"Expected format: scene_{{scene_id}}__{{obj_name}} or s{{scene_id}}__{{obj_name}}"
            )

        scene_part, obj_name = asset_id.split("__", 1)

        # Extract scene_id from scene_X or sX (short form)
        if scene_part.startswith("scene_"):
            scene_id = scene_part[6:]  # Remove "scene_" prefix
        elif scene_part.startswith("s"):
            scene_id = scene_part[1:]  # Remove "s" prefix (short form)
        else:
            raise ValueError(
                f"Invalid scene identifier: {scene_part}. "
                f"Expected format: scene_{{scene_id}} or s{{scene_id}}"
            )

        # Build file path: root_dir/scene_X/assets/{obj_name}.glb
        file_path = self.root_dir / f"scene_{scene_id}" / "assets" / f"{obj_name}.glb"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Asset file not found: {file_path}. "
                f"Make sure the SceneWeaver scene has been converted."
            )

        # Extract category from object name for description
        category = self._extract_category(obj_name)

        return AssetInfo(
            asset_id=asset_id,
            file_path=file_path,
            description=f"{category} (SceneWeaver procedural object)",
            extra_rotation_transform=None
        )

    def _extract_category(self, obj_name: str) -> str:
        """
        Extract category from SceneWeaver object name.

        Args:
            obj_name: Object name like "5348940_BedFactory" or "9391942_wardrobe"

        Returns:
            Category string (e.g., "bed", "wardrobe")
        """
        parts = obj_name.split("_")
        if len(parts) > 1:
            # Handle factory names like "BedFactory", "SimpleBookcaseFactory"
            factory = parts[-1]
            if factory.endswith("Factory"):
                factory = factory[:-7]  # Remove "Factory" suffix

            # Convert camelCase to lowercase
            result = []
            for char in factory:
                if char.isupper() and result:
                    result.append(" ")
                result.append(char.lower())

            return "".join(result).strip()

        return "object"

import json
import logging
from pathlib import Path
from .base import BaseAssetDataset, DatasetConfig, AssetInfo
from .registry import register_dataset

logger = logging.getLogger(__name__)

@register_dataset("layoutvlm_objathor")
class LayoutVLMObjathorAssetDataset(BaseAssetDataset):
    """
    Dataset for LayoutVLM-Objathor assets.

    Supports two asset sources:
    - layoutvlm-objathor: Curated 675 assets with per-asset data.json
    - objathor-assets: Full 50K Objaverse assets with central annotations.json

    Falls back to objathor-assets if asset not found in layoutvlm-objathor.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """
        Initialize the asset dataset.

        Args:
            dataset_config: the configuration for the dataset
        """

        self.asset_id_prefix = dataset_config.asset_id_prefix
        self.root_dir = Path(dataset_config.dataset_root_path).expanduser().resolve()

        # Fallback to full objathor-assets
        self.fallback_dir = self.root_dir.parent / "objathor-assets"
        self.fallback_metadata = None
        fallback_metadata_path = self.fallback_dir / "annotations.json"
        if fallback_metadata_path.exists():
            with open(fallback_metadata_path, "r") as f:
                self.fallback_metadata = json.load(f)
    
    def get_asset_info(self, asset_id: str) -> AssetInfo:
        """
        Get information about the asset.

        Args:
            asset_id: the ID of the asset

        Returns:
            AssetInfo object containing the asset's information
        """

        # Try layoutvlm-objathor first
        asset_data_json_path = self.root_dir / asset_id / "data.json"

        if asset_data_json_path.exists():
            file_path = self.root_dir / asset_id / f"{asset_id}.glb"
            with open(asset_data_json_path, "r") as f:
                asset_data_json = json.load(f)
            metadata = asset_data_json["annotations"]
            asset_description = f"{metadata['category']}, {metadata['description']}, {metadata['materials']}"

        # Fall back to objathor-assets
        elif self.fallback_metadata and asset_id in self.fallback_metadata:
            logger.warning(f"Asset {asset_id} not found in layoutvlm-objathor, falling back to objathor-assets")
            file_path = self.fallback_dir / asset_id / f"{asset_id}.glb"
            if not file_path.exists():
                raise FileNotFoundError(f"Asset GLB {file_path} not found.")
            metadata = self.fallback_metadata[asset_id]
            asset_description = metadata.get("category", "")
            if metadata.get("ref_category"):
                asset_description += f" - {metadata['ref_category']}"
            if metadata.get("description"):
                asset_description += f", {metadata['description']}"
            elif metadata.get("description_auto"):
                asset_description += f", {metadata['description_auto']}"

        else:
            raise FileNotFoundError(f"Asset {asset_id} not found in layoutvlm-objathor or objathor-assets.")

        return AssetInfo(
            asset_id=asset_id,
            file_path=file_path,
            description=asset_description,
            extra_rotation_transform=None
        )

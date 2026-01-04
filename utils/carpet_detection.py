"""Carpet/rug detection utility for SceneEval.

Carpets and rugs should be excluded from collision and equilibrium metrics
because they are flat, flexible objects that don't meaningfully participate
in physics collision.

Detection patterns by method:
- SceneAgent: Check SDF file for missing <collision> geometry (thin coverings)
- SceneWeaver: factory pattern like "*_RugFactory_*"
- IDesign: ID like "area_rug_1" or description contains "rug"
- Holodeck: object_name contains "rug" like "dining_area_rug-0"
- LayoutVLM_curated: asset key contains "carpet" like "carpet-0"
- HSM: no rug objects
- LayoutVLM_objaverse: no rug objects
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Carpet keywords - must appear as whole words or with common separators
# Note: "mat" is excluded as a standalone keyword because it's too common
# (e.g., "mattress", "material"). We only match "doormat" or "mat_" patterns.
CARPET_KEYWORDS = ["rug", "carpet", "doormat"]


def has_collision_geometry(sdf_path: str | Path) -> bool:
    """Check if an SDF file has collision geometry.

    Args:
        sdf_path: Path to the SDF file

    Returns:
        True if the SDF file has <collision> elements, False otherwise.
    """
    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        logger.warning(f"SDF file not found: {sdf_path}")
        return True  # Assume has collision if file not found

    try:
        content = sdf_path.read_text()
        return "<collision" in content
    except Exception as e:
        logger.warning(f"Error reading SDF file {sdf_path}: {e}")
        return True  # Assume has collision on error


def is_thin_covering_sceneagent(sdf_path: str | Path) -> bool:
    """Check if a SceneAgent object is a thin covering without collision geometry.

    Thin coverings (rugs, floor mats, etc.) in SceneAgent have SDF files with
    only <visual> elements and no <collision> elements. Wall-mounted thin
    coverings (pictures, posters) DO have collision geometry.

    Args:
        sdf_path: Path to the object's SDF file

    Returns:
        True if the object is a thin covering that should be excluded from
        collision checks.
    """
    return not has_collision_geometry(sdf_path)


def is_carpet_object(obj_id: str, description: str = "", sdf_path: str | Path = None) -> bool:
    """Check if object is a carpet/rug that should be excluded from collision checks.

    Args:
        obj_id: The object identifier (e.g., "idx5_scene-agent.rug_0")
        description: Optional object description (used for IDesign)
        sdf_path: Optional path to SDF file (used for SceneAgent collision check)

    Returns:
        True if the object is a carpet/rug that should be excluded.
    """
    id_lower = obj_id.lower()

    # For SceneAgent objects, check if SDF has collision geometry
    # Objects without collision geometry are thin coverings (rugs, floor mats)
    if "scene-agent" in id_lower and sdf_path is not None:
        if is_thin_covering_sceneagent(sdf_path):
            return True

    # Check for SceneWeaver factory patterns (e.g., "4061705_RugFactory")
    # This is a specific pattern that's unambiguous
    if "rugfactory" in id_lower:
        return True

    # Check object ID patterns with word boundary awareness
    # Use regex to match keywords as whole words or with common separators
    for keyword in CARPET_KEYWORDS:
        # Match keyword surrounded by non-alphanumeric chars or at start/end
        # This prevents matching "mattress" when looking for "mat"
        pattern = rf'(^|[^a-z]){keyword}([^a-z]|$)'
        if re.search(pattern, id_lower):
            return True

    # Check description (primarily for IDesign)
    if description:
        desc_lower = description.lower()
        for keyword in CARPET_KEYWORDS:
            pattern = rf'(^|[^a-z]){keyword}([^a-z]|$)'
            if re.search(pattern, desc_lower):
                return True

    return False

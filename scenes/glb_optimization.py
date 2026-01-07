"""GLB optimization utilities for web export.

Note: Most optimization is now handled by gltf-transform (see scenes/gltf_transform.py).
This module only contains Blender-specific pre-processing that must happen before export.
"""

import logging

import bpy

logger = logging.getLogger(__name__)


def hide_placeholder_objects() -> list[str]:
    """Hide SceneWeaver placeholder/bbox objects from export.

    Placeholders are bounding box representations of failed asset spawns
    and should not be included in final GLB exports.

    Returns:
        List of hidden object names.
    """
    hidden = []
    for obj in bpy.data.objects:
        name_lower = obj.name.lower()
        if 'placeholder' in name_lower or 'bbox' in name_lower:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden.append(obj.name)
    if hidden:
        logger.info(f"Hidden {len(hidden)} placeholder objects from GLB export")
    return hidden

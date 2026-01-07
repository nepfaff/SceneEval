"""GLB optimization utilities for web export.

Note: Most optimization is now handled by gltf-transform (see scenes/gltf_transform.py).
This module only contains Blender-specific pre-processing that must happen before export.
"""

import logging
from collections import defaultdict

import bpy
import numpy as np

logger = logging.getLogger(__name__)


def get_object_diagonal(obj: bpy.types.Object) -> float:
    """Get the diagonal length of an object's bounding box in meters.

    Args:
        obj: Blender object.

    Returns:
        Diagonal length in meters.
    """
    if obj.type != "MESH":
        return 0.0

    try:
        dims = obj.dimensions
        return float(np.sqrt(dims.x**2 + dims.y**2 + dims.z**2))
    except Exception:
        return 1.0


def get_target_texture_size(diagonal: float, max_size: int = 512) -> int:
    """Determine target texture size based on object diagonal.

    Args:
        diagonal: Object bounding box diagonal in meters.
        max_size: Maximum allowed texture size.

    Returns:
        Target texture size (128, 256, or 512).
    """
    if diagonal < 0.3:  # Small objects (<30cm): cups, books, small decor
        return min(128, max_size)
    elif diagonal < 1.0:  # Medium objects (30cm-1m): chairs, lamps
        return min(256, max_size)
    else:  # Large objects (>1m): sofas, beds, walls
        return min(512, max_size)


def resize_textures_adaptive(max_size: int = 512) -> dict:
    """Resize textures based on the size of objects using them.

    For each texture, finds the largest object using it and resizes
    the texture based on that object's bounding box diagonal:
    - Small objects (<30cm): 128px
    - Medium objects (30cm-1m): 256px
    - Large objects (>1m): 512px (capped at max_size)

    Args:
        max_size: Maximum texture size (default 512).

    Returns:
        Dict with resize statistics.
    """
    stats = {
        "textures_resized": 0,
        "textures_skipped": 0,
        "size_mapping": {},  # image_name -> (original_size, new_size, largest_object)
    }

    # Step 1: Build mapping from image -> list of (object, diagonal)
    image_to_objects: dict[bpy.types.Image, list[tuple[str, float]]] = defaultdict(list)

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render or obj.hide_viewport:
            continue

        diagonal = get_object_diagonal(obj)

        # Find all images used by this object's materials
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes or not mat.node_tree:
                continue

            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    image_to_objects[node.image].append((obj.name, diagonal))

    # Step 2: For each image, find largest object and resize accordingly
    for img, objects in image_to_objects.items():
        if not objects:
            continue

        # Skip non-file images
        if img.source != "FILE" and img.packed_file is None:
            stats["textures_skipped"] += 1
            continue

        # Skip images with zero size
        if img.size[0] == 0 or img.size[1] == 0:
            stats["textures_skipped"] += 1
            continue

        # Find largest object using this texture
        largest_obj_name, largest_diagonal = max(objects, key=lambda x: x[1])
        target_size = get_target_texture_size(largest_diagonal, max_size)

        # Get current size
        current_max = max(img.size)
        original_size = (img.size[0], img.size[1])

        # Only resize if current size is larger than target
        if current_max > target_size:
            scale = target_size / current_max
            new_width = max(1, int(img.size[0] * scale))
            new_height = max(1, int(img.size[1] * scale))

            try:
                img.scale(new_width, new_height)
                stats["textures_resized"] += 1
                stats["size_mapping"][img.name] = {
                    "original": original_size,
                    "new": (new_width, new_height),
                    "largest_object": largest_obj_name,
                    "object_diagonal_m": round(largest_diagonal, 2),
                }
                logger.debug(
                    f"Resized {img.name}: {original_size} -> ({new_width}, {new_height}) "
                    f"(largest obj: {largest_obj_name}, {largest_diagonal:.2f}m)"
                )
            except Exception as e:
                logger.warning(f"Failed to resize {img.name}: {e}")
                stats["textures_skipped"] += 1
        else:
            stats["textures_skipped"] += 1

    logger.info(
        f"Adaptive texture resize: {stats['textures_resized']} resized, "
        f"{stats['textures_skipped']} skipped"
    )
    return stats


def merge_vertices_by_distance(distance: float = 0.0001, min_verts: int = 100) -> dict:
    """Merge duplicate vertices by distance to enable mesh simplification.

    Scene-agent and some other pipelines export meshes with split vertices
    (every triangle has unique vertices, ~3.0 verts/tri ratio). This prevents
    mesh simplification from working. Merging vertices first enables proper
    simplification.

    Also sets smooth shading on all faces for better normal interpolation.

    Args:
        distance: Merge distance in meters (default 0.1mm).
        min_verts: Minimum vertices to process (skip small meshes).

    Returns:
        Dict with merge statistics.
    """
    import bmesh

    stats = {
        "meshes_processed": 0,
        "vertices_before": 0,
        "vertices_after": 0,
        "meshes_skipped": 0,
    }

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render or obj.hide_viewport:
            continue

        mesh = obj.data
        orig_verts = len(mesh.vertices)

        if orig_verts < min_verts:
            stats["meshes_skipped"] += 1
            continue

        stats["vertices_before"] += orig_verts

        # Set smooth shading on all faces
        for poly in mesh.polygons:
            poly.use_smooth = True

        # Merge duplicate vertices
        try:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
            bm.to_mesh(mesh)
            bm.free()
            stats["meshes_processed"] += 1
            stats["vertices_after"] += len(mesh.vertices)
        except Exception as e:
            logger.warning(f"Failed to merge vertices on {obj.name}: {e}")
            stats["vertices_after"] += orig_verts

    if stats["vertices_before"] > 0:
        reduction = (1 - stats["vertices_after"] / stats["vertices_before"]) * 100
        logger.info(
            f"Vertex merge: {stats['meshes_processed']} meshes, "
            f"{stats['vertices_before']:,} -> {stats['vertices_after']:,} verts "
            f"({reduction:.0f}% reduction)"
        )

    return stats


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

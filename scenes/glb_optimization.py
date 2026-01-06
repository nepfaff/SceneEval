"""GLB optimization utilities for web export.

Optimizes Blender scenes for web viewing with smaller file sizes:
- Vertex deduplication
- Mesh decimation
- Mesh deduplication (linking identical meshes)
- Material deduplication
- Texture resizing and JPEG conversion
- Hierarchy flattening
- Orphan data cleanup
"""

import hashlib
import logging
import bpy
import bmesh
import mathutils
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def get_bounding_box_diagonal(obj: bpy.types.Object) -> float:
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
        return np.sqrt(dims.x**2 + dims.y**2 + dims.z**2)
    except Exception:
        return 1.0


def optimize_mesh_geometry(obj: bpy.types.Object, max_decimation_ratio: float = 0.5) -> dict:
    """Optimize mesh by merging duplicate vertices and decimating.

    Args:
        obj: Blender mesh object.
        max_decimation_ratio: Maximum decimation (0.5 = reduce to 50% of faces).

    Returns:
        Stats dict with original/final vertex/face counts.
    """
    if obj.type != "MESH":
        return {}

    mesh = obj.data
    original_verts = len(mesh.vertices)
    original_faces = len(mesh.polygons)

    # Step 1: Merge duplicate vertices (0.1mm threshold)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        before_verts = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        bm.to_mesh(mesh)
    finally:
        bm.free()

    mesh.update()
    after_merge_verts = len(mesh.vertices)
    after_merge_faces = len(mesh.polygons)

    # Step 2: Decimation for high-poly meshes
    final_faces = after_merge_faces
    if after_merge_faces > 5000:
        # Calculate ratio to achieve target
        target_faces = max(1000, int(after_merge_faces * max_decimation_ratio))
        ratio = target_faces / after_merge_faces

        # Apply decimation modifier
        modifier = obj.modifiers.new(name="WebDecimate", type="DECIMATE")
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True

        # Apply the modifier
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.modifier_apply(modifier="WebDecimate")
            final_faces = len(mesh.polygons)
        except Exception as e:
            logger.warning(f"Failed to apply decimation to {obj.name}: {e}")
            # Remove modifier if apply failed
            obj.modifiers.remove(modifier)

    return {
        "object": obj.name,
        "original_verts": original_verts,
        "original_faces": original_faces,
        "after_merge_verts": after_merge_verts,
        "after_merge_faces": after_merge_faces,
        "final_faces": final_faces,
    }


def hash_mesh_geometry(mesh: bpy.types.Mesh) -> str:
    """Create a hash of mesh geometry for deduplication.

    Args:
        mesh: Blender mesh data.

    Returns:
        MD5 hash string of vertex positions and face indices.
    """
    try:
        # Get vertex positions (rounded for tolerance)
        verts = np.array([v.co[:] for v in mesh.vertices])
        verts_rounded = np.round(verts, decimals=4)

        # Get face vertex indices
        faces = []
        for poly in mesh.polygons:
            faces.append(tuple(sorted(poly.vertices)))

        # Create hash
        data = str(verts_rounded.tobytes()) + str(sorted(faces))
        return hashlib.md5(data.encode()).hexdigest()
    except Exception:
        return str(id(mesh))


def deduplicate_meshes() -> int:
    """Link duplicate meshes to share geometry data.

    Returns:
        Number of meshes deduplicated.
    """
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not mesh_objects:
        return 0

    # Group by geometry hash
    mesh_groups: dict[str, list[bpy.types.Object]] = {}
    for obj in mesh_objects:
        mesh_hash = hash_mesh_geometry(obj.data)
        mesh_groups.setdefault(mesh_hash, []).append(obj)

    # Link duplicates to master mesh
    dedup_count = 0
    for objects in mesh_groups.values():
        if len(objects) > 1:
            master_mesh = objects[0].data
            for obj in objects[1:]:
                if obj.data != master_mesh:
                    old_mesh = obj.data
                    obj.data = master_mesh
                    # Remove old mesh if no longer used
                    if old_mesh.users == 0:
                        bpy.data.meshes.remove(old_mesh)
                    dedup_count += 1

    logger.info(f"Deduplicated {dedup_count} meshes")
    return dedup_count


def hash_material_properties(mat: bpy.types.Material) -> str:
    """Hash material by PBR properties and texture paths.

    Args:
        mat: Blender material.

    Returns:
        MD5 hash string of material properties.
    """
    props = []

    if not mat.use_nodes or not mat.node_tree:
        return str(id(mat))

    try:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Get base color
            base_color = bsdf.inputs.get("Base Color")
            if base_color:
                props.append(tuple(round(v, 3) for v in base_color.default_value))

            # Get roughness
            roughness = bsdf.inputs.get("Roughness")
            if roughness:
                props.append(round(roughness.default_value, 3))

            # Get metallic
            metallic = bsdf.inputs.get("Metallic")
            if metallic:
                props.append(round(metallic.default_value, 3))

        # Add texture paths
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                props.append(node.image.filepath or node.image.name)

        return hashlib.md5(str(props).encode()).hexdigest()
    except Exception:
        return str(id(mat))


def deduplicate_materials() -> int:
    """Link identical materials to reduce unique material count.

    Returns:
        Number of materials deduplicated.
    """
    # Group by property hash
    material_groups: dict[str, list[bpy.types.Material]] = {}

    for mat in bpy.data.materials:
        if not mat.users:
            continue
        mat_hash = hash_material_properties(mat)
        material_groups.setdefault(mat_hash, []).append(mat)

    # Replace duplicates with master material
    dedup_count = 0
    for materials in material_groups.values():
        if len(materials) > 1:
            master = materials[0]
            for mat in materials[1:]:
                # Find all objects using this material and replace
                for obj in bpy.data.objects:
                    if not hasattr(obj, "material_slots"):
                        continue
                    for slot in obj.material_slots:
                        if slot.material == mat:
                            slot.material = master
                dedup_count += 1

    logger.info(f"Deduplicated {dedup_count} materials")
    return dedup_count


def compute_adaptive_texture_size(obj: bpy.types.Object, max_size: int = 512) -> int:
    """Compute texture size based on object dimensions.

    Args:
        obj: Blender object.
        max_size: Maximum texture size.

    Returns:
        Recommended texture size (128, 256, or 512).
    """
    diagonal = get_bounding_box_diagonal(obj)
    if diagonal < 0.3:
        return 128  # Small (<30cm): cups, books, decor
    elif diagonal < 1.0:
        return 256  # Medium: chairs, lamps
    else:
        return min(512, max_size)  # Large: sofas, beds, tables


def resize_textures(max_size: int = 512) -> int:
    """Resize textures based on object size.

    Args:
        max_size: Maximum texture dimension.

    Returns:
        Number of textures resized.
    """
    resize_count = 0

    for img in bpy.data.images:
        if img.size[0] == 0 or img.size[1] == 0:
            continue
        if img.source != "FILE":
            continue

        current_max = max(img.size)
        if current_max > max_size:
            # Calculate new size maintaining aspect ratio
            scale = max_size / current_max
            new_width = int(img.size[0] * scale)
            new_height = int(img.size[1] * scale)

            try:
                img.scale(new_width, new_height)
                resize_count += 1
            except Exception as e:
                logger.warning(f"Failed to resize image {img.name}: {e}")

    logger.info(f"Resized {resize_count} textures to max {max_size}px")
    return resize_count


def convert_textures_to_jpeg(quality: int = 85) -> int:
    """Convert PNG textures to JPEG where alpha is not used.

    Args:
        quality: JPEG quality (1-100).

    Returns:
        Number of textures converted.
    """
    convert_count = 0

    for img in bpy.data.images:
        if img.file_format != "PNG":
            continue
        if img.channels > 3:  # Has alpha channel
            continue
        if img.source != "FILE":
            continue

        try:
            img.file_format = "JPEG"
            convert_count += 1
        except Exception as e:
            logger.warning(f"Failed to convert image {img.name} to JPEG: {e}")

    logger.info(f"Converted {convert_count} textures to JPEG")
    return convert_count


def remove_unused_data() -> dict:
    """Remove orphaned meshes, materials, textures, etc.

    Returns:
        Dict with counts of removed items per data type.
    """
    stats = {}

    for attr in ["meshes", "materials", "images", "textures", "node_groups"]:
        data_block = getattr(bpy.data, attr, None)
        if data_block is None:
            continue

        removed = 0
        for item in list(data_block):
            if item.users == 0:
                try:
                    data_block.remove(item)
                    removed += 1
                except Exception:
                    pass
        stats[attr] = removed

    logger.info(f"Removed unused data: {stats}")
    return stats


def flatten_hierarchy(max_depth: int = 2) -> int:
    """Flatten deep object hierarchies.

    Args:
        max_depth: Maximum allowed parent depth.

    Returns:
        Number of objects unparented.
    """
    unparented = 0

    for obj in bpy.data.objects:
        # Count depth
        depth = 0
        parent = obj.parent
        while parent:
            depth += 1
            parent = parent.parent

        if depth > max_depth:
            # Store world matrix before unparenting
            world_matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world_matrix
            unparented += 1

    logger.info(f"Flattened hierarchy: unparented {unparented} objects")
    return unparented


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


def optimize_scene_for_web(
    max_texture_size: int = 512,
    max_decimation_ratio: float = 0.5,
    convert_to_jpeg: bool = True,
) -> dict:
    """Apply all optimizations to prepare scene for web GLB export.

    Args:
        max_texture_size: Maximum texture dimension in pixels.
        max_decimation_ratio: Maximum face reduction ratio.
        convert_to_jpeg: Whether to convert PNG to JPEG.

    Returns:
        Dict with optimization statistics.
    """
    stats = {}

    # 0. Hide placeholder objects (SceneWeaver bboxes)
    hidden_placeholders = hide_placeholder_objects()
    stats["hidden_placeholders"] = hidden_placeholders

    # 1. Optimize mesh geometry (vertex merge + decimation)
    mesh_stats = []
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    for obj in mesh_objects:
        result = optimize_mesh_geometry(obj, max_decimation_ratio)
        if result:
            mesh_stats.append(result)
    stats["mesh_optimization"] = mesh_stats

    # 2. Deduplicate meshes
    stats["meshes_deduplicated"] = deduplicate_meshes()

    # 3. Deduplicate materials
    stats["materials_deduplicated"] = deduplicate_materials()

    # 4. Flatten hierarchy
    stats["objects_unparented"] = flatten_hierarchy(max_depth=2)

    # 5. Remove unused data
    stats["unused_removed"] = remove_unused_data()

    # 6. Resize textures
    stats["textures_resized"] = resize_textures(max_texture_size)

    # 7. Convert to JPEG
    if convert_to_jpeg:
        stats["textures_converted_jpeg"] = convert_textures_to_jpeg(quality=85)

    return stats


def center_scene_at_origin() -> tuple[float, float, float]:
    """Move the scene so its bounding box center is at the world origin.

    This improves navigation in GLB viewers by ensuring the camera orbits
    around the scene center rather than a potentially offset point.

    Returns:
        The offset that was applied (original center position).
    """
    # Find all visible mesh objects
    visible_objects = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.visible_get() and not obj.hide_render
    ]

    if not visible_objects:
        logger.warning("No visible mesh objects to center")
        return (0.0, 0.0, 0.0)

    # Calculate combined bounding box
    min_corner = [float("inf")] * 3
    max_corner = [float("-inf")] * 3

    for obj in visible_objects:
        # Get world-space bounding box corners
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        for corner in bbox_corners:
            for i in range(3):
                min_corner[i] = min(min_corner[i], corner[i])
                max_corner[i] = max(max_corner[i], corner[i])

    # Calculate center
    center = mathutils.Vector([
        (min_corner[i] + max_corner[i]) / 2
        for i in range(3)
    ])

    if center.length < 0.001:
        logger.info("Scene already centered at origin")
        return (0.0, 0.0, 0.0)

    # Move all root-level objects (those without parents)
    root_objects = [obj for obj in bpy.data.objects if obj.parent is None]
    for obj in root_objects:
        obj.location -= center

    logger.info(f"Centered scene at origin (offset: {center.x:.2f}, {center.y:.2f}, {center.z:.2f})")
    return (center.x, center.y, center.z)


def export_optimized_glb(
    output_path: Path,
    use_draco: bool = True,
    draco_compression_level: int = 6,
    center_scene: bool = True,
) -> bool:
    """Export scene as optimized GLB file.

    Args:
        output_path: Output file path.
        use_draco: Enable Draco mesh compression.
        draco_compression_level: Draco compression level (0-10).
        center_scene: Center scene at world origin before export.

    Returns:
        True if export succeeded.
    """
    try:
        # Center scene at origin for better navigation in GLB viewers
        if center_scene:
            center_scene_at_origin()

        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            export_draco_mesh_compression_enable=use_draco,
            export_draco_mesh_compression_level=draco_compression_level,
            export_image_format="AUTO",  # JPEG for opaque, PNG for alpha
            export_materials="EXPORT",
            export_cameras=False,
            export_lights=False,
            use_visible=True,
            use_renderable=True,
            use_active_collection=False,
            use_active_scene=True,
        )
        logger.info(f"Exported optimized GLB to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export GLB: {e}")
        return False

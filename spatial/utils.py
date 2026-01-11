"""Shared utility functions for spatial operations."""

import logging
import trimesh
import numpy as np

logger = logging.getLogger(__name__)

# Max vertices for spatial queries (to prevent memory explosion)
SPATIAL_QUERY_MAX_VERTICES = 5000


def downsample_for_query(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Voxel downsample mesh if needed for spatial queries.

    This prevents memory explosion and slow R-tree queries on high-poly meshes
    during proximity operations (on_surface, ray casting, etc.).

    Args:
        mesh: the mesh to potentially downsample

    Returns:
        The original mesh if under threshold, or a downsampled copy
    """
    if not hasattr(mesh, 'vertices'):
        logger.warning(f"downsample_for_query: mesh has no vertices attribute, type={type(mesh)}")
        return mesh

    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
    logger.info(f"downsample_for_query: input mesh has {vertex_count} vertices, {face_count} faces")

    if vertex_count <= SPATIAL_QUERY_MAX_VERTICES:
        logger.info(f"downsample_for_query: mesh under threshold ({SPATIAL_QUERY_MAX_VERTICES}), returning unchanged")
        return mesh

    original_vertices = len(mesh.vertices)

    # Compute voxel pitch to achieve target vertex count
    # Estimate: vertices ≈ (mesh_size / voxel_pitch)^3
    mesh_extents = mesh.bounds[1] - mesh.bounds[0]
    mesh_size = np.mean(mesh_extents)
    # Target: max_vertices ≈ (mesh_size / pitch)^3
    # => pitch ≈ mesh_size / (max_vertices)^(1/3)
    target_pitch = mesh_size / (SPATIAL_QUERY_MAX_VERTICES ** (1 / 3))

    # Ensure minimum pitch to avoid issues with very small meshes
    target_pitch = max(target_pitch, 0.01)  # At least 1cm

    try:
        # Create voxel-downsampled copy
        downsampled_mesh = mesh.copy()
        downsampled_mesh.merge_vertices()
        downsampled_mesh = downsampled_mesh.voxelized(pitch=target_pitch).marching_cubes

        logger.info(
            f"Voxel downsampled for spatial query: {original_vertices} -> {len(downsampled_mesh.vertices)} vertices "
            f"(pitch={target_pitch:.4f}m)"
        )
        return downsampled_mesh
    except Exception as e:
        logger.warning(f"Voxel downsampling failed: {e}, using original mesh")
        return mesh

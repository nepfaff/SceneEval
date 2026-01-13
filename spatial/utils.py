"""Shared utility functions for spatial operations."""

import logging
import multiprocessing as mp
import trimesh
import numpy as np

logger = logging.getLogger(__name__)

# Max vertices for spatial queries (to prevent memory explosion)
SPATIAL_QUERY_MAX_VERTICES = 5000

# Timeout for subprocess voxelization (seconds)
VOXEL_SUBPROCESS_TIMEOUT = 30


def _voxel_downsample_worker(vertices, faces, target_pitch, result_queue):
    """
    Subprocess worker for voxelization (isolates potential segfaults).

    Runs in a separate process so that segfaults in marching_cubes
    don't kill the main evaluation process.
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()
        downsampled = mesh.voxelized(pitch=target_pitch).marching_cubes
        result_queue.put({
            'success': True,
            'vertices': downsampled.vertices.copy(),
            'faces': downsampled.faces.copy()
        })
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)})


def _quadric_decimation_fallback(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """
    Fallback downsampling using quadric decimation.

    This is more robust than voxelization for degenerate geometry.
    """
    try:
        # Calculate face count target (roughly 2 faces per vertex for manifold meshes)
        target_faces = target_vertices * 2
        simplified = mesh.simplify_quadric_decimation(target_faces)
        logger.info(
            f"Quadric decimation fallback: {len(mesh.vertices)} -> {len(simplified.vertices)} vertices"
        )
        return simplified
    except Exception as e:
        logger.warning(f"Quadric decimation fallback failed: {e}")
        return mesh


def downsample_for_query(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Downsample mesh if needed for spatial queries, with robust fallback handling.

    This prevents memory explosion and slow R-tree queries on high-poly meshes
    during proximity operations (on_surface, ray casting, etc.).

    Uses subprocess isolation for voxelization to catch segfaults, with
    quadric decimation as a fallback.

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
    mesh_extents = mesh.bounds[1] - mesh.bounds[0]
    mesh_size = np.mean(mesh_extents)
    target_pitch = mesh_size / (SPATIAL_QUERY_MAX_VERTICES ** (1 / 3))
    target_pitch = max(target_pitch, 0.01)  # At least 1cm

    # Try voxelization in subprocess (isolates segfaults)
    try:
        result_queue = mp.Queue()
        p = mp.Process(
            target=_voxel_downsample_worker,
            args=(mesh.vertices.copy(), mesh.faces.copy(), target_pitch, result_queue)
        )
        p.start()
        p.join(timeout=VOXEL_SUBPROCESS_TIMEOUT)

        if p.is_alive():
            # Timeout - kill subprocess
            p.terminate()
            p.join(timeout=5)
            logger.warning(f"Voxel downsampling timed out after {VOXEL_SUBPROCESS_TIMEOUT}s, trying quadric decimation")
            return _quadric_decimation_fallback(mesh, SPATIAL_QUERY_MAX_VERTICES)

        if p.exitcode != 0:
            # Subprocess crashed (segfault = 139, etc.)
            exit_reason = "segfault" if p.exitcode == 139 else f"exit code {p.exitcode}"
            logger.warning(f"Voxel downsampling subprocess failed ({exit_reason}), trying quadric decimation")
            return _quadric_decimation_fallback(mesh, SPATIAL_QUERY_MAX_VERTICES)

        # Subprocess completed - check result
        if result_queue.empty():
            logger.warning("Voxel downsampling returned no result, trying quadric decimation")
            return _quadric_decimation_fallback(mesh, SPATIAL_QUERY_MAX_VERTICES)

        result = result_queue.get_nowait()

        if result.get('success'):
            downsampled_mesh = trimesh.Trimesh(
                vertices=result['vertices'],
                faces=result['faces']
            )
            logger.info(
                f"Voxel downsampled for spatial query: {original_vertices} -> {len(downsampled_mesh.vertices)} vertices "
                f"(pitch={target_pitch:.4f}m)"
            )
            return downsampled_mesh
        else:
            # Voxelization raised a Python exception
            logger.warning(f"Voxel downsampling failed: {result.get('error')}, trying quadric decimation")
            return _quadric_decimation_fallback(mesh, SPATIAL_QUERY_MAX_VERTICES)

    except Exception as e:
        # Fallback for any unexpected errors in subprocess management
        logger.warning(f"Subprocess voxel downsampling error: {e}, trying quadric decimation")
        return _quadric_decimation_fallback(mesh, SPATIAL_QUERY_MAX_VERTICES)

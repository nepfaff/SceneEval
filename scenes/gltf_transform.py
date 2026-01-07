"""GLB optimization using gltf-transform CLI.

Replaces custom Blender-based optimization with industry-standard tools
for better quality and smaller file sizes.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run_gltf_transform(args: list[str], timeout: int = 300) -> tuple[bool, str]:
    """Run a gltf-transform command.

    Args:
        args: Command arguments (without 'gltf-transform' prefix).
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, output/error message).
    """
    cmd = ["gltf-transform"] + args
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"gltf-transform failed: {error_msg}")
            return False, error_msg
        return True, result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"gltf-transform timed out after {timeout}s")
        return False, f"Timeout after {timeout}s"
    except FileNotFoundError:
        logger.error("gltf-transform not found. Install with: npm install -g @gltf-transform/cli")
        return False, "gltf-transform not found"
    except Exception as e:
        logger.error(f"gltf-transform error: {e}")
        return False, str(e)


def optimize_glb(
    input_path: Path,
    output_path: Path,
    max_texture_size: int = 512,  # Unused - kept for API compatibility
    simplify_error: float = 0.005,
    use_draco: bool = True,
    use_webp: bool = True,
    webp_quality: int = 80,
) -> dict:
    """Optimize a GLB file using gltf-transform pipeline.

    Pipeline:
    1. Weld vertices (merge duplicates)
    2. Simplify meshes (error-based, 0.5% max deviation)
    3. Deduplicate meshes, materials, textures
    4. Flatten hierarchy
    5. (Texture resize done in Blender with adaptive sizing)
    6. Compress textures (WebP)
    7. Prune unused data
    8. Center scene (for camera orbiting)
    9. Draco compression (MUST be last - other steps decompress it)

    Note: Texture resizing is now done adaptively in Blender before export,
    based on object bounding box dimensions. See glb_optimization.resize_textures_adaptive.

    Note: Vertex merging (to enable simplification for split-vertex meshes like
    scene-agent) is done in Blender. See glb_optimization.merge_vertices_by_distance.

    Args:
        input_path: Input GLB file.
        output_path: Output GLB file.
        max_texture_size: Unused - texture sizing done in Blender.
        simplify_error: Simplification error threshold (0.005 = 0.5% of bounds).
        use_draco: Apply Draco mesh compression.
        use_webp: Convert textures to WebP.
        webp_quality: WebP quality (1-100).

    Returns:
        Dict with optimization stats and success status.
    """
    stats = {
        "success": False,
        "input_size_mb": 0,
        "output_size_mb": 0,
        "reduction_percent": 0,
        "steps_completed": [],
        "errors": [],
    }

    if not input_path.exists():
        stats["errors"].append(f"Input file not found: {input_path}")
        return stats

    stats["input_size_mb"] = input_path.stat().st_size / (1024 * 1024)

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        current = input_path
        step_num = 0

        def next_temp() -> Path:
            nonlocal step_num
            step_num += 1
            return tmpdir / f"step_{step_num}.glb"

        # Step 1: Weld vertices (merge bitwise identical vertices)
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "weld", str(current), str(next_file),
        ])
        if success:
            stats["steps_completed"].append("weld")
            current = next_file
        else:
            stats["errors"].append(f"weld: {msg}")
            # Continue anyway - weld is optional

        # Step 2: Simplify meshes (error-based)
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "simplify", str(current), str(next_file),
            "--ratio", "0.0",  # Try to reduce as much as possible
            "--error", str(simplify_error),  # Stop when error exceeds threshold
        ])
        if success:
            stats["steps_completed"].append("simplify")
            current = next_file
        else:
            stats["errors"].append(f"simplify: {msg}")
            # Continue without simplification

        # Step 3: Deduplicate
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "dedup", str(current), str(next_file)
        ])
        if success:
            stats["steps_completed"].append("dedup")
            current = next_file
        else:
            stats["errors"].append(f"dedup: {msg}")

        # Step 4: Flatten hierarchy
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "flatten", str(current), str(next_file)
        ])
        if success:
            stats["steps_completed"].append("flatten")
            current = next_file
        else:
            stats["errors"].append(f"flatten: {msg}")

        # Step 5: Resize textures - SKIPPED (done adaptively in Blender)
        # Adaptive texture sizing is now done in Blender before export based on
        # object bounding box dimensions (see glb_optimization.resize_textures_adaptive)

        # Step 6: Convert to WebP
        if use_webp:
            next_file = next_temp()
            success, msg = run_gltf_transform([
                "webp", str(current), str(next_file),
                "--quality", str(webp_quality),
            ])
            if success:
                stats["steps_completed"].append("webp")
                current = next_file
            else:
                stats["errors"].append(f"webp: {msg}")

        # Step 7: Prune unused data (before Draco - prune decompresses Draco)
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "prune", str(current), str(next_file)
        ])
        if success:
            stats["steps_completed"].append("prune")
            current = next_file
        else:
            stats["errors"].append(f"prune: {msg}")

        # Step 8: Center scene (before Draco - center decompresses Draco)
        next_file = next_temp()
        success, msg = run_gltf_transform([
            "center", str(current), str(next_file)
        ])
        if success:
            stats["steps_completed"].append("center")
            current = next_file
        else:
            stats["errors"].append(f"center: {msg}")

        # Step 9: Draco compression (MUST be last - other steps decompress it)
        if use_draco:
            next_file = next_temp()
            success, msg = run_gltf_transform([
                "draco", str(current), str(next_file),
            ])
            if success:
                stats["steps_completed"].append("draco")
                current = next_file
            else:
                stats["errors"].append(f"draco: {msg}")

        # Copy final result to output
        try:
            shutil.copy2(current, output_path)
            stats["success"] = True
            stats["output_size_mb"] = output_path.stat().st_size / (1024 * 1024)
            if stats["input_size_mb"] > 0:
                stats["reduction_percent"] = round(
                    (1 - stats["output_size_mb"] / stats["input_size_mb"]) * 100, 1
                )
        except Exception as e:
            stats["errors"].append(f"Failed to copy output: {e}")

    return stats


def optimize_glb_single_command(
    input_path: Path,
    output_path: Path,
    simplify_error: float = 0.01,
    texture_compress: str = "webp",
    mesh_compress: str = "draco",
) -> dict:
    """Optimize GLB using single gltf-transform optimize command.

    This is faster but less configurable than the step-by-step approach.

    Args:
        input_path: Input GLB file.
        output_path: Output GLB file.
        simplify_error: Simplification error threshold.
        texture_compress: Texture compression (webp, ktx2, or none).
        mesh_compress: Mesh compression (draco, meshopt, or none).

    Returns:
        Dict with optimization stats.
    """
    stats = {
        "success": False,
        "input_size_mb": 0,
        "output_size_mb": 0,
        "reduction_percent": 0,
        "errors": [],
    }

    if not input_path.exists():
        stats["errors"].append(f"Input file not found: {input_path}")
        return stats

    stats["input_size_mb"] = input_path.stat().st_size / (1024 * 1024)

    args = ["optimize", str(input_path), str(output_path)]

    # Add simplification
    args.extend(["--simplify", "--simplify-error", str(simplify_error)])

    # Add texture compression
    if texture_compress and texture_compress != "none":
        args.extend(["--texture-compress", texture_compress])

    # Add mesh compression
    if mesh_compress and mesh_compress != "none":
        args.extend(["--compress", mesh_compress])

    success, msg = run_gltf_transform(args, timeout=600)

    if success:
        stats["success"] = True
        stats["output_size_mb"] = output_path.stat().st_size / (1024 * 1024)
        if stats["input_size_mb"] > 0:
            stats["reduction_percent"] = round(
                (1 - stats["output_size_mb"] / stats["input_size_mb"]) * 100, 1
            )
    else:
        stats["errors"].append(msg)

    return stats

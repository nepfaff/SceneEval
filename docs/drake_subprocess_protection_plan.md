# Plan: Subprocess Protection for Drake Collision Detection

> **Status**: Not implemented. Saved for reference if segfaults become more frequent.
> **Issue**: Scene 197 (LayoutVLM_Objaverse) caused a segfault in Drake's `ComputeSignedDistancePairwiseClosestPoints()`.

## Problem
Drake's `ComputeSignedDistancePairwiseClosestPoints()` can segfault on certain geometry configurations. Segfaults bypass Python's try/except and kill the entire evaluation process.

## Solution
Run the Drake collision metric in a subprocess. Export trimesh objects from main process, load them in subprocess to guarantee identical geometry. No extra plant builds.

---

## Implementation

### 1. Add `DrakeSceneData` class to `metrics/drake_utils.py`

A minimal dataclass that provides the interface `create_drake_plant_from_scene` needs:

```python
@dataclass
class DrakeSceneData:
    """Minimal scene data for Drake plant creation (duck-types as Scene)."""
    t_objs: dict[str, trimesh.Trimesh]
    carpet_obj_ids: set[str]
    t_architecture: dict[str, trimesh.Trimesh]

    def get_obj_ids(self) -> list[str]:
        return list(self.t_objs.keys())
```

### 2. Add helper functions to `metrics/drake_utils.py`

Functions to export/import trimesh data:

```python
def export_scene_data_for_subprocess(
    scene: Scene,
    output_dir: Path,
) -> Path:
    """Export scene trimesh data for subprocess consumption.

    Returns path to metadata JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export each object mesh
    objs_dir = output_dir / "trimesh_objs"
    objs_dir.mkdir(exist_ok=True)
    for obj_id, t_obj in scene.t_objs.items():
        safe_name = obj_id.replace(" ", "_").replace("/", "_")
        t_obj.export(objs_dir / f"{safe_name}.glb")

    # Export architecture meshes
    arch_dir = output_dir / "trimesh_arch"
    arch_dir.mkdir(exist_ok=True)
    for arch_id, t_arch in scene.t_architecture.items():
        safe_name = arch_id.replace(" ", "_").replace("/", "_")
        t_arch.export(arch_dir / f"{safe_name}.glb")

    # Save metadata
    metadata = {
        "obj_ids": list(scene.t_objs.keys()),
        "carpet_obj_ids": list(scene.carpet_obj_ids),
        "arch_ids": list(scene.t_architecture.keys()),
    }
    metadata_path = output_dir / "scene_metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    return metadata_path


def load_scene_data_from_export(metadata_path: Path) -> DrakeSceneData:
    """Load exported scene data into DrakeSceneData."""
    output_dir = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())

    # Load object meshes
    objs_dir = output_dir / "trimesh_objs"
    t_objs = {}
    for obj_id in metadata["obj_ids"]:
        safe_name = obj_id.replace(" ", "_").replace("/", "_")
        t_objs[obj_id] = trimesh.load(objs_dir / f"{safe_name}.glb", force="mesh")

    # Load architecture meshes
    arch_dir = output_dir / "trimesh_arch"
    t_architecture = {}
    for arch_id in metadata["arch_ids"]:
        safe_name = arch_id.replace(" ", "_").replace("/", "_")
        t_architecture[arch_id] = trimesh.load(arch_dir / f"{safe_name}.glb", force="mesh")

    return DrakeSceneData(
        t_objs=t_objs,
        carpet_obj_ids=set(metadata["carpet_obj_ids"]),
        t_architecture=t_architecture,
    )
```

### 3. New File: `metrics/drake_collision_worker.py`

Subprocess entry point that loads exported data and runs collision detection:

```python
#!/usr/bin/env python3
"""Subprocess worker for Drake collision detection - isolates segfaults."""
import json
import sys
from pathlib import Path


def main():
    config_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    config = json.loads(config_path.read_text())

    try:
        from metrics.drake_utils import (
            load_scene_data_from_export,
            create_drake_plant_from_scene,
            detect_penetrating_pairs,
        )

        # Load exported scene data (guarantees identical geometry)
        scene_data = load_scene_data_from_export(Path(config["metadata_path"]))

        # Create Drake plant
        builder, plant, scene_graph, obj_id_to_model_name = create_drake_plant_from_scene(
            scene=scene_data,  # DrakeSceneData duck-types as Scene
            time_step=0.0,
            temp_dir=Path(config["drake_scene_dir"]),
            weld_to_world=[],
            coacd_threshold=config.get("coacd_threshold", 0.05),
            vhacd_max_convex_hulls=config.get("vhacd_max_convex_hulls", 128),
            vhacd_resolution=config.get("vhacd_resolution", 400000),
            vhacd_max_recursion_depth=config.get("vhacd_max_recursion_depth", 10),
            vhacd_max_num_vertices_per_ch=config.get("vhacd_max_num_vertices_per_ch", 64),
            vhacd_min_volume_percent_error=config.get("vhacd_min_volume_percent_error", 1.0),
            vhacd_shrink_wrap=config.get("vhacd_shrink_wrap", True),
            vhacd_fill_mode=config.get("vhacd_fill_mode", "flood"),
            vhacd_min_edge_length=config.get("vhacd_min_edge_length", 2),
            vhacd_find_best_plane=config.get("vhacd_find_best_plane", False),
            decomposition_method=config["decomposition_method"],
        )

        # Build diagram and context
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Detect penetrating pairs (CRASH POINT)
        penetrating_pairs = detect_penetrating_pairs(
            plant=plant,
            scene_graph=scene_graph,
            context=context,
            threshold=config["penetration_threshold"],
            obj_id_to_model_name=obj_id_to_model_name,
        )

        # Write results
        result = {
            "success": True,
            "penetrating_pairs": [
                {"obj_a": a, "obj_b": b, "depth": d}
                for a, b, d in penetrating_pairs
            ],
            "obj_id_to_model_name": obj_id_to_model_name,
        }
        output_path.write_text(json.dumps(result))

    except Exception as e:
        import traceback
        output_path.write_text(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 4. Add subprocess wrapper to `metrics/drake_utils.py`

```python
def run_drake_collision_subprocess(
    scene: Scene,
    drake_scene_dir: Path,
    decomposition_method: str,
    penetration_threshold: float = 0.001,
    timeout: float = 300.0,
    **decomposition_kwargs,
) -> tuple[list[tuple[str, str, float]], dict[str, str], str | None]:
    """Run Drake collision detection in subprocess to isolate segfaults.

    Returns: (penetrating_pairs, obj_id_to_model_name, error_message)
    """
    import subprocess

    # Export scene data for subprocess
    export_dir = drake_scene_dir / "subprocess_data"
    metadata_path = export_scene_data_for_subprocess(scene, export_dir)

    # Write config
    config = {
        "metadata_path": str(metadata_path),
        "drake_scene_dir": str(drake_scene_dir),
        "decomposition_method": decomposition_method,
        "penetration_threshold": penetration_threshold,
        **decomposition_kwargs,
    }
    config_path = drake_scene_dir / "collision_config.json"
    output_path = drake_scene_dir / "collision_results.json"
    config_path.write_text(json.dumps(config))

    # Run subprocess
    worker = Path(__file__).parent / "drake_collision_worker.py"
    cmd = [sys.executable, str(worker), str(config_path), str(output_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # SceneEval root
        )

        if result.returncode == -11:  # SIGSEGV
            return [], {}, "Drake collision detection segfaulted (SIGSEGV)"
        if result.returncode == -6:  # SIGABRT
            return [], {}, "Drake collision detection aborted (SIGABRT)"

        if not output_path.exists():
            stderr = result.stderr[:500] if result.stderr else "no stderr"
            return [], {}, f"Subprocess failed (exit {result.returncode}): {stderr}"

        data = json.loads(output_path.read_text())

        if not data.get("success"):
            return [], {}, data.get("error", "Unknown error")

        pairs = [(p["obj_a"], p["obj_b"], p["depth"]) for p in data["penetrating_pairs"]]
        obj_id_to_model_name = data.get("obj_id_to_model_name", {})
        return pairs, obj_id_to_model_name, None

    except subprocess.TimeoutExpired:
        return [], {}, f"Drake collision detection timed out after {timeout}s"
    except Exception as e:
        return [], {}, f"Subprocess error: {e}"
```

### 5. Modify `metrics/drake_collision.py`

Replace `DrakeCollisionMetricBase.run()` to use subprocess:

```python
def run(self, verbose: bool = False) -> MetricResult:
    """Run the metric in a subprocess to protect against segfaults."""
    from .drake_utils import run_drake_collision_subprocess

    # Setup output directory
    drake_scene_dir = self.output_dir / self.drake_scene_folder
    drake_scene_dir.mkdir(parents=True, exist_ok=True)

    # Gather decomposition parameters
    decomposition_kwargs = {
        "coacd_threshold": getattr(self.cfg, "coacd_threshold", 0.05),
        "vhacd_max_convex_hulls": getattr(self.cfg, "vhacd_max_convex_hulls", 128),
        "vhacd_resolution": getattr(self.cfg, "vhacd_resolution", 400000),
        "vhacd_max_recursion_depth": getattr(self.cfg, "vhacd_max_recursion_depth", 10),
        "vhacd_max_num_vertices_per_ch": getattr(self.cfg, "vhacd_max_num_vertices_per_ch", 64),
        "vhacd_min_volume_percent_error": getattr(self.cfg, "vhacd_min_volume_percent_error", 1.0),
        "vhacd_shrink_wrap": getattr(self.cfg, "vhacd_shrink_wrap", True),
        "vhacd_fill_mode": getattr(self.cfg, "vhacd_fill_mode", "flood"),
        "vhacd_min_edge_length": getattr(self.cfg, "vhacd_min_edge_length", 2),
        "vhacd_find_best_plane": getattr(self.cfg, "vhacd_find_best_plane", False),
    }

    # Run in subprocess (exports scene data, loads in subprocess, runs collision)
    penetrating_pairs, obj_id_to_model_name, error = run_drake_collision_subprocess(
        scene=self.scene,
        drake_scene_dir=drake_scene_dir,
        decomposition_method=self.decomposition_method,
        penetration_threshold=self.cfg.penetration_threshold,
        timeout=300.0,
        **decomposition_kwargs,
    )

    method_name = self.decomposition_method.upper()

    if error:
        return MetricResult(
            message=f"Drake collision ({method_name}): {error}",
            data={"error": error, "decomposition_method": self.decomposition_method},
        )

    # Rest of existing result processing (lines 147-285)...
    # Process penetrating_pairs into collision_results, compute statistics, return MetricResult
```

---

## Files to Modify

| File | Action |
|------|--------|
| `metrics/drake_collision_worker.py` | Create new - subprocess entry point |
| `metrics/drake_utils.py` | Add `DrakeSceneData`, `export_scene_data_for_subprocess`, `load_scene_data_from_export`, `run_drake_collision_subprocess` |
| `metrics/drake_collision.py` | Replace `run()` to use subprocess wrapper |

---

## Data Flow

```
Main Process                              Subprocess
     |                                         |
     v                                         |
scene.t_objs (trimesh objects)                 |
     |                                         |
     v                                         |
export_scene_data_for_subprocess()             |
  -> writes GLB files + metadata.json          |
     |                                         |
     v                                         |
subprocess.run(worker.py)  -----------------> |
     |                                         v
     |                              load_scene_data_from_export()
     |                                -> identical trimesh objects
     |                                         |
     |                                         v
     |                              create_drake_plant_from_scene()
     |                                         |
     |                                         v
     |                              detect_penetrating_pairs()
     |                                  (may segfault here)
     |                                         |
     |                                         v
     |                              write results JSON
     |                                         |
     v  <--------------------------------------+
read results JSON or detect crash
     |
     v
return MetricResult
```

---

## Why Results Are Identical

1. **Same geometry**: Trimesh objects exported as GLB, loaded exactly in subprocess
2. **Same VHACD/CoACD params**: All decomposition parameters passed via config JSON
3. **Same collision threshold**: Passed via config JSON
4. **Deterministic algorithms**: VHACD and Drake collision are deterministic

---

## Verification

1. **Test with failing scene**: Run on `LayoutVLM_Objaverse` scene 197 - should record error and continue
2. **Test normal operation**: Run on a passing scene - compare results with/without subprocess (should be identical)
3. **Check error format**: Verify error appears correctly in `eval_result.json`

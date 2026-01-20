#!/usr/bin/env python3
"""
generate_results_markdown.py - Generate markdown results for SceneEval

Auto-discovers methods and scenes, extracts v2 metrics, and generates
comprehensive markdown tables with breakdowns by category and difficulty.

Usage:
    python generate_results_markdown.py
"""

import csv
import json
import math
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t as t_dist


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_results")
ANNOTATIONS_PATH = Path("/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/annotations.csv")
OUTPUT_PATH = Path("/home/ubuntu/SceneEval/results_SceneEval.md")

# Path to scene-agent converted scenes (contains sdfPath with object types)
SCENEAGENT_CONVERTED_DIR = Path("/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted")

# Object types that should be welded (architectural)
ARCHITECTURAL_TYPES = {"wall_mounted", "ceiling_mounted"}

# Methods to include and their display order
# Our method first, then ablations, then baselines
ROOM_METHODS = [
    # Our method
    "SceneAgent_Ours_Room",
    # Ablations
    "SceneAgent_NoCritic",
    "SceneAgent_MaxOneCritic",
    "SceneAgent_NoObserveScene",
    "SceneAgent_NoAgentMemory",
    "SceneAgent_NoSpecializedTools",
    "SceneAgent_HSSD",
    "SceneAgent_NoAssetValidation",
    # Baselines
    "HSM",
    "Holodeck",
    "IDesign",
    "LayoutVLM_Curated_Fixed",
    "LayoutVLM_Objaverse_Fixed",
    "SceneWeaver",
]

HOUSE_METHODS = [
    "SceneAgent_Ours_House",
    "Holodeck",
]

# Methods with nested directory structure (method/method/scene_X)
NESTED_METHODS = {"LayoutVLM_Curated_Fixed", "LayoutVLM_Objaverse_Fixed"}

# Metrics configuration: (metric_name, field_name, abbreviation, direction, display_name)
# direction: "higher" = higher is better, "lower" = lower is better, None = neutral
METRICS_CONFIG = [
    ("obj_count", "obj_count", "#Obj", None, "Object Count"),
    ("ObjCountMetric", "satisfaction_rate", "CNT", "higher", "Object Count"),
    ("ObjAttributeMetric", "satisfaction_rate", "ATR", "higher", "Object Attribute"),
    ("ObjObjRelationshipMetric", "satisfaction_rate", "OOR", "higher", "Obj-Obj Relationship"),
    ("ObjArchRelationshipMetric", "satisfaction_rate", "OAR", "higher", "Obj-Arch Relationship"),
    ("AccessibilityMetric", "avg_accessibility", "ACC", "higher", "Accessibility"),
    ("NavigabilityMetric", "navigability", "NAV", "higher", "Navigability"),
    ("DrakeCollisionMetricSceneAgent", "frac_obj_in_collision", "DRK", "lower", "Drake Collision"),
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "frac_stable", "STB", "higher", "Stability"),
    ("OutOfBoundMetric", "frac_out_of_bound", "OOB", "lower", "Out of Bounds"),
]

# Equilibrium statistics configuration for detailed displacement/rotation analysis
# Format: (filter_mode, stat_type, value_type, abbreviation, direction, display_name)
# filter_mode: "all" = all non-welded objects, "unstable" = only unstable objects
# stat_type: "mean", "max", "min", "mode"
# value_type: "disp" = displacement (meters), "rot" = rotation (radians)
EQUILIBRIUM_STATS_CONFIG = [
    # All non-welded objects (stable + unstable combined)
    ("all", "mean", "disp", "AMD", "lower", "All Mean Disp"),
    ("all", "max", "disp", "AXD", "lower", "All Max Disp"),
    ("all", "min", "disp", "AND", None, "All Min Disp"),
    ("all", "mode", "disp", "AOD", None, "All Mode Disp"),
    ("all", "mean", "rot", "AMR", "lower", "All Mean Rot"),
    ("all", "max", "rot", "AXR", "lower", "All Max Rot"),
    ("all", "min", "rot", "ANR", None, "All Min Rot"),
    ("all", "mode", "rot", "AOR", None, "All Mode Rot"),
    # Unstable objects only (moving objects)
    ("unstable", "mean", "disp", "UMD", "lower", "Unstable Mean Disp"),
    ("unstable", "max", "disp", "UXD", "lower", "Unstable Max Disp"),
    ("unstable", "min", "disp", "UND", None, "Unstable Min Disp"),
    ("unstable", "mode", "disp", "UOD", None, "Unstable Mode Disp"),
    ("unstable", "mean", "rot", "UMR", "lower", "Unstable Mean Rot"),
    ("unstable", "max", "rot", "UXR", "lower", "Unstable Max Rot"),
    ("unstable", "min", "rot", "UNR", None, "Unstable Min Rot"),
    ("unstable", "mode", "rot", "UOR", None, "Unstable Mode Rot"),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_annotations(annotations_path: Path) -> dict:
    """
    Load annotations.csv and return scene metadata.

    Returns:
        dict with keys:
            - scene_info: {scene_id: {category, difficulty, is_house, description}}
            - room_ids: set of room scene IDs
            - house_ids: set of house scene IDs
            - categories: {category_name: set of scene_ids}
            - difficulties: {difficulty_name: set of scene_ids} (room scenes only)
    """
    scene_info = {}
    room_ids = set()
    house_ids = set()
    categories = defaultdict(set)
    difficulties = defaultdict(set)

    with open(annotations_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = int(row["ID"])
            is_house = row.get("is_house", "False").strip().lower() == "true"
            category = row.get("Category", "").strip()
            difficulty = row.get("Difficulty", "").strip()
            description = row.get("Description", "").strip()

            scene_info[scene_id] = {
                "category": category,
                "difficulty": difficulty,
                "is_house": is_house,
                "description": description,
            }

            if is_house:
                house_ids.add(scene_id)
            else:
                room_ids.add(scene_id)
                if category:
                    categories[category].add(scene_id)
                if difficulty:
                    difficulties[difficulty].add(scene_id)

    return {
        "scene_info": scene_info,
        "room_ids": room_ids,
        "house_ids": house_ids,
        "categories": dict(categories),
        "difficulties": dict(difficulties),
    }


def discover_method_scenes(results_dir: Path, method: str, nested: bool = False) -> dict[int, Path]:
    """
    Discover available scenes for a method.

    Returns:
        dict mapping scene_id -> path to eval_result_v2.json
    """
    scene_paths = {}

    if nested:
        method_dir = results_dir / method / method
    else:
        method_dir = results_dir / method

    if not method_dir.exists():
        return scene_paths

    for scene_dir in method_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        match = re.match(r"scene_(\d+)", scene_dir.name)
        if not match:
            continue

        scene_id = int(match.group(1))
        eval_v2_path = scene_dir / "eval_result_v2.json"

        if eval_v2_path.exists():
            scene_paths[scene_id] = eval_v2_path

    return scene_paths


def extract_metric_value(results: dict, metric_name: str, field_name: str, num_objects: int) -> float | None:
    """Extract a specific metric value from v2 results."""

    # Special case: object count - always return the count (even 0)
    if metric_name == "obj_count" and field_name == "obj_count":
        return num_objects

    # Handle Drake collision metrics - try SceneAgent first, then VHACD
    if metric_name == "DrakeCollisionMetricSceneAgent" and metric_name not in results:
        if "DrakeCollisionMetricVHACD" in results:
            metric_name = "DrakeCollisionMetricVHACD"
        elif "DrakeCollisionMetricCoACD" in results:
            metric_name = "DrakeCollisionMetricCoACD"

    # Handle Equilibrium metrics - try SceneAgent first, then VHACD
    if metric_name == "ArchitecturalWeldedEquilibriumMetricSceneAgent" and metric_name not in results:
        if "ArchitecturalWeldedEquilibriumMetricVHACD" in results:
            metric_name = "ArchitecturalWeldedEquilibriumMetricVHACD"
        elif "ArchitecturalWeldedEquilibriumMetricCoACD" in results:
            metric_name = "ArchitecturalWeldedEquilibriumMetricCoACD"

    if metric_name not in results:
        return None

    metric_result = results[metric_name]
    metric_data = metric_result.get("data", {})

    # Handle satisfaction_rate computed from message
    if field_name == "satisfaction_rate":
        message = metric_result.get("message", "")
        match = re.search(r"(\d+)/(\d+)", message)
        if match:
            num, denom = int(match.group(1)), int(match.group(2))
            if denom > 0:
                return num / denom
        # "No ... to evaluate" = scene has no such requirements (legitimately excluded)
        if "No " in message and "to evaluate" in message:
            return None
        # "No ... have all objects present" = requirements exist but objects missing = 0% failure
        if "No " in message and "have all objects present" in message:
            return 0.0
        return None

    # Handle AccessibilityMetric avg_accessibility
    if metric_name == "AccessibilityMetric" and field_name == "avg_accessibility":
        scores = []
        for obj_id, obj_data in metric_data.items():
            if isinstance(obj_data, dict) and "max" in obj_data:
                max_score = obj_data["max"]
                if max_score >= 0:
                    scores.append(max_score)
        if scores:
            return sum(scores) / len(scores)
        # No accessibility data - return None (metric couldn't be computed)
        return None

    # Handle NavigabilityMetric
    if metric_name == "NavigabilityMetric" and field_name == "navigability":
        return metric_data.get("navigability")

    # Handle DrakeCollisionMetric - compute fraction from counts
    # 0 objects = 0% collision (nothing to collide)
    if "DrakeCollisionMetric" in metric_name and field_name == "frac_obj_in_collision":
        num_in_collision = metric_data.get("num_obj_in_collision", 0)
        if num_objects > 0:
            return num_in_collision / num_objects
        return 0.0

    # Handle ArchitecturalWeldedEquilibriumMetric - compute fraction stable
    if "ArchitecturalWeldedEquilibriumMetric" in metric_name and field_name == "frac_stable":
        num_stable = metric_data.get("num_stable_objects", 0)
        num_unstable = metric_data.get("num_unstable_objects", 0)
        total = num_stable + num_unstable
        if total > 0:
            return num_stable / total
        return None

    # Handle OutOfBoundMetric
    # 0 objects = 0% out of bounds (nothing to be out of bounds)
    if metric_name == "OutOfBoundMetric" and field_name == "frac_out_of_bound":
        count = sum(1 for obj_data in metric_data.values()
                   if isinstance(obj_data, dict) and obj_data.get("out_of_bound", False))
        if num_objects > 0:
            return count / num_objects
        return 0.0

    # Direct field access
    if field_name in metric_data:
        return metric_data[field_name]

    return None


def compute_mode(values: list[float], decimal_places: int = 3) -> float | None:
    """
    Compute mode of continuous values by rounding to decimal_places.
    Returns the most common rounded value, or None if empty.
    """
    if not values:
        return None
    rounded = [round(v, decimal_places) for v in values]
    counter = Counter(rounded)
    mode_value, _ = counter.most_common(1)[0]
    return mode_value


def extract_equilibrium_object_stats(results: dict, filter_mode: str) -> dict | None:
    """
    Extract displacement and rotation statistics from equilibrium per_object_results.

    Args:
        results: The results dict from eval_result_v2.json
        filter_mode: "all" for all non-welded objects, "unstable" for unstable only

    Returns:
        Dict with keys: mean_disp, max_disp, min_disp, mode_disp,
                        mean_rot, max_rot, min_rot, mode_rot
        Or None if no matching objects found.
    """
    # Find the equilibrium metric (try different decomposition methods)
    metric_name = None
    for name in ["ArchitecturalWeldedEquilibriumMetricSceneAgent",
                 "ArchitecturalWeldedEquilibriumMetricVHACD",
                 "ArchitecturalWeldedEquilibriumMetricCoACD"]:
        if name in results:
            metric_name = name
            break

    if metric_name is None:
        return None

    metric_data = results[metric_name].get("data", {})
    per_object_results = metric_data.get("per_object_results", {})

    if not per_object_results:
        return None

    # Filter objects based on filter_mode
    displacements = []
    rotations = []

    for obj_id, obj_data in per_object_results.items():
        if not isinstance(obj_data, dict):
            continue

        # Skip welded objects for both modes
        if obj_data.get("welded", False):
            continue

        # For "unstable" mode, only include unstable objects
        if filter_mode == "unstable" and obj_data.get("stable", True):
            continue

        disp = obj_data.get("displacement")
        rot = obj_data.get("rotation")

        if disp is not None:
            displacements.append(disp)
        if rot is not None:
            rotations.append(rot)

    # If no matching objects, return None
    if not displacements and not rotations:
        return None

    stats = {}

    # Displacement stats
    if displacements:
        stats["mean_disp"] = mean(displacements)
        stats["max_disp"] = max(displacements)
        stats["min_disp"] = min(displacements)
        stats["mode_disp"] = compute_mode(displacements)
    else:
        stats["mean_disp"] = None
        stats["max_disp"] = None
        stats["min_disp"] = None
        stats["mode_disp"] = None

    # Rotation stats
    if rotations:
        stats["mean_rot"] = mean(rotations)
        stats["max_rot"] = max(rotations)
        stats["min_rot"] = min(rotations)
        stats["mode_rot"] = compute_mode(rotations)
    else:
        stats["mean_rot"] = None
        stats["max_rot"] = None
        stats["min_rot"] = None
        stats["mode_rot"] = None

    return stats


def load_scene_metrics(eval_path: Path) -> dict:
    """Load metrics from eval_result file (v1 or v2)."""
    with open(eval_path, "r") as f:
        data = json.load(f)

    num_objects = len(data.get("obj_ids", []))
    results = data.get("results", {})

    metrics = {}
    for metric_name, field_name, abbrev, direction, display_name in METRICS_CONFIG:
        value = extract_metric_value(results, metric_name, field_name, num_objects)
        metrics[abbrev] = value

    return metrics


def load_scene_equilibrium_stats(eval_path: Path) -> dict:
    """Load equilibrium statistics from eval_result file."""
    with open(eval_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    stats = {}

    # Extract stats for both filter modes
    for filter_mode, stat_type, value_type, abbrev, direction, display_name in EQUILIBRIUM_STATS_CONFIG:
        # Get stats for this filter mode (cached per filter_mode)
        cache_key = f"_cache_{filter_mode}"
        if cache_key not in stats:
            stats[cache_key] = extract_equilibrium_object_stats(results, filter_mode)

        cached_stats = stats[cache_key]
        if cached_stats is not None:
            key = f"{stat_type}_{value_type}"
            stats[abbrev] = cached_stats.get(key)
        else:
            stats[abbrev] = None

    # Remove cache keys
    for key in list(stats.keys()):
        if key.startswith("_cache_"):
            del stats[key]

    return stats


# =============================================================================
# GROUND TRUTH STABILITY & VLM CLASSIFICATION
# =============================================================================

# Mapping from VLM support type to GT object type
VLM_TO_GT_TYPE = {
    "ground": "furniture",
    "wall": "wall_mounted",
    "ceiling": "ceiling_mounted",
    "object": "manipuland",
}

# All valid GT object types
GT_OBJECT_TYPES = {"furniture", "wall_mounted", "ceiling_mounted", "manipuland"}


def extract_object_types_from_scene_json(method: str, scene_id: int) -> dict[str, str] | None:
    """
    Extract object_type from sdfPath for each object in scene JSON.

    Args:
        method: Method name (e.g., "SceneAgent_Ours_House")
        scene_id: Scene ID

    Returns:
        Dict mapping object_id (e.g., "reception_counter_desk_0") to object_type
        (e.g., "furniture", "wall_mounted", "ceiling_mounted", "manipuland")
        Returns None if scene JSON not found.
    """
    scene_json = SCENEAGENT_CONVERTED_DIR / method / f"scene_{scene_id}.json"
    if not scene_json.exists():
        return None

    with open(scene_json) as f:
        data = json.load(f)

    object_types = {}
    for obj in data.get("scene", {}).get("object", []):
        obj_id = obj.get("id")
        sdf_path = obj.get("sdfPath", "")
        if obj_id and sdf_path:
            # sdfPath format: "{object_type}/sdf/{asset_name}/..."
            obj_type = sdf_path.split("/")[0]
            object_types[obj_id] = obj_type

    return object_types


def load_vlm_support_types(eval_path: Path) -> dict[str, str] | None:
    """
    Load VLM support type classifications from obj_support_type_result.json.

    Returns:
        Dict mapping full object ID to VLM support type ("ground", "wall", "ceiling", "object")
        Returns None if file not found.
    """
    support_type_path = eval_path.parent / "obj_support_type_result.json"
    if not support_type_path.exists():
        return None

    with open(support_type_path) as f:
        return json.load(f)


def compute_ground_truth_stability(
    eval_path: Path,
    object_types: dict[str, str],
    vlm_support_types: dict[str, str]
) -> dict | None:
    """
    Compute stability using ground truth object types and VLM classification accuracy.

    Uses obj_support_type_result.json for VLM classifications (ground/wall/ceiling/object)
    and compares with GT types from sdfPath (furniture/wall_mounted/ceiling_mounted/manipuland).

    Returns:
        Dict with stability and classification metrics.
    """
    with open(eval_path) as f:
        data = json.load(f)

    results = data.get("results", {})

    # Find equilibrium metric
    metric_name = None
    for name in ["ArchitecturalWeldedEquilibriumMetricSceneAgent",
                 "ArchitecturalWeldedEquilibriumMetricVHACD",
                 "ArchitecturalWeldedEquilibriumMetricCoACD"]:
        if name in results:
            metric_name = name
            break

    if metric_name is None:
        return None

    metric_data = results[metric_name].get("data", {})
    per_object_results = metric_data.get("per_object_results", {})

    if not per_object_results:
        return None

    # Current VLM-based counts
    num_welded = metric_data.get("num_welded_objects", 0)
    num_stable = metric_data.get("num_stable_objects", 0)
    num_unstable = metric_data.get("num_unstable_objects", 0)

    # Classification counts
    total_matched = 0
    correct_classifications = 0
    unstable_architectural = 0  # Arch objects misclassified that were unstable

    # Per-class counts for detailed breakdown
    class_counts = {gt: {"total": 0, "correct": 0} for gt in GT_OBJECT_TYPES}

    # Build reverse lookup for VLM support types (full_id -> support_type)
    # Need to match per_object_results keys to vlm_support_types keys

    for obj_id, obj_data in per_object_results.items():
        if not isinstance(obj_data, dict):
            continue

        # Extract raw object ID from prefixed ID
        # Two formats:
        # 1. "idx55_scene-agent.scene_203__painting_0" -> "painting_0" (split on __)
        # 2. "8_print_0" -> "print_0" (remove leading number prefix)
        if "__" in obj_id:
            raw_id = obj_id.split("__")[-1]
            # For full format, use the original for VLM lookup
            vlm_lookup_id = obj_id
        else:
            match = re.match(r"^\d+_(.+)$", obj_id)
            raw_id = match.group(1) if match else obj_id
            vlm_lookup_id = None  # Will need to find matching key

        gt_type = object_types.get(raw_id)
        if gt_type is None or gt_type not in GT_OBJECT_TYPES:
            continue

        # Find VLM classification for this object
        vlm_support = None
        if vlm_lookup_id and vlm_lookup_id in vlm_support_types:
            vlm_support = vlm_support_types[vlm_lookup_id]
        else:
            # Try to find matching key in vlm_support_types
            for vlm_id, support in vlm_support_types.items():
                if vlm_id.endswith(f"__{raw_id}"):
                    vlm_support = support
                    break

        if vlm_support is None:
            continue

        total_matched += 1
        class_counts[gt_type]["total"] += 1

        # Check if VLM classification matches GT
        vlm_gt_equivalent = VLM_TO_GT_TYPE.get(vlm_support)
        if vlm_gt_equivalent == gt_type:
            correct_classifications += 1
            class_counts[gt_type]["correct"] += 1
        else:
            # Misclassification - check if it affects stability
            gt_is_arch = gt_type in ARCHITECTURAL_TYPES
            vlm_thinks_arch = vlm_support in {"wall", "ceiling"}

            if gt_is_arch and not vlm_thinks_arch:
                # Should have been welded but wasn't
                if not obj_data.get("stable", True):
                    unstable_architectural += 1

    # Compute stabilities
    total_non_welded = num_stable + num_unstable
    vlm_stability = num_stable / total_non_welded if total_non_welded > 0 else None

    # Ground truth stability
    gt_stable = num_stable + unstable_architectural
    gt_unstable = num_unstable - unstable_architectural
    gt_total = gt_stable + gt_unstable
    gt_stability = gt_stable / gt_total if gt_total > 0 else None

    # Classification accuracy
    accuracy = correct_classifications / total_matched if total_matched > 0 else None

    return {
        "vlm_stability": vlm_stability,
        "gt_stability": gt_stability,
        "unstable_architectural": unstable_architectural,
        "total_matched": total_matched,
        "correct_classifications": correct_classifications,
        "accuracy": accuracy,
        "class_counts": class_counts,
        "num_welded": num_welded,
        "num_stable": num_stable,
        "num_unstable": num_unstable,
    }


def load_scene_ground_truth_stability(method: str, eval_path: Path, scene_id: int) -> dict | None:
    """Load ground truth stability for a single scene."""
    # Only compute for SceneAgent methods (they have sdfPath info)
    if not method.startswith("SceneAgent"):
        return None

    object_types = extract_object_types_from_scene_json(method, scene_id)
    if object_types is None:
        return None

    vlm_support_types = load_vlm_support_types(eval_path)
    if vlm_support_types is None:
        return None

    return compute_ground_truth_stability(eval_path, object_types, vlm_support_types)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_metrics(method_data: dict[int, dict], scene_ids: set[int]) -> tuple[dict, int]:
    """
    Aggregate metrics over a subset of scenes.

    Args:
        method_data: dict mapping scene_id -> metrics dict
        scene_ids: set of scene IDs to aggregate over

    Returns:
        (aggregated_metrics, count) where aggregated_metrics has (mean, ci) tuples
        ci is the 95% confidence interval half-width, or None if n < 2
    """
    # Filter to scenes in both method_data and scene_ids
    valid_ids = set(method_data.keys()) & scene_ids
    count = len(valid_ids)

    if count == 0:
        return {}, 0

    # Collect all values per metric (filter out None and NaN)
    values_by_metric = defaultdict(list)

    for scene_id in valid_ids:
        metrics = method_data[scene_id]
        for abbrev, value in metrics.items():
            if value is not None and not math.isnan(value):
                values_by_metric[abbrev].append(value)

    # Compute mean and 95% CI for each metric
    aggregated = {}
    for abbrev, values in values_by_metric.items():
        n = len(values)
        if n == 0:
            continue
        elif n == 1:
            # Can't compute variance with 1 sample
            aggregated[abbrev] = (values[0], None)
        else:
            # Compute mean, SD, and 95% CI
            m = mean(values)
            sd = stdev(values)
            # t-value for 95% CI with n-1 degrees of freedom
            t_val = t_dist.ppf(0.975, n - 1)
            ci = t_val * (sd / math.sqrt(n))
            aggregated[abbrev] = (m, ci)

    return aggregated, count


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================

def format_value(value: tuple[float, float | None] | None, abbrev: str) -> str:
    """Format a metric value with optional CI for display.

    Args:
        value: Either None, or a tuple (mean, ci) where ci may be None
        abbrev: Metric abbreviation to determine formatting
    """
    if value is None:
        return "-"

    mean_val, ci = value

    if abbrev == "#Obj":
        if ci is not None:
            return f"{mean_val:.1f} ± {ci:.1f}"
        return f"{mean_val:.1f}"
    else:
        # Percentage format
        if ci is not None:
            return f"{mean_val:.1%} ± {ci:.1%}"
        return f"{mean_val:.1%}"


def format_equilibrium_value(value: tuple[float, float | None] | None, abbrev: str) -> str:
    """Format an equilibrium statistic value with optional CI for display.

    Args:
        value: Either None, or a tuple (mean, ci) where ci may be None
        abbrev: Metric abbreviation to determine formatting
    """
    if value is None:
        return "-"

    mean_val, ci = value

    # Handle NaN and infinity
    if math.isnan(mean_val) or math.isinf(mean_val):
        return "-"

    # Detect value type from abbreviation suffix
    # D = displacement (meters), R = rotation (radians)
    if abbrev.endswith("D"):
        # Displacement in meters - show in mm if < 1m, otherwise m
        if abs(mean_val) < 0.001:
            mean_str = f"{mean_val*1000:.3f}mm"
            ci_str = f"{ci*1000:.3f}mm" if ci is not None else None
        elif abs(mean_val) < 1.0:
            mean_str = f"{mean_val*1000:.1f}mm"
            ci_str = f"{ci*1000:.1f}mm" if ci is not None else None
        else:
            mean_str = f"{mean_val:.3f}m"
            ci_str = f"{ci:.3f}m" if ci is not None else None
    elif abbrev.endswith("R"):
        # Rotation in radians
        if abs(mean_val) < 0.001:
            mean_str = f"{mean_val:.4f}rad"
            ci_str = f"{ci:.4f}rad" if ci is not None else None
        else:
            mean_str = f"{mean_val:.3f}rad"
            ci_str = f"{ci:.3f}rad" if ci is not None else None
    else:
        mean_str = f"{mean_val:.4f}"
        ci_str = f"{ci:.4f}" if ci is not None else None

    if ci_str is not None:
        return f"{mean_str} ± {ci_str}"
    return mean_str


def generate_metric_legend() -> str:
    """Generate the metric legend section."""
    lines = [
        "## Metric Legend",
        "",
        "| Abbrev | Metric | Description | Direction |",
        "|:-------|:-------|:------------|:----------|",
    ]

    for metric_name, field_name, abbrev, direction, display_name in METRICS_CONFIG:
        if direction == "higher":
            dir_str = "↑ Higher is better"
        elif direction == "lower":
            dir_str = "↓ Lower is better"
        else:
            dir_str = "-"

        if abbrev == "#Obj":
            desc = "Average number of objects per scene"
        elif abbrev == "CNT":
            desc = "Satisfaction rate for object count requirements"
        elif abbrev == "ATR":
            desc = "Satisfaction rate for object attribute requirements"
        elif abbrev == "OOR":
            desc = "Satisfaction rate for object-object spatial relationships"
        elif abbrev == "OAR":
            desc = "Satisfaction rate for object-architecture relationships"
        elif abbrev == "ACC":
            desc = "Average accessibility score for objects"
        elif abbrev == "NAV":
            desc = "Fraction of floor area that is navigable"
        elif abbrev == "DRK":
            desc = "Fraction of objects in collision (physics simulation)"
        elif abbrev == "STB":
            desc = "Fraction of objects that are stable (physics simulation)"
        elif abbrev == "OOB":
            desc = "Fraction of objects outside room boundaries"
        else:
            desc = display_name

        lines.append(f"| {abbrev} | {display_name} | {desc} | {dir_str} |")

    # Add error bar explanation
    lines.extend([
        "",
        "**Error Bars:** Values are reported as mean ± 95% confidence interval (CI). "
        "The CI is computed using the t-distribution: CI = t(0.975, n-1) × (SD / √n), "
        "where n is the number of scenes. Non-overlapping CIs suggest statistically significant differences.",
    ])

    return "\n".join(lines)


def generate_equilibrium_legend() -> str:
    """Generate the equilibrium statistics legend section."""
    lines = [
        "## Equilibrium Statistics Legend",
        "",
        "Statistics are computed from physics simulation displacement/rotation values.",
        "",
        "**Error Bars:** Values are reported as mean ± 95% CI across scenes (see Metric Legend for formula).",
        "",
        "**Object Sets:**",
        "- **All (A\\*)**: All non-welded objects (stable + unstable combined)",
        "- **Unstable (U\\*)**: Only unstable objects (objects that moved during simulation)",
        "",
        "**Statistics:**",
        "- **Mean**: Average value across objects",
        "- **Max (X)**: Maximum value",
        "- **Min (N)**: Minimum value",
        "- **Mode (O)**: Most common value (rounded to 3 decimal places)",
        "",
        "**Value Types:**",
        "- **Disp (\\*D)**: Displacement in meters (shown as mm when < 1m)",
        "- **Rot (\\*R)**: Rotation in radians",
        "",
        "| Abbrev | Description |",
        "|:-------|:------------|",
    ]

    abbrev_descriptions = {
        "AMD": "All objects: Mean displacement",
        "AXD": "All objects: Max displacement",
        "AND": "All objects: Min displacement",
        "AOD": "All objects: Mode displacement",
        "AMR": "All objects: Mean rotation",
        "AXR": "All objects: Max rotation",
        "ANR": "All objects: Min rotation",
        "AOR": "All objects: Mode rotation",
        "UMD": "Unstable objects: Mean displacement",
        "UXD": "Unstable objects: Max displacement",
        "UND": "Unstable objects: Min displacement",
        "UOD": "Unstable objects: Mode displacement",
        "UMR": "Unstable objects: Mean rotation",
        "UXR": "Unstable objects: Max rotation",
        "UNR": "Unstable objects: Min rotation",
        "UOR": "Unstable objects: Mode rotation",
    }

    for filter_mode, stat_type, value_type, abbrev, direction, display_name in EQUILIBRIUM_STATS_CONFIG:
        desc = abbrev_descriptions.get(abbrev, display_name)
        lines.append(f"| {abbrev} | {desc} |")

    return "\n".join(lines)


def generate_scene_counts_table(
    all_method_data: dict[str, dict[int, dict]],
    methods: list[str],
    annotations: dict,
) -> str:
    """Generate scene counts table by category."""
    categories = annotations["categories"]
    category_order = ["SceneEval-100", "type_diversity", "object_density", "themed_scenes"]

    lines = [
        "## Scene Counts",
        "",
        "| Category | " + " | ".join(methods) + " |",
        "|:---------|" + "|".join(["---:" for _ in methods]) + "|",
    ]

    totals = {m: 0 for m in methods}

    for category in category_order:
        cat_ids = categories.get(category, set())
        row = [category]
        for method in methods:
            method_data = all_method_data.get(method, {})
            count = len(set(method_data.keys()) & cat_ids)
            row.append(str(count))
            totals[method] += count
        lines.append("| " + " | ".join(row) + " |")

    # Total row
    row = ["**Total**"]
    for method in methods:
        row.append(f"**{totals[method]}**")
    lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_results_table(
    all_method_data: dict[str, dict[int, dict]],
    methods: list[str],
    scene_ids: set[int],
    title: str,
) -> str:
    """Generate a results table for a set of scenes."""
    abbrevs = [m[2] for m in METRICS_CONFIG]

    lines = [
        f"## {title}",
        "",
        "| Method | N | " + " | ".join(abbrevs) + " |",
        "|:---|---:|" + "|".join(["---:" for _ in abbrevs]) + "|",
    ]

    for method in methods:
        method_data = all_method_data.get(method, {})
        agg, count = aggregate_metrics(method_data, scene_ids)

        row = [method, str(count)]
        for abbrev in abbrevs:
            value = agg.get(abbrev)
            row.append(format_value(value, abbrev))

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_equilibrium_stats_table(
    all_equilibrium_data: dict[str, dict[int, dict]],
    methods: list[str],
    scene_ids: set[int],
    title: str,
) -> str:
    """Generate an equilibrium statistics table for a set of scenes."""
    # Split stats into two groups: All non-welded and Unstable only
    all_abbrevs = [cfg[3] for cfg in EQUILIBRIUM_STATS_CONFIG if cfg[0] == "all"]
    unstable_abbrevs = [cfg[3] for cfg in EQUILIBRIUM_STATS_CONFIG if cfg[0] == "unstable"]

    lines = [
        f"## {title}",
        "",
        "### All Non-Welded Objects",
        "",
        "| Method | N | " + " | ".join(all_abbrevs) + " |",
        "|:---|---:|" + "|".join(["---:" for _ in all_abbrevs]) + "|",
    ]

    for method in methods:
        method_data = all_equilibrium_data.get(method, {})
        agg, count = aggregate_metrics(method_data, scene_ids)

        row = [method, str(count)]
        for abbrev in all_abbrevs:
            value = agg.get(abbrev)
            row.append(format_equilibrium_value(value, abbrev))

        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### Unstable Objects Only",
        "",
        "| Method | N | " + " | ".join(unstable_abbrevs) + " |",
        "|:---|---:|" + "|".join(["---:" for _ in unstable_abbrevs]) + "|",
    ])

    for method in methods:
        method_data = all_equilibrium_data.get(method, {})
        agg, count = aggregate_metrics(method_data, scene_ids)

        row = [method, str(count)]
        for abbrev in unstable_abbrevs:
            value = agg.get(abbrev)
            row.append(format_equilibrium_value(value, abbrev))

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_gt_stability_section(
    all_gt_stability_data: dict[str, dict[int, dict]],
    annotations: dict,
) -> str:
    """
    Generate the VLM Support Type Classification Ablation section.

    Shows VLM classification accuracy and its impact on stability.
    """
    room_ids = annotations["room_ids"]
    house_ids = annotations["house_ids"]

    lines = [
        "## VLM Support Type Classification Ablation",
        "",
        "This analysis evaluates VLM support type classification accuracy by comparing",
        "VLM predictions (`obj_support_type_result.json`) with ground truth from scene-agent's sdfPath.",
        "",
        "**Classification Mapping:**",
        "- VLM `ground` ↔ GT `furniture`",
        "- VLM `wall` ↔ GT `wall_mounted`",
        "- VLM `ceiling` ↔ GT `ceiling_mounted`",
        "- VLM `object` ↔ GT `manipuland`",
        "",
    ]

    # Store aggregated data
    aggregated_data = []

    # Aggregate data by method and scene type
    for method in sorted(all_gt_stability_data.keys()):
        method_data = all_gt_stability_data[method]

        for scene_type, scene_ids in [("Room", room_ids), ("House", house_ids)]:
            valid_ids = set(method_data.keys()) & scene_ids
            if not valid_ids:
                continue

            # Aggregate statistics
            total_vlm_stability = 0.0
            total_gt_stability = 0.0
            total_matched = 0
            total_correct = 0
            total_unstable_arch = 0
            count_vlm = 0
            count_gt = 0

            # Per-class aggregation
            class_totals = {gt: {"total": 0, "correct": 0} for gt in GT_OBJECT_TYPES}

            for scene_id in valid_ids:
                stats = method_data[scene_id]
                if stats.get("vlm_stability") is not None:
                    total_vlm_stability += stats["vlm_stability"]
                    count_vlm += 1
                if stats.get("gt_stability") is not None:
                    total_gt_stability += stats["gt_stability"]
                    count_gt += 1
                total_matched += stats.get("total_matched", 0)
                total_correct += stats.get("correct_classifications", 0)
                total_unstable_arch += stats.get("unstable_architectural", 0)

                # Aggregate per-class counts
                class_counts = stats.get("class_counts", {})
                for gt_type in GT_OBJECT_TYPES:
                    if gt_type in class_counts:
                        class_totals[gt_type]["total"] += class_counts[gt_type].get("total", 0)
                        class_totals[gt_type]["correct"] += class_counts[gt_type].get("correct", 0)

            # Compute averages
            avg_vlm = total_vlm_stability / count_vlm if count_vlm > 0 else None
            avg_gt = total_gt_stability / count_gt if count_gt > 0 else None
            overall_accuracy = total_correct / total_matched if total_matched > 0 else None

            aggregated_data.append({
                "method": method,
                "scene_type": scene_type,
                "n": len(valid_ids),
                "vlm_stb": avg_vlm,
                "gt_stb": avg_gt,
                "total_matched": total_matched,
                "total_correct": total_correct,
                "accuracy": overall_accuracy,
                "class_totals": class_totals,
                "unstable_arch": total_unstable_arch,
            })

    # Build classification accuracy table
    lines.extend([
        "### Classification Accuracy",
        "",
        "| Method | Scene Type | N | Obj/Scene | Accuracy |",
        "|:-------|:-----------|--:|----------:|---------:|",
    ])

    for data in aggregated_data:
        acc_str = f"{data['accuracy']:.1%}" if data['accuracy'] is not None else "-"
        obj_per_scene = data['total_matched'] / data['n'] if data['n'] > 0 else 0
        lines.append(
            f"| {data['method']} | {data['scene_type']} | {data['n']} | "
            f"{obj_per_scene:.1f} | {acc_str} |"
        )

    lines.append("")

    # Build per-class accuracy table
    lines.extend([
        "### Per-Class Accuracy",
        "",
        "| Method | Scene Type | Furniture | Wall Mounted | Ceiling Mounted | Manipuland |",
        "|:-------|:-----------|----------:|-------------:|----------------:|-----------:|",
    ])

    for data in aggregated_data:
        class_accs = []
        for gt_type in ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]:
            ct = data["class_totals"][gt_type]
            if ct["total"] > 0:
                acc = ct["correct"] / ct["total"]
                class_accs.append(f"{acc:.1%}")
            else:
                class_accs.append("-")

        lines.append(
            f"| {data['method']} | {data['scene_type']} | "
            f"{class_accs[0]} | {class_accs[1]} | {class_accs[2]} | {class_accs[3]} |"
        )

    lines.append("")

    # Build stability impact table
    lines.extend([
        "### Stability Impact",
        "",
        "| Method | Scene Type | N | VLM STB | GT STB | Diff | Unstable Arch |",
        "|:-------|:-----------|--:|--------:|-------:|-----:|--------------:|",
    ])

    for data in aggregated_data:
        vlm_str = f"{data['vlm_stb']:.1%}" if data['vlm_stb'] is not None else "-"
        gt_str = f"{data['gt_stb']:.1%}" if data['gt_stb'] is not None else "-"
        if data['vlm_stb'] is not None and data['gt_stb'] is not None:
            diff_str = f"+{(data['gt_stb'] - data['vlm_stb']):.1%}"
        else:
            diff_str = "-"

        lines.append(
            f"| {data['method']} | {data['scene_type']} | {data['n']} | "
            f"{vlm_str} | {gt_str} | {diff_str} | {data['unstable_arch']} |"
        )

    lines.append("")

    # Add interpretation
    lines.extend([
        "**Interpretation:**",
        "",
        "- **Accuracy**: Fraction of objects where VLM support type matches ground truth",
        "- **Per-Class Accuracy**: Accuracy broken down by ground truth object type",
        "- **GT STB**: Stability if architectural objects were correctly identified (oracle)",
        "- **Unstable Arch**: Misclassified architectural objects that fell (causes STB gap)",
        "",
    ])

    return "\n".join(lines)


def generate_markdown_report(
    all_method_data: dict[str, dict[int, dict]],
    all_equilibrium_data: dict[str, dict[int, dict]],
    all_gt_stability_data: dict[str, dict[int, dict]],
    annotations: dict,
) -> str:
    """Generate the complete markdown report."""
    room_ids = annotations["room_ids"]
    house_ids = annotations["house_ids"]
    categories = annotations["categories"]
    difficulties = annotations["difficulties"]

    sections = []

    # Header
    sections.append("# SceneEval Results Comparison")
    sections.append("")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("")

    # Metric legend
    sections.append(generate_metric_legend())
    sections.append("")
    sections.append(generate_equilibrium_legend())
    sections.append("")
    sections.append("---")
    sections.append("")

    # Room scenes section
    sections.append(f"# Room Scenes ({len(room_ids)} scenes)")
    sections.append("")

    # Filter to room methods that have data
    room_methods = [m for m in ROOM_METHODS if m in all_method_data]

    # Scene counts by category
    sections.append(generate_scene_counts_table(all_method_data, room_methods, annotations))
    sections.append("")

    # Overall results for all room scenes
    sections.append(generate_results_table(
        all_method_data, room_methods, room_ids,
        "Overall Results (All Room Scenes)"
    ))
    sections.append("")

    # Results by category
    sections.append("## Results by Category")
    sections.append("")

    category_order = ["SceneEval-100", "type_diversity", "object_density", "themed_scenes"]
    for category in category_order:
        cat_ids = categories.get(category, set())
        sections.append(generate_results_table(
            all_method_data, room_methods, cat_ids,
            f"{category} ({len(cat_ids)} scenes)"
        ))
        sections.append("")

    # Results by difficulty
    sections.append("## Results by Difficulty")
    sections.append("")

    difficulty_order = ["easy", "medium", "hard"]
    for difficulty in difficulty_order:
        diff_ids = difficulties.get(difficulty, set())
        sections.append(generate_results_table(
            all_method_data, room_methods, diff_ids,
            f"{difficulty.capitalize()} ({len(diff_ids)} scenes)"
        ))
        sections.append("")

    # Equilibrium statistics for room scenes
    sections.append(generate_equilibrium_stats_table(
        all_equilibrium_data, room_methods, room_ids,
        "Equilibrium Statistics (All Room Scenes)"
    ))
    sections.append("")

    # House scenes section
    sections.append("---")
    sections.append("")
    sections.append(f"# House Scenes ({len(house_ids)} scenes)")
    sections.append("")

    # Filter to house methods that have data
    house_methods = [m for m in HOUSE_METHODS if m in all_method_data]

    sections.append(generate_results_table(
        all_method_data, house_methods, house_ids,
        "Overall Results (House Scenes)"
    ))
    sections.append("")

    # Equilibrium statistics for house scenes
    sections.append(generate_equilibrium_stats_table(
        all_equilibrium_data, house_methods, house_ids,
        "Equilibrium Statistics (House Scenes)"
    ))
    sections.append("")

    # VLM Support Type Classification Ablation (if GT stability data available)
    if all_gt_stability_data:
        sections.append("---")
        sections.append("")
        sections.append("# VLM Classification Ablation")
        sections.append("")
        sections.append(generate_gt_stability_section(all_gt_stability_data, annotations))
        sections.append("")

    return "\n".join(sections)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading annotations from: {ANNOTATIONS_PATH}")
    annotations = load_annotations(ANNOTATIONS_PATH)

    print(f"Room scenes: {len(annotations['room_ids'])}")
    print(f"House scenes: {len(annotations['house_ids'])}")
    print(f"Categories: {list(annotations['categories'].keys())}")
    print(f"Difficulties: {list(annotations['difficulties'].keys())}")

    # Discover and load all method data
    all_methods = set(ROOM_METHODS) | set(HOUSE_METHODS)
    all_method_data = {}
    all_equilibrium_data = {}
    all_gt_stability_data = {}

    for method in all_methods:
        nested = method in NESTED_METHODS
        print(f"Discovering scenes for: {method} (nested={nested})")

        scene_paths = discover_method_scenes(RESULTS_DIR, method, nested)
        print(f"  Found {len(scene_paths)} scenes with v2 results")

        if not scene_paths:
            continue

        # Load metrics, equilibrium stats, and ground truth stability for each scene
        method_data = {}
        equilibrium_data = {}
        gt_stability_data = {}
        for scene_id, eval_path in scene_paths.items():
            try:
                metrics = load_scene_metrics(eval_path)
                method_data[scene_id] = metrics
                eq_stats = load_scene_equilibrium_stats(eval_path)
                equilibrium_data[scene_id] = eq_stats
                # Load ground truth stability (only for SceneAgent methods)
                gt_stats = load_scene_ground_truth_stability(method, eval_path, scene_id)
                if gt_stats:
                    gt_stability_data[scene_id] = gt_stats
            except Exception as e:
                print(f"  Error loading scene {scene_id}: {e}")

        all_method_data[method] = method_data
        all_equilibrium_data[method] = equilibrium_data
        if gt_stability_data:
            all_gt_stability_data[method] = gt_stability_data
            print(f"  Loaded GT stability for {len(gt_stability_data)} scenes")

    # Generate markdown report
    print(f"\nGenerating markdown report...")
    report = generate_markdown_report(all_method_data, all_equilibrium_data, all_gt_stability_data, annotations)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"Report written to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()

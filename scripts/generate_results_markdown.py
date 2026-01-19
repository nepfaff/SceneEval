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
from statistics import mean


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_results")
ANNOTATIONS_PATH = Path("/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/annotations.csv")
OUTPUT_PATH = Path("/home/ubuntu/SceneEval/results_SceneEval.md")

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
# AGGREGATION
# =============================================================================

def aggregate_metrics(method_data: dict[int, dict], scene_ids: set[int]) -> tuple[dict, int]:
    """
    Aggregate metrics over a subset of scenes.

    Args:
        method_data: dict mapping scene_id -> metrics dict
        scene_ids: set of scene IDs to aggregate over

    Returns:
        (aggregated_metrics, count) where aggregated_metrics has average values
    """
    # Filter to scenes in both method_data and scene_ids
    valid_ids = set(method_data.keys()) & scene_ids
    count = len(valid_ids)

    if count == 0:
        return {}, 0

    # Accumulate values
    value_sums = defaultdict(float)
    value_counts = defaultdict(int)

    for scene_id in valid_ids:
        metrics = method_data[scene_id]
        for abbrev, value in metrics.items():
            if value is not None:
                value_sums[abbrev] += value
                value_counts[abbrev] += 1

    # Compute averages
    averages = {}
    for abbrev in value_sums:
        if value_counts[abbrev] > 0:
            averages[abbrev] = value_sums[abbrev] / value_counts[abbrev]

    return averages, count


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================

def format_value(value: float | None, abbrev: str) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"

    if abbrev == "#Obj":
        return f"{value:.1f}"
    else:
        return f"{value:.1%}"


def format_equilibrium_value(value: float | None, abbrev: str) -> str:
    """Format an equilibrium statistic value for display."""
    if value is None:
        return "-"

    # Handle NaN and infinity
    if math.isnan(value) or math.isinf(value):
        return "-"

    # Detect value type from abbreviation suffix
    # D = displacement (meters), R = rotation (radians)
    if abbrev.endswith("D"):
        # Displacement in meters - show in mm if < 1m, otherwise m
        if abs(value) < 0.001:
            return f"{value*1000:.3f}mm"
        elif abs(value) < 1.0:
            return f"{value*1000:.1f}mm"
        else:
            return f"{value:.3f}m"
    elif abbrev.endswith("R"):
        # Rotation in radians
        if abs(value) < 0.001:
            return f"{value:.4f}rad"
        else:
            return f"{value:.3f}rad"
    else:
        return f"{value:.4f}"


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

    return "\n".join(lines)


def generate_equilibrium_legend() -> str:
    """Generate the equilibrium statistics legend section."""
    lines = [
        "## Equilibrium Statistics Legend",
        "",
        "Statistics are computed from physics simulation displacement/rotation values.",
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


def generate_markdown_report(
    all_method_data: dict[str, dict[int, dict]],
    all_equilibrium_data: dict[str, dict[int, dict]],
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

    for method in all_methods:
        nested = method in NESTED_METHODS
        print(f"Discovering scenes for: {method} (nested={nested})")

        scene_paths = discover_method_scenes(RESULTS_DIR, method, nested)
        print(f"  Found {len(scene_paths)} scenes with v2 results")

        if not scene_paths:
            continue

        # Load metrics and equilibrium stats for each scene
        method_data = {}
        equilibrium_data = {}
        for scene_id, eval_path in scene_paths.items():
            try:
                metrics = load_scene_metrics(eval_path)
                method_data[scene_id] = metrics
                eq_stats = load_scene_equilibrium_stats(eval_path)
                equilibrium_data[scene_id] = eq_stats
            except Exception as e:
                print(f"  Error loading scene {scene_id}: {e}")

        all_method_data[method] = method_data
        all_equilibrium_data[method] = equilibrium_data

    # Generate markdown report
    print(f"\nGenerating markdown report...")
    report = generate_markdown_report(all_method_data, all_equilibrium_data, annotations)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"Report written to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()

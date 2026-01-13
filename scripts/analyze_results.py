#!/usr/bin/env python3
"""
analyze_results.py - Analyze SceneEval results with auto-discovery

Automatically discovers methods and scenes from a results directory,
extracts all metrics, reports scene coverage, and outputs summary statistics.

Usage:
    python analyze_results.py /path/to/results_dir [--output metrics.csv]
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Try to import tabulate for pretty tables, fall back to simple formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# =============================================================================
# METRIC CONFIGURATION
# =============================================================================

METRIC_COLUMNS = [
    # VLM-based semantic metrics
    ("ObjCountMetric", "satisfaction_rate"),
    ("ObjAttributeMetric", "satisfaction_rate"),
    ("ObjObjRelationshipMetric", "satisfaction_rate"),
    ("ObjArchRelationshipMetric", "satisfaction_rate"),
    ("SupportMetric", "satisfaction_rate"),
    ("AccessibilityMetric", "avg_accessibility"),
    # Common metrics
    ("CollisionMetric", "num_obj_in_collision"),
    ("CollisionMetric", "frac_obj_in_collision"),
    ("NavigabilityMetric", "navigability"),
    ("OutOfBoundMetric", "num_out_of_bound"),
    ("OutOfBoundMetric", "frac_out_of_bound"),
    ("OpeningClearanceMetric", "doors_blocked"),
    # Drake collision metrics (variants)
    ("DrakeCollisionMetricVHACD", "frac_obj_in_collision"),
    ("DrakeCollisionMetricCoACD", "frac_obj_in_collision"),
    ("DrakeCollisionMetricSceneAgent", "frac_obj_in_collision"),
    # Architectural equilibrium metrics
    ("ArchitecturalWeldedEquilibriumMetricVHACD", "frac_unstable_objects"),
    ("ArchitecturalWeldedEquilibriumMetricCoACD", "frac_unstable_objects"),
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "frac_unstable_objects"),
    # Combined equilibrium metrics
    ("CombinedWeldedEquilibriumMetricVHACD", "frac_unstable_objects"),
    ("CombinedWeldedEquilibriumMetricCoACD", "frac_unstable_objects"),
    ("CombinedWeldedEquilibriumMetricSceneAgent", "frac_unstable_objects"),
]

# Short names for console display
METRIC_SHORT_NAMES = {
    "ObjCountMetric.satisfaction_rate": "CNT",
    "ObjAttributeMetric.satisfaction_rate": "ATR",
    "ObjObjRelationshipMetric.satisfaction_rate": "OOR",
    "ObjArchRelationshipMetric.satisfaction_rate": "OAR",
    "SupportMetric.satisfaction_rate": "SUP",
    "AccessibilityMetric.avg_accessibility": "ACC",
    "CollisionMetric.frac_obj_in_collision": "COL",
    "NavigabilityMetric.navigability": "NAV",
    "OutOfBoundMetric.frac_out_of_bound": "OOB",
    "DrakeCollisionMetricVHACD.frac_obj_in_collision": "DC-V",
    "DrakeCollisionMetricCoACD.frac_obj_in_collision": "DC-C",
    "DrakeCollisionMetricSceneAgent.frac_obj_in_collision": "DC-S",
    "ArchitecturalWeldedEquilibriumMetricVHACD.frac_unstable_objects": "AW-V",
    "ArchitecturalWeldedEquilibriumMetricCoACD.frac_unstable_objects": "AW-C",
    "ArchitecturalWeldedEquilibriumMetricSceneAgent.frac_unstable_objects": "AW-S",
    "CombinedWeldedEquilibriumMetricVHACD.frac_unstable_objects": "CW-V",
    "CombinedWeldedEquilibriumMetricCoACD.frac_unstable_objects": "CW-C",
    "CombinedWeldedEquilibriumMetricSceneAgent.frac_unstable_objects": "CW-S",
}


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================

def discover_methods_and_scenes(results_dir: Path) -> dict[str, set[str]]:
    """
    Discover all methods and their scenes from a results directory.

    Returns:
        Dict mapping method_name -> set of scene names
    """
    method_scenes = defaultdict(set)

    # Find all eval_result.json files
    for eval_file in results_dir.glob("*/scene_*/eval_result.json"):
        method_name = eval_file.parent.parent.name
        scene_name = eval_file.parent.name
        method_scenes[method_name].add(scene_name)

    return dict(method_scenes)


def get_all_scenes(method_scenes: dict[str, set[str]]) -> set[str]:
    """Get union of all scenes across all methods."""
    all_scenes = set()
    for scenes in method_scenes.values():
        all_scenes.update(scenes)
    return all_scenes


def natural_sort_key(s: str):
    """Sort key for natural sorting (scene_2 before scene_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_metric_value(results: dict, metric_name: str, field: str, num_objects: int = 0):
    """Extract a specific field value from a metric's results."""
    if metric_name not in results:
        return None

    metric_result = results[metric_name]
    metric_data = metric_result.get("data", {})

    # Handle satisfaction_rate computed from message
    if field == "satisfaction_rate":
        message = metric_result.get("message", "")
        match = re.search(r"(\d+)/(\d+)", message)
        if match:
            num, denom = int(match.group(1)), int(match.group(2))
            if denom > 0:
                return round(num / denom, 4)
        if "No " in message and "to evaluate" in message:
            return None
        return None

    # Handle AccessibilityMetric avg_accessibility
    if metric_name == "AccessibilityMetric" and field == "avg_accessibility":
        scores = []
        for obj_id, obj_data in metric_data.items():
            if isinstance(obj_data, dict) and "max" in obj_data:
                max_score = obj_data["max"]
                if max_score >= 0:
                    scores.append(max_score)
        if scores:
            return round(sum(scores) / len(scores), 4)
        return None

    # Handle OutOfBoundMetric
    if metric_name == "OutOfBoundMetric" and field == "num_out_of_bound":
        count = sum(1 for obj_data in metric_data.values()
                   if isinstance(obj_data, dict) and obj_data.get("out_of_bound", False))
        return count

    if metric_name == "OutOfBoundMetric" and field == "frac_out_of_bound":
        count = sum(1 for obj_data in metric_data.values()
                   if isinstance(obj_data, dict) and obj_data.get("out_of_bound", False))
        return round(count / num_objects, 4) if num_objects > 0 else None

    # Handle OpeningClearanceMetric
    if metric_name == "OpeningClearanceMetric" and field == "doors_blocked":
        door_clearance = metric_data.get("door_clearance", {})
        blocked = 0
        for door_name, directions in door_clearance.items():
            for direction_data in directions:
                if direction_data.get("interfering_obj_ids", []):
                    blocked += 1
                    break
        return blocked

    # Fraction of objects in collision
    if field == "frac_obj_in_collision":
        num_in_collision = metric_data.get("num_obj_in_collision", 0)
        return round(num_in_collision / num_objects, 4) if num_objects > 0 else None

    # Fraction of unstable objects
    if field == "frac_unstable_objects":
        num_stable = metric_data.get("num_stable_objects", 0)
        num_unstable = metric_data.get("num_unstable_objects", 0)
        total_simulated = num_stable + num_unstable
        return round(num_unstable / total_simulated, 4) if total_simulated > 0 else None

    # Direct field access
    if field in metric_data:
        return metric_data[field]

    return None


def load_eval_result(results_dir: Path, method: str, scene: str) -> tuple[dict, int]:
    """Load eval_result.json for a method/scene combination."""
    eval_path = results_dir / method / scene / "eval_result.json"
    if not eval_path.exists():
        return {}, 0

    with open(eval_path, "r") as f:
        data = json.load(f)

    num_objects = len(data.get("obj_ids", []))
    return data.get("results", {}), num_objects


# =============================================================================
# AGGREGATION
# =============================================================================

def compute_method_averages(rows: list[dict]) -> dict:
    """Compute average values across rows for a method."""
    averages = {"num_objects": 0}

    # Initialize accumulators
    value_sums = defaultdict(float)
    value_counts = defaultdict(int)

    for row in rows:
        if row.get("num_objects"):
            value_sums["num_objects"] += row["num_objects"]
            value_counts["num_objects"] += 1

        for metric_name, field in METRIC_COLUMNS:
            col_key = f"{metric_name}.{field}"
            val = row.get(col_key)
            if val is not None:
                value_sums[col_key] += val
                value_counts[col_key] += 1

    # Compute averages
    for key in value_sums:
        if value_counts[key] > 0:
            averages[key] = round(value_sums[key] / value_counts[key], 4)

    return averages


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_scene_coverage(method_scenes: dict[str, set[str]], all_scenes: set[str]):
    """Print scene coverage report."""
    print("\n" + "=" * 80)
    print("SCENE COVERAGE REPORT")
    print("=" * 80)

    sorted_methods = sorted(method_scenes.keys())
    sorted_all_scenes = sorted(all_scenes, key=natural_sort_key)

    print(f"\nTotal unique scenes across all methods: {len(all_scenes)}")
    print(f"Total methods: {len(method_scenes)}")

    # Coverage table
    coverage_data = []
    for method in sorted_methods:
        scenes = method_scenes[method]
        missing = all_scenes - scenes
        missing_count = len(missing)
        coverage_pct = len(scenes) / len(all_scenes) * 100 if all_scenes else 0

        # Truncate missing scenes list for display
        missing_list = sorted(missing, key=natural_sort_key)
        if len(missing_list) > 5:
            missing_str = ", ".join(missing_list[:5]) + f", ... (+{len(missing_list)-5} more)"
        elif missing_list:
            missing_str = ", ".join(missing_list)
        else:
            missing_str = "(none)"

        coverage_data.append([
            method,
            len(scenes),
            f"{coverage_pct:.1f}%",
            missing_count,
            missing_str
        ])

    headers = ["Method", "Scenes", "Coverage", "Missing", "Missing Scenes"]

    if HAS_TABULATE:
        print("\n" + tabulate(coverage_data, headers=headers, tablefmt="simple"))
    else:
        # Simple table formatting
        print("\n" + "\t".join(headers))
        print("-" * 100)
        for row in coverage_data:
            print("\t".join(str(x) for x in row))


def print_metric_summary(method_averages: dict[str, dict]):
    """Print metric summary table."""
    print("\n" + "=" * 80)
    print("METRIC SUMMARY (Per-Method Averages)")
    print("=" * 80)

    # Select key metrics for display
    display_metrics = [
        ("num_objects", "#Obj"),
        ("ObjCountMetric.satisfaction_rate", "CNT"),
        ("ObjAttributeMetric.satisfaction_rate", "ATR"),
        ("ObjObjRelationshipMetric.satisfaction_rate", "OOR"),
        ("ObjArchRelationshipMetric.satisfaction_rate", "OAR"),
        ("SupportMetric.satisfaction_rate", "SUP"),
        ("AccessibilityMetric.avg_accessibility", "ACC"),
        ("CollisionMetric.frac_obj_in_collision", "COL"),
        ("NavigabilityMetric.navigability", "NAV"),
        ("OutOfBoundMetric.frac_out_of_bound", "OOB"),
    ]

    sorted_methods = sorted(method_averages.keys())

    table_data = []
    for method in sorted_methods:
        avgs = method_averages[method]
        row = [method]
        for metric_key, _ in display_metrics:
            val = avgs.get(metric_key)
            if val is None:
                row.append("-")
            elif metric_key == "num_objects":
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.2%}")
        table_data.append(row)

    headers = ["Method"] + [short for _, short in display_metrics]

    if HAS_TABULATE:
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
    else:
        print("\n" + "\t".join(headers))
        print("-" * 120)
        for row in table_data:
            print("\t".join(str(x) for x in row))

    print("\nMetric Legend:")
    print("  CNT=Count, ATR=Attribute, OOR=Obj-Obj Rel, OAR=Obj-Arch Rel")
    print("  SUP=Support, ACC=Accessibility, COL=Collision, NAV=Navigability, OOB=OutOfBound")
    print("  Higher is better: CNT, ATR, OOR, OAR, SUP, ACC, NAV")
    print("  Lower is better: COL, OOB")


def write_csv(output_path: Path, results_dir: Path, method_scenes: dict[str, set[str]],
              method_averages: dict[str, dict]):
    """Write full results to CSV."""

    # Build header
    header = ["method", "scene", "num_objects"]
    for metric_name, field in METRIC_COLUMNS:
        header.append(f"{metric_name}.{field}")

    rows = []
    sorted_methods = sorted(method_scenes.keys())

    for i, method in enumerate(sorted_methods):
        if i > 0:
            rows.append([])  # Empty row between methods

        sorted_scenes = sorted(method_scenes[method], key=natural_sort_key)

        for scene in sorted_scenes:
            results, num_objects = load_eval_result(results_dir, method, scene)

            row = [method, scene, num_objects]
            for metric_name, field in METRIC_COLUMNS:
                value = extract_metric_value(results, metric_name, field, num_objects)
                row.append(value if value is not None else "")
            rows.append(row)

        # Add average row
        avgs = method_averages[method]
        avg_row = [method, "Average", avgs.get("num_objects", "")]
        for metric_name, field in METRIC_COLUMNS:
            col_key = f"{metric_name}.{field}"
            val = avgs.get(col_key)
            avg_row.append(val if val is not None else "")
        rows.append(avg_row)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nCSV written to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SceneEval results with auto-discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_results.py /path/to/results
    python analyze_results.py /path/to/results --output metrics.csv
    python analyze_results.py /path/to/results --no-csv
        """
    )
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output CSV path (default: results_dir/metric_summary.csv)")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV generation")

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover methods and scenes
    print(f"Scanning: {args.results_dir}")
    method_scenes = discover_methods_and_scenes(args.results_dir)

    if not method_scenes:
        print("Error: No methods found. Looking for */scene_*/eval_result.json", file=sys.stderr)
        sys.exit(1)

    all_scenes = get_all_scenes(method_scenes)

    # Print coverage report
    print_scene_coverage(method_scenes, all_scenes)

    # Extract metrics and compute averages
    print("\nExtracting metrics...")
    method_averages = {}

    for method, scenes in method_scenes.items():
        method_rows = []
        for scene in scenes:
            results, num_objects = load_eval_result(args.results_dir, method, scene)

            row = {"num_objects": num_objects}
            for metric_name, field in METRIC_COLUMNS:
                col_key = f"{metric_name}.{field}"
                row[col_key] = extract_metric_value(results, metric_name, field, num_objects)
            method_rows.append(row)

        method_averages[method] = compute_method_averages(method_rows)

    # Print metric summary
    print_metric_summary(method_averages)

    # Write CSV
    if not args.no_csv:
        output_path = args.output or (args.results_dir / "metric_summary.csv")
        write_csv(output_path, args.results_dir, method_scenes, method_averages)

    print("\nDone!")


if __name__ == "__main__":
    main()

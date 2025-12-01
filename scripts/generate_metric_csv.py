#!/usr/bin/env python3
"""
generate_metric_csv.py - Generate CSV summary of evaluation metrics

Creates a CSV file with all evaluation metrics for each method/scene combination.
Handles different metric variants (CoACD/VHACD for most methods, SceneAgent-specific).
"""

import csv
import json
import os
import re
from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/SceneEval/output_eval")
OUTPUT_CSV = OUTPUT_DIR / "metric_summary.csv"

METHODS = ["HSM", "Holodeck", "IDesign", "LayoutVLM", "SceneAgent", "SceneWeaver"]
SCENES = ["scene_39", "scene_56", "scene_74", "scene_94", "scene_106"]

# Define all metric columns - will extract specific values from each
METRIC_COLUMNS = [
    # VLM-based semantic metrics
    ("ObjCountMetric", "satisfaction_rate"),  # Computed from message
    ("ObjAttributeMetric", "satisfaction_rate"),  # Computed from message
    ("ObjObjRelationshipMetric", "satisfaction_rate"),  # Computed from message
    ("ObjArchRelationshipMetric", "satisfaction_rate"),  # Computed from message
    ("SupportMetric", "satisfaction_rate"),  # Computed from message
    ("AccessibilityMetric", "avg_accessibility"),  # Computed: average max score (excluding -1)
    # Common metrics
    ("CollisionMetric", "num_obj_in_collision"),
    ("CollisionMetric", "frac_obj_in_collision"),  # Computed: over total objects
    ("CollisionMetric", "max_penetration_depth"),
    ("NavigabilityMetric", "navigability"),
    ("OutOfBoundMetric", "num_out_of_bound"),  # Computed
    ("OutOfBoundMetric", "frac_out_of_bound"),  # Computed: over total objects
    ("OpeningClearanceMetric", "doors_blocked"),  # Computed
    # Drake collision metrics (CoACD/VHACD for most, SceneAgent variant)
    ("DrakeCollisionMetricCoACD", "num_obj_in_collision"),
    ("DrakeCollisionMetricCoACD", "frac_obj_in_collision"),  # Computed
    ("DrakeCollisionMetricCoACD", "max_penetration_depth"),
    ("DrakeCollisionMetricVHACD", "num_obj_in_collision"),
    ("DrakeCollisionMetricVHACD", "frac_obj_in_collision"),  # Computed
    ("DrakeCollisionMetricVHACD", "max_penetration_depth"),
    ("DrakeCollisionMetricSceneAgent", "num_obj_in_collision"),
    ("DrakeCollisionMetricSceneAgent", "frac_obj_in_collision"),  # Computed
    ("DrakeCollisionMetricSceneAgent", "max_penetration_depth"),
    # Architectural equilibrium metrics
    ("ArchitecturalWeldedEquilibriumMetricCoACD", "scene_stable"),
    ("ArchitecturalWeldedEquilibriumMetricCoACD", "num_unstable_objects"),
    ("ArchitecturalWeldedEquilibriumMetricCoACD", "frac_unstable_objects"),  # Computed: over simulated
    ("ArchitecturalWeldedEquilibriumMetricCoACD", "max_displacement"),
    ("ArchitecturalWeldedEquilibriumMetricVHACD", "scene_stable"),
    ("ArchitecturalWeldedEquilibriumMetricVHACD", "num_unstable_objects"),
    ("ArchitecturalWeldedEquilibriumMetricVHACD", "frac_unstable_objects"),  # Computed: over simulated
    ("ArchitecturalWeldedEquilibriumMetricVHACD", "max_displacement"),
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "scene_stable"),
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "num_unstable_objects"),
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "frac_unstable_objects"),  # Computed: over simulated
    ("ArchitecturalWeldedEquilibriumMetricSceneAgent", "max_displacement"),
    # Combined equilibrium metrics
    ("CombinedWeldedEquilibriumMetricCoACD", "scene_stable"),
    ("CombinedWeldedEquilibriumMetricCoACD", "num_unstable_objects"),
    ("CombinedWeldedEquilibriumMetricCoACD", "frac_unstable_objects"),  # Computed: over simulated
    ("CombinedWeldedEquilibriumMetricCoACD", "max_displacement"),
    ("CombinedWeldedEquilibriumMetricVHACD", "scene_stable"),
    ("CombinedWeldedEquilibriumMetricVHACD", "num_unstable_objects"),
    ("CombinedWeldedEquilibriumMetricVHACD", "frac_unstable_objects"),  # Computed: over simulated
    ("CombinedWeldedEquilibriumMetricVHACD", "max_displacement"),
    ("CombinedWeldedEquilibriumMetricSceneAgent", "scene_stable"),
    ("CombinedWeldedEquilibriumMetricSceneAgent", "num_unstable_objects"),
    ("CombinedWeldedEquilibriumMetricSceneAgent", "frac_unstable_objects"),  # Computed: over simulated
    ("CombinedWeldedEquilibriumMetricSceneAgent", "max_displacement"),
]


def extract_metric_value(results: dict, metric_name: str, field: str, num_objects: int = 0):
    """Extract a specific field value from a metric's results."""
    if metric_name not in results:
        return ""

    metric_result = results[metric_name]
    metric_data = metric_result.get("data", {})

    # Handle satisfaction_rate computed from message (e.g., "2/3 requirements are satisfied")
    if field == "satisfaction_rate":
        message = metric_result.get("message", "")
        # Match patterns like "2/3 requirements" or "12/15 objects"
        match = re.search(r"(\d+)/(\d+)", message)
        if match:
            num, denom = int(match.group(1)), int(match.group(2))
            if denom > 0:
                return round(num / denom, 4)
        # If "No ... specs to evaluate" return empty (not applicable)
        if "No " in message and "to evaluate" in message:
            return ""
        return ""

    # Handle AccessibilityMetric avg_accessibility (average of max scores, excluding -1)
    if metric_name == "AccessibilityMetric" and field == "avg_accessibility":
        scores = []
        for obj_id, obj_data in metric_data.items():
            if isinstance(obj_data, dict) and "max" in obj_data:
                max_score = obj_data["max"]
                if max_score >= 0:  # Exclude -1 (no applicable functional sides)
                    scores.append(max_score)
        if scores:
            return round(sum(scores) / len(scores), 4)
        return ""

    # Handle computed fields
    if metric_name == "OutOfBoundMetric" and field == "num_out_of_bound":
        # Count objects that are out of bounds
        count = 0
        for obj_id, obj_data in metric_data.items():
            if isinstance(obj_data, dict) and obj_data.get("out_of_bound", False):
                count += 1
        return count

    if metric_name == "OutOfBoundMetric" and field == "frac_out_of_bound":
        # Fraction of objects out of bounds
        count = 0
        for obj_id, obj_data in metric_data.items():
            if isinstance(obj_data, dict) and obj_data.get("out_of_bound", False):
                count += 1
        return round(count / num_objects, 4) if num_objects > 0 else ""

    if metric_name == "OpeningClearanceMetric" and field == "doors_blocked":
        # Count blocked doors (doors with interfering objects in any direction)
        door_clearance = metric_data.get("door_clearance", {})
        blocked = 0
        for door_name, directions in door_clearance.items():
            for direction_data in directions:
                if direction_data.get("interfering_obj_ids", []):
                    blocked += 1
                    break  # Count each door only once
        return blocked

    # Fraction of objects in collision (over total objects)
    if field == "frac_obj_in_collision":
        num_in_collision = metric_data.get("num_obj_in_collision", 0)
        return round(num_in_collision / num_objects, 4) if num_objects > 0 else ""

    # Fraction of unstable objects (over total simulated = stable + unstable)
    if field == "frac_unstable_objects":
        num_stable = metric_data.get("num_stable_objects", 0)
        num_unstable = metric_data.get("num_unstable_objects", 0)
        total_simulated = num_stable + num_unstable
        return round(num_unstable / total_simulated, 4) if total_simulated > 0 else ""

    # Direct field access
    if field in metric_data:
        return metric_data[field]

    return ""


def load_eval_result(method: str, scene: str) -> tuple[dict, int]:
    """Load eval_result.json for a method/scene combination.

    Returns:
        Tuple of (results dict, num_objects)
    """
    eval_path = OUTPUT_DIR / method / scene / "eval_result.json"
    if not eval_path.exists():
        return {}, 0

    with open(eval_path, "r") as f:
        data = json.load(f)

    num_objects = len(data.get("obj_ids", []))
    return data.get("results", {}), num_objects


def compute_average_row(method: str, scene_rows: list) -> list:
    """Compute average values across scene rows for a method."""
    avg_row = [method, "Average"]

    # Average num_objects (column index 2)
    num_obj_values = [row[2] for row in scene_rows if isinstance(row[2], (int, float))]
    if num_obj_values:
        avg = sum(num_obj_values) / len(num_obj_values)
        avg_row.append(round(avg, 1) if avg != int(avg) else int(avg))
    else:
        avg_row.append("")

    # Average metric columns (starting at index 3)
    num_cols = len(METRIC_COLUMNS)
    for col_idx in range(num_cols):
        values = []
        for row in scene_rows:
            val = row[col_idx + 3]  # +3 to skip method, scene, num_objects columns
            if isinstance(val, (int, float)):
                values.append(val)
            elif isinstance(val, bool):
                values.append(1 if val else 0)  # Convert bool to numeric for averaging

        if values:
            avg = sum(values) / len(values)
            # Round to reasonable precision
            avg_row.append(round(avg, 4) if avg != int(avg) else int(avg))
        else:
            avg_row.append("")

    return avg_row


def generate_csv():
    """Generate the metric summary CSV."""
    # Build header
    header = ["method", "scene", "num_objects"]
    for metric_name, field in METRIC_COLUMNS:
        header.append(f"{metric_name}.{field}")

    rows = []

    for i, method in enumerate(METHODS):
        # Add empty row between methods (except before first)
        if i > 0:
            rows.append([])

        method_rows = []
        for scene in SCENES:
            results, num_objects = load_eval_result(method, scene)

            row = [method, scene, num_objects]
            for metric_name, field in METRIC_COLUMNS:
                value = extract_metric_value(results, metric_name, field, num_objects)
                row.append(value)

            rows.append(row)
            method_rows.append(row)

        # Add average row for this method
        avg_row = compute_average_row(method, method_rows)
        rows.append(avg_row)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Generated: {OUTPUT_CSV}")
    print(f"Total rows: {len(rows)} ({len(METHODS)} methods × {len(SCENES)} scenes + {len(METHODS)} averages + {len(METHODS)-1} separator rows)")


if __name__ == "__main__":
    generate_csv()

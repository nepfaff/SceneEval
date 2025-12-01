#!/usr/bin/env python3
"""
generate_metric_table.py - Generate markdown table and matplotlib figure from metric CSV

Reads the metric summary CSV and creates:
1. A markdown table with methods as rows and metrics as columns
2. A matplotlib figure with the table and legend
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION: Metrics to include and their 3-letter acronyms
# Based on abbreviations from https://arxiv.org/html/2503.16848v2
# =============================================================================
METRIC_CONFIG = {
    # Format: "csv_column_name": ("acronym", "full_description", "group")
    # Group is used for ranking - metrics in same group are compared together
    # None group means no ranking (descriptive metric only)
    # VLM-based semantic metrics (from HSM paper - higher is better)
    "num_objects": ("#OB", "Num. Objects (per scene)", None),
    "ObjCountMetric.satisfaction_rate": ("CNT", "Obj. Count Satisfaction", "CNT"),
    "ObjAttributeMetric.satisfaction_rate": ("ATR", "Object Attribute", "ATR"),
    "ObjObjRelationshipMetric.satisfaction_rate": ("OOR", "Object-Object Relationship", "OOR"),
    "ObjArchRelationshipMetric.satisfaction_rate": ("OAR", "Object-Arch Relationship", "OAR"),
    "SupportMetric.satisfaction_rate": ("SUP", "Object Support", "SUP"),
    "AccessibilityMetric.avg_accessibility": ("ACC", "Object Accessibility", "ACC"),
    # Common metrics (from HSM paper)
    "CollisionMetric.frac_obj_in_collision": ("COL", "Object Collision Rate", "COL"),
    "NavigabilityMetric.navigability": ("NAV", "Scene Navigability", "NAV"),
    "OutOfBoundMetric.frac_out_of_bound": ("OOB", "Out-of-Bound Rate", "OOB"),
    # Drake collision metrics (grouped together for ranking)
    "DrakeCollisionMetricCoACD.frac_obj_in_collision": ("DCC", "Drake Collision (CoACD)", "DC"),
    "DrakeCollisionMetricVHACD.frac_obj_in_collision": ("DCV", "Drake Collision (VHACD)", "DC"),
    "DrakeCollisionMetricSceneAgent.frac_obj_in_collision": ("DCS", "Drake Collision (Ours)", "DC"),
    # Architectural equilibrium metrics (grouped together for ranking)
    "ArchitecturalWeldedEquilibriumMetricCoACD.frac_unstable_objects": ("AWC", "Arch. Equilibrium (CoACD)", "AW"),
    "ArchitecturalWeldedEquilibriumMetricVHACD.frac_unstable_objects": ("AWV", "Arch. Equilibrium (VHACD)", "AW"),
    "ArchitecturalWeldedEquilibriumMetricSceneAgent.frac_unstable_objects": ("AWS", "Arch. Equilibrium (Ours)", "AW"),
    # Combined equilibrium metrics (grouped together for ranking)
    "CombinedWeldedEquilibriumMetricCoACD.frac_unstable_objects": ("CWC", "Comb. Equilibrium (CoACD)", "CW"),
    "CombinedWeldedEquilibriumMetricVHACD.frac_unstable_objects": ("CWV", "Comb. Equilibrium (VHACD)", "CW"),
    "CombinedWeldedEquilibriumMetricSceneAgent.frac_unstable_objects": ("CWS", "Comb. Equilibrium (Ours)", "CW"),
}

# Define which metric groups are "higher is better" vs "lower is better"
# Semantic metrics (CNT, ATR, OOR, OAR, SUP, ACC) and NAV are higher is better
HIGHER_IS_BETTER = {"CNT", "ATR", "OOR", "OAR", "SUP", "ACC", "NAV"}

# Method display order and renaming
METHOD_ORDER = ["HSM", "Holodeck", "IDesign", "LayoutVLM", "SceneWeaver", "SceneAgent"]
METHOD_DISPLAY_NAMES = {
    "HSM": "HSM",
    "Holodeck": "Holodeck",
    "IDesign": "IDesign",
    "LayoutVLM": "LayoutVLM",
    "SceneWeaver": "SceneWeaver",
    "SceneAgent": "Ours",
}

INPUT_CSV = Path("/home/ubuntu/SceneEval/output_eval/metric_summary.csv")
OUTPUT_MD = Path("/home/ubuntu/SceneEval/output_eval/metric_table.md")
OUTPUT_PNG = Path("/home/ubuntu/SceneEval/output_eval/metric_table.png")


def load_averages(csv_path: Path) -> dict:
    """Load average rows from CSV, keyed by method name."""
    averages = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scene") == "Average":
                method = row["method"]
                averages[method] = row
    return averages


def get_numeric_value(val: str) -> float | None:
    """Convert string value to float, return None if invalid."""
    if val == "" or val is None or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def format_value(val: str, metric_name: str) -> str:
    """Format a metric value for display."""
    if val == "" or val is None:
        return "-"
    try:
        num = float(val)
        # num_objects should be displayed as a number, not percentage
        if metric_name == "num_objects":
            return f"{num:.1f}"
        return f"{num:.2%}"
    except ValueError:
        return val


def compute_rankings(averages: dict) -> dict:
    """
    Compute best and 2nd best for each metric group.
    Returns dict: {(method, metric_col): rank} where rank is 1 (best) or 2 (2nd best)
    """
    rankings = {}

    # Get unique groups (skip None groups - they're descriptive only)
    groups = {}
    for metric_col, (acronym, desc, group) in METRIC_CONFIG.items():
        if group is None:
            continue  # Skip descriptive metrics (no ranking)
        if group not in groups:
            groups[group] = []
        groups[group].append(metric_col)

    # For each group, find best value per method, then rank methods
    for group, metric_cols in groups.items():
        method_values = {}  # method -> (best_value, metric_col)

        for method in METHOD_ORDER:
            if method not in averages:
                continue
            row_data = averages[method]

            # Find best available value for this method in this group
            best_val = None
            best_col = None
            for metric_col in metric_cols:
                val = get_numeric_value(row_data.get(metric_col, ""))
                if val is not None:
                    if best_val is None:
                        best_val = val
                        best_col = metric_col
                    else:
                        # For grouped metrics, pick the one that exists
                        # (each method typically has only one variant)
                        best_val = val
                        best_col = metric_col

            if best_val is not None:
                method_values[method] = (best_val, best_col)

        # Rank methods for this group
        if not method_values:
            continue

        # Determine if higher or lower is better
        higher_better = group in HIGHER_IS_BETTER

        # Sort methods by value
        sorted_methods = sorted(
            method_values.items(),
            key=lambda x: x[1][0],
            reverse=higher_better
        )

        # Find best value and all methods with that value
        if len(sorted_methods) >= 1:
            best_val = sorted_methods[0][1][0]
            best_methods = [(m, col) for m, (v, col) in sorted_methods if v == best_val]

            # Mark all tied-for-best as rank 1
            for method, col in best_methods:
                rankings[(method, col)] = 1

            # Only assign 2nd best if there's exactly one best (no tie)
            if len(best_methods) == 1 and len(sorted_methods) >= 2:
                # Find 2nd best value
                second_val = sorted_methods[1][1][0]
                second_methods = [(m, col) for m, (v, col) in sorted_methods if v == second_val and v != best_val]

                # Mark all tied-for-second as rank 2
                for method, col in second_methods:
                    rankings[(method, col)] = 2

    return rankings


def generate_markdown_table(averages: dict) -> str:
    """Generate markdown table string."""
    rankings = compute_rankings(averages)

    # Header
    acronyms = [METRIC_CONFIG[m][0] for m in METRIC_CONFIG.keys()]
    header = "| Method | " + " | ".join(acronyms) + " |"
    separator = "|:---" + "|:---:" * len(acronyms) + "|"

    # Rows
    rows = []
    for method in METHOD_ORDER:
        if method not in averages:
            continue
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        row_data = averages[method]

        values = []
        for metric_col in METRIC_CONFIG.keys():
            val = row_data.get(metric_col, "")
            formatted = format_value(val, metric_col)

            # Apply bold/italic based on ranking
            rank = rankings.get((method, metric_col))
            if rank == 1:
                formatted = f"**{formatted}**"
            elif rank == 2:
                formatted = f"*{formatted}*"

            values.append(formatted)

        row = f"| {display_name} | " + " | ".join(values) + " |"
        rows.append(row)

    # Legend
    legend_lines = ["\n### Metric Legend\n"]
    for metric_col, (acronym, description, group) in METRIC_CONFIG.items():
        legend_lines.append(f"- **{acronym}**: {description}")

    table = "\n".join([header, separator] + rows)
    legend = "\n".join(legend_lines)

    return table + "\n" + legend


def generate_matplotlib_figure(averages: dict):
    """Generate matplotlib figure with table and legend."""
    rankings = compute_rankings(averages)

    # Prepare data
    methods = []
    method_keys = []  # Keep original method names for ranking lookup
    data = []

    for method in METHOD_ORDER:
        if method not in averages:
            continue
        methods.append(METHOD_DISPLAY_NAMES.get(method, method))
        method_keys.append(method)
        row_data = averages[method]

        row_values = []
        for metric_col in METRIC_CONFIG.keys():
            val = row_data.get(metric_col, "")
            row_values.append(format_value(val, metric_col))
        data.append(row_values)

    acronyms = [METRIC_CONFIG[m][0] for m in METRIC_CONFIG.keys()]
    metric_cols = list(METRIC_CONFIG.keys())

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))

    # Table subplot
    ax_table = fig.add_subplot(2, 1, 1)
    ax_table.axis("off")

    # Create table
    table = ax_table.table(
        cellText=data,
        rowLabels=methods,
        colLabels=acronyms,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Define colors for each metric group (different shades)
    # None group uses gray (descriptive/no ranking)
    GROUP_COLORS = {
        None: "#616161",  # Gray for descriptive metrics
        # Semantic metrics (green shades - higher is better)
        "CNT": "#1B5E20",  # Darkest green
        "ATR": "#2E7D32",  # Dark green
        "OOR": "#388E3C",  # Medium green
        "OAR": "#43A047",  # Light green
        "SUP": "#4CAF50",  # Lighter green
        "ACC": "#66BB6A",  # Lightest green
        # Plausibility metrics (blue shades)
        "COL": "#1F4E79",  # Dark blue
        "NAV": "#2E75B6",  # Medium blue
        "OOB": "#5B9BD5",  # Light blue
        # Drake metrics (blue-purple shades)
        "DC": "#2F5496",   # Dark blue-purple
        "AW": "#4472C4",   # Standard blue
        "CW": "#7030A0",   # Purple
    }

    # Style header row with group-based colors
    for j, metric_col in enumerate(metric_cols):
        group = METRIC_CONFIG[metric_col][2]
        color = GROUP_COLORS.get(group, "#4472C4")
        table[(0, j)].set_facecolor(color)
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Style row labels
    for i, method in enumerate(methods):
        table[(i + 1, -1)].set_facecolor("#D9E2F3")
        table[(i + 1, -1)].set_text_props(weight="bold")

    # Apply bold/italic based on rankings
    for i, method in enumerate(method_keys):
        for j, metric_col in enumerate(metric_cols):
            rank = rankings.get((method, metric_col))
            cell = table[(i + 1, j)]
            if rank == 1:
                # Bold for best - gold background
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#FFD700")
            elif rank == 2:
                # Italic for 2nd best - silver background
                cell.set_text_props(style="italic")
                cell.set_facecolor("#C0C0C0")

    ax_table.set_title("Evaluation Metrics Summary (Bold=Best, Italic=2nd Best)",
                       fontsize=14, weight="bold", pad=20)

    # Legend subplot
    ax_legend = fig.add_subplot(2, 1, 2)
    ax_legend.axis("off")

    # Create legend text
    descriptions = [(acr, desc) for acr, desc, group in METRIC_CONFIG.values()]

    # Split into two columns
    n_metrics = len(descriptions)
    col1 = descriptions[: (n_metrics + 1) // 2]
    col2 = descriptions[(n_metrics + 1) // 2 :]

    col1_text = "\n".join([f"{acr}: {desc}" for acr, desc in col1])
    col2_text = "\n".join([f"{acr}: {desc}" for acr, desc in col2])

    ax_legend.text(
        0.05, 0.95, "Metric Legend:", transform=ax_legend.transAxes, fontsize=12,
        verticalalignment="top", weight="bold"
    )
    ax_legend.text(
        0.05, 0.8, col1_text, transform=ax_legend.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace"
    )
    ax_legend.text(
        0.5, 0.8, col2_text, transform=ax_legend.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace"
    )

    # Add ranking note (left) and color legend (right)
    ax_legend.text(
        0.05, 0.25, "Note:\n"
                    "Metrics in the same group (DC*, AW*, CW*) are compared together for ranking.\n"
                    "Higher is better: CNT, ATR, OOR, OAR, SUP, NAV\n"
                    "Lower is better: all others",
        transform=ax_legend.transAxes, fontsize=9, style="italic", verticalalignment="top"
    )

    color_legend = (
        "Header Colors:\n"
        "  \u2588 Gray: Descriptive (no ranking)\n"
        "  \u2588 Green: Semantic (higher is better)\n"
        "  \u2588 Blue: Plausibility\n"
        "  \u2588 Purple: Physics equilibrium"
    )
    ax_legend.text(
        0.55, 0.25, color_legend,
        transform=ax_legend.transAxes, fontsize=9, fontfamily="monospace", verticalalignment="top"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Generated: {OUTPUT_PNG}")
    plt.close()


def main():
    # Load data
    averages = load_averages(INPUT_CSV)

    # Generate markdown table
    md_table = generate_markdown_table(averages)
    with open(OUTPUT_MD, "w") as f:
        f.write(md_table)
    print(f"Generated: {OUTPUT_MD}")
    print("\n" + md_table)

    # Generate matplotlib figure
    generate_matplotlib_figure(averages)


if __name__ == "__main__":
    main()

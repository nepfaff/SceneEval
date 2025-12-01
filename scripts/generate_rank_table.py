#!/usr/bin/env python3
"""
generate_rank_table.py - Generate rank-based scoring table and figure

Computes rank-based scores for each method across all metrics and generates:
1. A markdown table with rankings
2. A matplotlib figure with the table
"""

import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = Path("/home/ubuntu/SceneEval/output_eval/metric_summary.csv")
OUTPUT_MD = Path("/home/ubuntu/SceneEval/output_eval/rank_table.md")
OUTPUT_PNG = Path("/home/ubuntu/SceneEval/output_eval/rank_table.png")

METHODS = ['HSM', 'Holodeck', 'IDesign', 'LayoutVLM', 'SceneWeaver', 'SceneAgent']
METHOD_DISPLAY = {
    'HSM': 'HSM', 'Holodeck': 'Holodeck', 'IDesign': 'IDesign',
    'LayoutVLM': 'LayoutVLM', 'SceneWeaver': 'SceneWeaver', 'SceneAgent': 'Ours'
}

# Metrics configuration: (name, csv_col or list of cols, higher_is_better, category)
METRICS = [
    ('CNT', 'ObjCountMetric.satisfaction_rate', True, 'Semantic'),
    ('ATR', 'ObjAttributeMetric.satisfaction_rate', True, 'Semantic'),
    ('OOR', 'ObjObjRelationshipMetric.satisfaction_rate', True, 'Semantic'),
    ('OAR', 'ObjArchRelationshipMetric.satisfaction_rate', True, 'Semantic'),
    ('SUP', 'SupportMetric.satisfaction_rate', True, 'Semantic'),
    ('ACC', 'AccessibilityMetric.avg_accessibility', True, 'Semantic'),
    ('NAV', 'NavigabilityMetric.navigability', True, 'Plaustic'),
    ('COL', 'CollisionMetric.frac_obj_in_collision', False, 'Plaustic'),
    ('OOB', 'OutOfBoundMetric.frac_out_of_bound', False, 'Plaustic'),
    ('DC', ['DrakeCollisionMetricCoACD.frac_obj_in_collision',
            'DrakeCollisionMetricVHACD.frac_obj_in_collision',
            'DrakeCollisionMetricSceneAgent.frac_obj_in_collision'], False, 'Physics'),
    ('AW', ['ArchitecturalWeldedEquilibriumMetricCoACD.frac_unstable_objects',
            'ArchitecturalWeldedEquilibriumMetricVHACD.frac_unstable_objects',
            'ArchitecturalWeldedEquilibriumMetricSceneAgent.frac_unstable_objects'], False, 'Physics'),
    ('CW', ['CombinedWeldedEquilibriumMetricCoACD.frac_unstable_objects',
            'CombinedWeldedEquilibriumMetricVHACD.frac_unstable_objects',
            'CombinedWeldedEquilibriumMetricSceneAgent.frac_unstable_objects'], False, 'Physics'),
]


def load_averages(csv_path: Path) -> dict:
    """Load average rows from CSV."""
    averages = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('scene') == 'Average':
                averages[row['method']] = row
    return averages


def get_val(row, col):
    """Get numeric value from row."""
    if isinstance(col, list):
        for c in col:
            v = row.get(c, '')
            if v != '' and v is not None:
                try:
                    return float(v)
                except:
                    pass
        return None
    v = row.get(col, '')
    if v == '' or v is None:
        return None
    try:
        return float(v)
    except:
        return None


def compute_rankings(averages: dict) -> dict:
    """Compute rankings for all metrics."""
    results = {
        'metric_rankings': {},  # metric -> [(method, rank, value, points), ...]
        'method_stats': defaultdict(lambda: {
            'total_points': 0, 'count': 0, 'ranks': {},
            'firsts': 0, 'seconds': 0, 'thirds': 0
        })
    }

    for metric_name, col, higher, category in METRICS:
        # Get values for all methods
        vals = {}
        for method in METHODS:
            if method in averages:
                v = get_val(averages[method], col)
                if v is not None:
                    vals[method] = v

        if not vals:
            continue

        # Sort by value
        sorted_methods = sorted(vals.items(), key=lambda x: x[1], reverse=higher)

        # Assign ranks with tie handling
        rankings = []
        prev_val = None
        prev_rank = 0
        for i, (method, val) in enumerate(sorted_methods):
            if val == prev_val:
                rank = prev_rank
            else:
                rank = i + 1
                prev_rank = rank
            prev_val = val

            # Points based on number of methods with data
            n_methods = len(vals)
            points = max(0, n_methods - rank + 1)
            rankings.append((method, rank, val, points))

            # Update method stats
            stats = results['method_stats'][method]
            stats['total_points'] += points
            stats['count'] += 1
            stats['ranks'][metric_name] = (rank, val, points)
            if rank == 1:
                stats['firsts'] += 1
            elif rank == 2:
                stats['seconds'] += 1
            elif rank == 3:
                stats['thirds'] += 1

        # Mark missing methods
        for method in METHODS:
            if method not in vals:
                results['method_stats'][method]['ranks'][metric_name] = (None, None, 0)

        results['metric_rankings'][metric_name] = rankings

    return results


def generate_markdown(results: dict) -> str:
    """Generate markdown table."""
    lines = []

    # Overall ranking table
    lines.append("## Rank-Based Scoring Summary")
    lines.append("")
    lines.append("Points: 1st=N, 2nd=N-1, ... where N = number of methods with data for that metric")
    lines.append("")
    lines.append("| Rank | Method | Points | Avg Rank | 1st | 2nd | 3rd |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")

    # Sort methods by total points
    sorted_methods = sorted(
        results['method_stats'].items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

    for rank, (method, stats) in enumerate(sorted_methods, 1):
        display = METHOD_DISPLAY.get(method, method)
        avg_rank = sum(r for m, (r, v, p) in stats['ranks'].items() if r is not None) / stats['count'] if stats['count'] > 0 else 0

        # Bold for first place
        if rank == 1:
            lines.append(f"| **{rank}** | **{display}** | **{stats['total_points']}** | **{avg_rank:.2f}** | **{stats['firsts']}** | {stats['seconds']} | {stats['thirds']} |")
        else:
            lines.append(f"| {rank} | {display} | {stats['total_points']} | {avg_rank:.2f} | {stats['firsts']} | {stats['seconds']} | {stats['thirds']} |")

    # Per-metric rankings table
    lines.append("")
    lines.append("## Per-Metric Rankings")
    lines.append("")
    lines.append("| Metric | 1st | 2nd | 3rd | 4th | 5th | 6th |")
    lines.append("|:---|:---|:---|:---|:---|:---|:---|")

    for metric_name, col, higher, category in METRICS:
        if metric_name not in results['metric_rankings']:
            continue
        rankings = results['metric_rankings'][metric_name]

        row = [metric_name]
        for method, rank, val, points in rankings:
            display = METHOD_DISPLAY.get(method, method)
            row.append(f"{display} ({val*100:.0f}%)")
        while len(row) < 7:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    # Detailed breakdown
    lines.append("")
    lines.append("## Detailed Rank Breakdown")
    lines.append("")

    for method, stats in sorted_methods:
        display = METHOD_DISPLAY.get(method, method)
        ranks_str = " ".join(f"{m}:{r if r else '-'}" for m, (r, v, p) in stats['ranks'].items())
        lines.append(f"- **{display}**: {ranks_str}")

    return "\n".join(lines)


def generate_figure(results: dict):
    """Generate matplotlib figure."""
    fig = plt.figure(figsize=(14, 10))

    # Sort methods by total points
    sorted_methods = sorted(
        results['method_stats'].items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

    # ===== Top subplot: Overall ranking table =====
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.axis('off')

    # Prepare data for overall table
    col_labels = ['Rank', 'Method', 'Points', 'Avg Rank', '1st', '2nd', '3rd']
    table_data = []

    for rank, (method, stats) in enumerate(sorted_methods, 1):
        display = METHOD_DISPLAY.get(method, method)
        avg_rank = sum(r for m, (r, v, p) in stats['ranks'].items() if r is not None) / stats['count'] if stats['count'] > 0 else 0
        table_data.append([
            str(rank), display, str(stats['total_points']),
            f"{avg_rank:.2f}", str(stats['firsts']), str(stats['seconds']), str(stats['thirds'])
        ])

    table1 = ax1.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table1[(0, j)].set_facecolor('#2E75B6')
        table1[(0, j)].set_text_props(color='white', weight='bold')

    # Highlight first place row
    for j in range(len(col_labels)):
        table1[(1, j)].set_facecolor('#FFD700')
        table1[(1, j)].set_text_props(weight='bold')

    # Silver for second
    for j in range(len(col_labels)):
        table1[(2, j)].set_facecolor('#C0C0C0')

    # Bronze for third
    for j in range(len(col_labels)):
        table1[(3, j)].set_facecolor('#CD7F32')

    ax1.set_title('Overall Method Rankings (Rank-Based Scoring)', fontsize=14, weight='bold', pad=20)

    # ===== Bottom subplot: Per-metric rankings =====
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis('off')

    # Prepare data for metric rankings
    metric_col_labels = ['Metric', '1st', '2nd', '3rd', '4th', '5th', '6th']
    metric_data = []

    for metric_name, col, higher, category in METRICS:
        if metric_name not in results['metric_rankings']:
            continue
        rankings = results['metric_rankings'][metric_name]

        row = [metric_name]
        for method, rank, val, points in rankings:
            display = METHOD_DISPLAY.get(method, method)
            row.append(f"{display}\n({val*100:.0f}%)")
        while len(row) < 7:
            row.append("-")
        metric_data.append(row)

    table2 = ax2.table(
        cellText=metric_data,
        colLabels=metric_col_labels,
        cellLoc='center',
        loc='center',
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.8)

    # Style header
    for j in range(len(metric_col_labels)):
        table2[(0, j)].set_facecolor('#2E75B6')
        table2[(0, j)].set_text_props(color='white', weight='bold')

    # Color 1st column cells by category
    category_colors = {
        'Semantic': '#4CAF50',
        'Plaustic': '#2196F3',
        'Physics': '#9C27B0'
    }

    for i, (metric_name, col, higher, category) in enumerate(METRICS):
        if metric_name in results['metric_rankings']:
            row_idx = [m for m, c, h, cat in METRICS if m in results['metric_rankings']].index(metric_name) + 1
            table2[(row_idx, 0)].set_facecolor(category_colors.get(category, '#FFFFFF'))
            table2[(row_idx, 0)].set_text_props(color='white', weight='bold')

    # Highlight 1st place cells (gold)
    for i, row in enumerate(metric_data, 1):
        table2[(i, 1)].set_facecolor('#FFD700')

    ax2.set_title('Per-Metric Rankings', fontsize=14, weight='bold', pad=20)

    # Add legend for categories
    legend_text = "Category Colors: Green=Semantic | Blue=Plausibility | Purple=Physics"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Generated: {OUTPUT_PNG}")
    plt.close()


def main():
    averages = load_averages(INPUT_CSV)
    results = compute_rankings(averages)

    # Generate markdown
    md_content = generate_markdown(results)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md_content)
    print(f"Generated: {OUTPUT_MD}")
    print("\n" + md_content)

    # Generate figure
    generate_figure(results)


if __name__ == "__main__":
    main()

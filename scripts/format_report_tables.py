#!/usr/bin/env python3
"""
Format evaluation report tables with bold for best and underline for 2nd best.
"""

import re
from pathlib import Path

# Higher is better for these metrics (semantic + NAV)
HIGHER_IS_BETTER = {"CNT", "ATR", "OOR", "OAR", "SUP", "ACC", "NAV"}

# Columns to skip (not ranked)
SKIP_COLUMNS = {"Method", "N", "Scenes", "Avg Objects", "Description",
                "What's Changed/Removed", "Rank", "Status", "Coverage", "Notes", "Abbrev"}


def strip_formatting(text: str) -> str:
    """Remove bold and underline markdown formatting."""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'<u>([^<]+)</u>', r'\1', text)
    return text.strip()


def parse_numeric_value(cell: str) -> float | None:
    """Parse a cell value to float, returning None if not numeric."""
    cell = strip_formatting(cell)
    if cell in ('-', '', 'N/A', '-1', None) or cell.startswith('-'):
        return None
    cell = cell.replace('%', '').strip()
    try:
        return float(cell)
    except ValueError:
        return None


def is_higher_better(col_name: str) -> bool:
    """Check if higher values are better for this column."""
    col_upper = col_name.upper().strip()
    for metric in HIGHER_IS_BETTER:
        if metric == col_upper or col_upper.startswith(metric):
            return True
    return False


def should_skip_column(col_name: str) -> bool:
    """Check if column should be skipped for ranking."""
    col_stripped = col_name.strip()
    return col_stripped in SKIP_COLUMNS or col_stripped == ""


def process_table_lines(table_lines: list[str]) -> list[str]:
    """Process a table and apply formatting."""
    if len(table_lines) < 3:  # Need header, separator, and at least one data row
        return table_lines

    header = table_lines[0]
    separator = table_lines[1]
    data_lines = table_lines[2:]

    # Parse header columns
    header_cells = [c.strip() for c in header.split('|')]
    header_cells = [c for c in header_cells if c]  # Remove empty

    if not header_cells:
        return table_lines

    # Filter out sub-header lines (like "**Baselines**")
    actual_data = []
    sub_header_indices = set()
    for i, line in enumerate(data_lines):
        stripped = strip_formatting(line)
        # Sub-headers typically have few pipes and contain bold text
        if '**' in line and line.count('|') <= 4 and not any(c.strip() and c.strip() not in ['', '**'] for c in line.split('|')[1:-1] if not c.strip().startswith('**')):
            sub_header_indices.add(i)
        else:
            actual_data.append((i, line))

    if not actual_data:
        return table_lines

    # Parse all values for each column
    num_cols = len(header_cells)
    column_data = {col_idx: [] for col_idx in range(num_cols)}  # col_idx -> list of (data_idx, value)

    for data_idx, (orig_idx, line) in enumerate(actual_data):
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c != '' or cells.count('') > 2]  # Handle edge cases
        # Re-split more carefully
        cells = line.split('|')[1:-1]  # Ignore first and last empty
        cells = [c.strip() for c in cells]

        for col_idx, cell in enumerate(cells):
            if col_idx >= num_cols:
                break
            val = parse_numeric_value(cell)
            if val is not None:
                column_data[col_idx].append((data_idx, val, strip_formatting(cell)))

    # Compute rankings for each column
    rankings = {}  # (data_idx, col_idx) -> rank (1=best, 2=second)

    for col_idx in range(num_cols):
        col_name = header_cells[col_idx] if col_idx < len(header_cells) else ""

        if should_skip_column(col_name):
            continue

        values = column_data[col_idx]
        if len(values) < 2:
            continue

        higher_better = is_higher_better(col_name)

        # Sort by value
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=higher_better)

        # Find best value and all rows with it
        best_val = sorted_vals[0][1]
        best_indices = [data_idx for data_idx, val, _ in sorted_vals if val == best_val]

        # Mark best
        for data_idx in best_indices:
            rankings[(data_idx, col_idx)] = 1

        # Only assign 2nd if exactly one best
        if len(best_indices) == 1 and len(sorted_vals) >= 2:
            second_val = sorted_vals[1][1]
            if second_val != best_val:
                second_indices = [data_idx for data_idx, val, _ in sorted_vals if val == second_val]
                for data_idx in second_indices:
                    rankings[(data_idx, col_idx)] = 2

    # Rebuild table with formatting
    result = [header, separator]

    data_idx = 0
    for orig_idx, line in enumerate(data_lines):
        if orig_idx in sub_header_indices:
            result.append(line)
            continue

        # Parse and rebuild this row
        cells = line.split('|')
        if len(cells) < 3:
            result.append(line)
            data_idx += 1
            continue

        new_cells = [cells[0]]  # Leading empty from split
        for col_idx, cell in enumerate(cells[1:-1]):
            cell_stripped = strip_formatting(cell.strip())
            rank = rankings.get((data_idx, col_idx))

            if rank == 1:
                formatted = f" **{cell_stripped}** "
            elif rank == 2:
                formatted = f" <u>{cell_stripped}</u> "
            else:
                formatted = f" {cell_stripped} "

            new_cells.append(formatted)

        new_cells.append(cells[-1])  # Trailing empty from split
        result.append('|'.join(new_cells))
        data_idx += 1

    return result


def find_and_process_tables(content: str) -> str:
    """Find all markdown tables and process them."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        # Check if this could be start of a table
        if '|' in lines[i] and i + 1 < len(lines) and '---' in lines[i + 1]:
            # Collect table lines
            table_lines = [lines[i], lines[i + 1]]
            j = i + 2
            while j < len(lines) and '|' in lines[j] and lines[j].strip():
                table_lines.append(lines[j])
                j += 1

            # Process and add table
            processed = process_table_lines(table_lines)
            result.extend(processed)
            i = j
        else:
            result.append(lines[i])
            i += 1

    return '\n'.join(result)


def main():
    input_path = Path("/home/ubuntu/SceneEval/reports/evaluation_report.md")

    # Read original content
    with open(input_path, 'r') as f:
        content = f.read()

    # Process tables
    processed = find_and_process_tables(content)

    # Write back
    with open(input_path, 'w') as f:
        f.write(processed)

    print(f"Formatted: {input_path}")


if __name__ == "__main__":
    main()

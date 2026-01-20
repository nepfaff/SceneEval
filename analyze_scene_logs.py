#!/usr/bin/env python3
"""
Analyze scene-agent log files to extract API cost and duration statistics.

Usage:
    python analyze_scene_logs.py /path/to/scene/logs
    python analyze_scene_logs.py ~/efs/nicholas/scene-agent-eval-scenes/scene-agent-ours-room-final/
"""

import argparse
import logging
import re
import statistics

from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
console_logger = logging.getLogger(__name__)

# Default pricing (GPT-5.2)
DEFAULT_INPUT_PRICE = 1.75  # $ per 1M tokens
DEFAULT_OUTPUT_PRICE = 14.00  # $ per 1M tokens
DEFAULT_CACHED_PRICE = 0.17  # $ per 1M tokens


@dataclass
class TokenUsage:
    """Token usage for a single agent call."""

    agent_name: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    total_tokens: int
    requests: int


@dataclass
class SceneStats:
    """Statistics for a single scene."""

    scene_name: str
    duration_seconds: float | None = None
    token_usages: list[TokenUsage] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.token_usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.token_usages)

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(u.reasoning_tokens for u in self.token_usages)

    @property
    def total_cached_tokens(self) -> int:
        return sum(u.cached_tokens for u in self.token_usages)

    @property
    def total_requests(self) -> int:
        return sum(u.requests for u in self.token_usages)

    def calculate_cost(
        self,
        input_price: float = DEFAULT_INPUT_PRICE,
        output_price: float = DEFAULT_OUTPUT_PRICE,
        cached_price: float = DEFAULT_CACHED_PRICE,
    ) -> float:
        """Calculate total API cost for this scene."""
        # Non-cached input tokens.
        non_cached_input = self.total_input_tokens - self.total_cached_tokens
        input_cost = non_cached_input * input_price / 1_000_000

        # Cached input tokens.
        cached_cost = self.total_cached_tokens * cached_price / 1_000_000

        # Output tokens (includes reasoning tokens).
        output_cost = (
            (self.total_output_tokens + self.total_reasoning_tokens)
            * output_price
            / 1_000_000
        )

        return input_cost + cached_cost + output_cost


def parse_token_usage(line: str) -> TokenUsage | None:
    """Parse a token usage log line."""
    # Pattern: [AGENT_NAME] Token usage: input=X, output=Y, reasoning=Z, ...
    pattern = (
        r"\[([^\]]+)\] Token usage: input=([\d,]+), output=([\d,]+), "
        r"reasoning=([\d,]+), cached=([\d,]+), total=([\d,]+), requests=(\d+)"
    )
    match = re.search(pattern, line)
    if not match:
        return None

    def parse_int(s: str) -> int:
        return int(s.replace(",", ""))

    return TokenUsage(
        agent_name=match.group(1),
        input_tokens=parse_int(match.group(2)),
        output_tokens=parse_int(match.group(3)),
        reasoning_tokens=parse_int(match.group(4)),
        cached_tokens=parse_int(match.group(5)),
        total_tokens=parse_int(match.group(6)),
        requests=int(match.group(7)),
    )


def parse_duration(line: str) -> float | None:
    """Parse a duration log line, returning seconds."""
    # Pattern: Scene generation completed successfully in HH:MM:SS.ffffff
    pattern = r"Scene generation completed successfully in (\d+):(\d+):(\d+)\.(\d+)"
    match = re.search(pattern, line)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    microseconds = int(match.group(4))

    total_seconds = hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000
    return total_seconds


def parse_log_file(
    log_path: Path, stats: SceneStats, seen_lines: set[str]
) -> None:
    """Parse a single log file and add token usages to stats (deduplicating)."""
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # Try to parse token usage (deduplicate exact lines).
            token_usage = parse_token_usage(line)
            if token_usage:
                if line not in seen_lines:
                    seen_lines.add(line)
                    stats.token_usages.append(token_usage)
                continue

            # Try to parse duration (only in scene.log).
            duration = parse_duration(line)
            if duration:
                stats.duration_seconds = duration


def parse_scene_log(log_path: Path) -> SceneStats:
    """Parse a scene log file and all room.log files in the scene directory."""
    scene_dir = log_path.parent
    scene_name = scene_dir.name
    stats = SceneStats(scene_name=scene_name)
    seen_lines: set[str] = set()

    # Parse the main scene.log
    parse_log_file(log_path, stats, seen_lines)

    # Also parse any room_*/room.log files (for house scenes)
    for room_log in sorted(scene_dir.glob("room_*/room.log")):
        parse_log_file(room_log, stats, seen_lines)

    return stats


def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_logs(log_dir: Path, verbose: bool = False) -> None:
    """Analyze all scene logs in the given directory."""
    # Find all scene.log files.
    log_files = sorted(log_dir.glob("scene_*/scene.log"))

    if not log_files:
        console_logger.warning(f"No scene.log files found in {log_dir}/scene_*/")
        return

    # Parse all logs.
    all_stats: list[SceneStats] = []
    for log_file in log_files:
        stats = parse_scene_log(log_file)
        all_stats.append(stats)

    # Filter scenes with valid data.
    scenes_with_duration = [s for s in all_stats if s.duration_seconds is not None]
    scenes_with_tokens = [s for s in all_stats if s.token_usages]

    # Calculate costs.
    costs = [s.calculate_cost() for s in scenes_with_tokens]
    durations = [s.duration_seconds for s in scenes_with_duration]

    # Aggregate token usage by agent type (per scene totals).
    agent_costs_per_scene: dict[str, list[float]] = {}
    agent_call_counts: dict[str, int] = {}
    for stats in scenes_with_tokens:
        # Group costs by agent within this scene.
        scene_agent_costs: dict[str, float] = {}
        for usage in stats.token_usages:
            non_cached = usage.input_tokens - usage.cached_tokens
            cost = (
                non_cached * DEFAULT_INPUT_PRICE / 1_000_000
                + usage.cached_tokens * DEFAULT_CACHED_PRICE / 1_000_000
                + (usage.output_tokens + usage.reasoning_tokens)
                * DEFAULT_OUTPUT_PRICE
                / 1_000_000
            )
            scene_agent_costs[usage.agent_name] = (
                scene_agent_costs.get(usage.agent_name, 0) + cost
            )
            agent_call_counts[usage.agent_name] = (
                agent_call_counts.get(usage.agent_name, 0) + 1
            )

        # Add this scene's totals to the per-scene list.
        for agent_name, agent_cost in scene_agent_costs.items():
            if agent_name not in agent_costs_per_scene:
                agent_costs_per_scene[agent_name] = []
            agent_costs_per_scene[agent_name].append(agent_cost)

    # Print results.
    console_logger.info("=" * 60)
    console_logger.info("Scene Generation Statistics")
    console_logger.info("=" * 60)
    console_logger.info(f"Directory: {log_dir}")
    console_logger.info(f"Scenes analyzed: {len(all_stats)}")
    console_logger.info(f"Scenes with duration data: {len(scenes_with_duration)}")
    console_logger.info(f"Scenes with token data: {len(scenes_with_tokens)}")
    console_logger.info("")

    # Duration stats.
    if durations:
        console_logger.info("-" * 40)
        console_logger.info("Duration")
        console_logger.info("-" * 40)
        console_logger.info(f"Mean:   {format_duration(statistics.mean(durations))}")
        # Mode: round to nearest minute for meaningful mode calculation.
        durations_rounded = [round(d / 60) * 60 for d in durations]
        try:
            mode_seconds = statistics.mode(durations_rounded)
            console_logger.info(f"Mode:   {format_duration(mode_seconds)}")
        except statistics.StatisticsError:
            pass  # No unique mode.
        if len(durations) > 1:
            console_logger.info(
                f"Std:    {format_duration(statistics.stdev(durations))}"
            )
        console_logger.info(f"Min:    {format_duration(min(durations))}")
        console_logger.info(f"Max:    {format_duration(max(durations))}")
        console_logger.info("")

    # Cost stats.
    if costs:
        console_logger.info("-" * 40)
        console_logger.info("API Cost")
        console_logger.info("-" * 40)
        console_logger.info(f"Mean:   ${statistics.mean(costs):.2f}")
        if len(costs) > 1:
            console_logger.info(f"Std:    ${statistics.stdev(costs):.2f}")
        console_logger.info(f"Min:    ${min(costs):.2f}")
        console_logger.info(f"Max:    ${max(costs):.2f}")
        console_logger.info(f"Total:  ${sum(costs):.2f}")
        console_logger.info("")

    # Token usage stats.
    if scenes_with_tokens:
        total_input = [s.total_input_tokens for s in scenes_with_tokens]
        total_output = [s.total_output_tokens for s in scenes_with_tokens]
        total_reasoning = [s.total_reasoning_tokens for s in scenes_with_tokens]
        total_cached = [s.total_cached_tokens for s in scenes_with_tokens]
        total_requests = [s.total_requests for s in scenes_with_tokens]

        console_logger.info("-" * 40)
        console_logger.info("Token Usage (per scene average)")
        console_logger.info("-" * 40)
        console_logger.info(f"Input:     {statistics.mean(total_input):,.0f}")
        console_logger.info(f"Output:    {statistics.mean(total_output):,.0f}")
        console_logger.info(f"Reasoning: {statistics.mean(total_reasoning):,.0f}")
        console_logger.info(f"Cached:    {statistics.mean(total_cached):,.0f}")
        console_logger.info(f"Requests:  {statistics.mean(total_requests):.1f}")
        console_logger.info("")

    # Agent breakdown.
    if agent_costs_per_scene:
        console_logger.info("-" * 40)
        console_logger.info("Cost by Agent Type (average per scene)")
        console_logger.info("-" * 40)
        total_avg_cost = statistics.mean(costs) if costs else 0

        # Calculate average cost per agent type per scene.
        agent_avg_costs = {
            name: statistics.mean(cost_list) if cost_list else 0
            for name, cost_list in agent_costs_per_scene.items()
        }

        # Sort by average cost descending.
        sorted_agents = sorted(
            agent_avg_costs.items(), key=lambda x: x[1], reverse=True
        )

        for agent_name, avg_cost in sorted_agents:
            pct = (avg_cost / total_avg_cost * 100) if total_avg_cost > 0 else 0
            count = agent_call_counts.get(agent_name, 0)
            scenes_with_agent = len(agent_costs_per_scene[agent_name])
            console_logger.info(
                f"{agent_name}: ${avg_cost:.2f} ({pct:.1f}%) "
                f"[{count} calls in {scenes_with_agent} scenes]"
            )
        console_logger.info("")

    console_logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scene-agent log files for cost and duration statistics."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Directory containing scene subdirectories (scene_000/, scene_001/, etc.)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-scene details"
    )

    args = parser.parse_args()

    # Expand user path.
    log_dir = args.path.expanduser()

    if not log_dir.exists():
        console_logger.error(f"Directory not found: {log_dir}")
        return 1

    analyze_logs(log_dir, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    main()

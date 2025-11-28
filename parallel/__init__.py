"""
Parallel evaluation module for SceneEval.

Provides ProcessPoolExecutor-based parallel scene evaluation.
"""

from .executor import ParallelSceneEvaluator
from .worker import evaluate_scene_worker

__all__ = ["ParallelSceneEvaluator", "evaluate_scene_worker"]

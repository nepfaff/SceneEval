"""
Parallel evaluation utilities for SceneEval.

For parallel execution, use scripts/run_parallel.sh which runs multiple
independent Python processes (avoiding bpy/Drake multiprocessing issues).
"""

from .worker import FileLoggingContext

__all__ = ["FileLoggingContext"]

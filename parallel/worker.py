"""
Logging utilities for scene evaluation.

This module provides FileLoggingContext for per-scene log files.
"""

import logging
import pathlib


class FileLoggingContext:
    """Context manager to redirect all loggers to scene-specific log files.

    This class captures ALL logging that occurs within its context.
    Based on scene-agent's FileLoggingContext.
    """

    def __init__(self, log_file_path: pathlib.Path, suppress_stdout: bool = False):
        """
        Args:
            log_file_path: Path to the scene-specific log file
            suppress_stdout: If True, prevents logs from also going to stdout
        """
        self.log_file_path = log_file_path
        self.suppress_stdout = suppress_stdout
        self.file_handler = None
        self.original_handlers = []

    def __enter__(self):
        """Set up file handler and redirect all loggers."""
        # Create file handler with consistent formatting.
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Get root logger to capture everything.
        root_logger = logging.getLogger()

        # Add our file handler.
        root_logger.addHandler(self.file_handler)

        if self.suppress_stdout:
            # Save original handlers and remove console handlers temporarily.
            self.original_handlers = root_logger.handlers[:]
            for handler in self.original_handlers:
                if handler != self.file_handler:
                    root_logger.removeHandler(handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up handlers and restore original state."""
        root_logger = logging.getLogger()

        # Remove our file handler.
        if self.file_handler in root_logger.handlers:
            root_logger.removeHandler(self.file_handler)

        if self.suppress_stdout and self.original_handlers:
            # Restore original handlers.
            for handler in self.original_handlers:
                if handler not in root_logger.handlers:
                    root_logger.addHandler(handler)

        # Close the file handler.
        if self.file_handler:
            self.file_handler.close()

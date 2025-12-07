"""
Utility modules for PyTorch Scientist.

Provides:
- Logging configuration
- Persistence helpers (JSON/YAML serialization)
- Timer utilities
- Hardware detection
"""

from pytorch_scientist.utils.logging import setup_logging, get_logger
from pytorch_scientist.utils.persistence import (
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_artifact,
    ArtifactManager,
)
from pytorch_scientist.utils.timing import Timer, benchmark_function

__all__ = [
    "setup_logging",
    "get_logger",
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_artifact",
    "ArtifactManager",
    "Timer",
    "benchmark_function",
]

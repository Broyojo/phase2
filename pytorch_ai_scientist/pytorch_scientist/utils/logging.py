"""
Logging configuration for PyTorch Scientist.

Provides structured logging with rich formatting and file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from pytorch_scientist.config import ResearchConfig

# Global logger registry
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    use_rich: bool = True,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_rich: Whether to use rich formatting (if available)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create handlers
    handlers: list[logging.Handler] = []

    # Console handler
    if use_rich and RICH_AVAILABLE:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handlers.append(handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger("pytorch_scientist")
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Add new handlers
    for h in handlers:
        h.setLevel(log_level)
        root_logger.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (will be prefixed with 'pytorch_scientist.')

    Returns:
        Configured logger instance
    """
    full_name = f"pytorch_scientist.{name}" if not name.startswith("pytorch_scientist") else name

    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)

    return _loggers[full_name]


def setup_from_config(config: "ResearchConfig") -> None:
    """
    Set up logging from a ResearchConfig.

    Args:
        config: Research configuration object
    """
    log_file = config.run_dir / "run.log" if config.save_intermediate else None
    setup_logging(
        level=config.log_level,
        log_file=log_file,
        use_rich=True,
    )

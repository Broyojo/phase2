"""
Persistence helpers for PyTorch Scientist.

Provides JSON/YAML serialization and artifact management.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from pytorch_scientist.utils.logging import get_logger

logger = get_logger("persistence")

T = TypeVar("T")


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses, enums, and paths."""

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON-serializable or a dataclass)
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, cls=EnhancedJSONEncoder, indent=indent)

    logger.debug(f"Saved JSON to {path}")


def load_json(path: Path | str) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {path}")
    return data


def save_yaml(data: Any, path: Path | str) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Data to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)

    # Handle enums and paths
    def convert_special_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_special_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_special_types(v) for v in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    data = convert_special_types(data)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Saved YAML to {path}")


def load_yaml(path: Path | str) -> Any:
    """
    Load data from a YAML file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    path = Path(path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    logger.debug(f"Loaded YAML from {path}")
    return data


def save_artifact(
    data: Any,
    name: str,
    run_dir: Path,
    format: str = "json",
) -> Path:
    """
    Save an artifact to the run directory.

    Args:
        data: Data to save
        name: Artifact name (without extension)
        run_dir: Run directory path
        format: Output format ('json' or 'yaml')

    Returns:
        Path to saved artifact
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if format == "json":
        path = run_dir / f"{name}.json"
        save_json(data, path)
    elif format == "yaml":
        path = run_dir / f"{name}.yaml"
        save_yaml(data, path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return path


class ArtifactManager:
    """
    Manages artifacts for a research run.

    Provides structured saving and loading of:
    - Literature summaries
    - Generated ideas
    - Search histories
    - Experiment results
    - Final summaries
    """

    def __init__(self, run_dir: Path | str):
        """
        Initialize artifact manager.

        Args:
            run_dir: Directory for this run
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.literature_dir = self.run_dir / "literature"
        self.ideas_dir = self.run_dir / "ideas"
        self.search_dir = self.run_dir / "search"
        self.results_dir = self.run_dir / "results"

        for d in [self.literature_dir, self.ideas_dir, self.search_dir, self.results_dir]:
            d.mkdir(exist_ok=True)

    def save_literature_summary(self, summary: Any) -> Path:
        """Save literature summary."""
        return save_artifact(summary, "literature_summary", self.literature_dir)

    def save_ideas(self, ideas: list[Any]) -> Path:
        """Save generated ideas."""
        return save_artifact({"ideas": ideas}, "ideas", self.ideas_dir)

    def save_selected_idea(self, idea: Any) -> Path:
        """Save the selected idea."""
        return save_artifact(idea, "selected_idea", self.ideas_dir)

    def save_search_history(self, history: list[Any]) -> Path:
        """Save search history."""
        return save_artifact({"history": history}, "search_history", self.search_dir)

    def save_best_result(self, result: Any) -> Path:
        """Save best result."""
        return save_artifact(result, "best_result", self.results_dir)

    def save_summary(self, summary: str) -> Path:
        """Save text summary as markdown."""
        path = self.results_dir / "summary.md"
        with open(path, "w") as f:
            f.write(summary)
        logger.debug(f"Saved summary to {path}")
        return path

    def save_run_metadata(self, metadata: dict[str, Any]) -> Path:
        """Save run metadata."""
        return save_artifact(metadata, "metadata", self.run_dir)

    def load_literature_summary(self) -> Any | None:
        """Load literature summary if exists."""
        path = self.literature_dir / "literature_summary.json"
        if path.exists():
            return load_json(path)
        return None

    def load_ideas(self) -> list[Any] | None:
        """Load ideas if exist."""
        path = self.ideas_dir / "ideas.json"
        if path.exists():
            data = load_json(path)
            return data.get("ideas", [])
        return None

    def load_search_history(self) -> list[Any] | None:
        """Load search history if exists."""
        path = self.search_dir / "search_history.json"
        if path.exists():
            data = load_json(path)
            return data.get("history", [])
        return None

    def load_best_result(self) -> Any | None:
        """Load best result if exists."""
        path = self.results_dir / "best_result.json"
        if path.exists():
            return load_json(path)
        return None

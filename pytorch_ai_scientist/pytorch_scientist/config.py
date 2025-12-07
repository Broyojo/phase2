"""
Global configuration dataclasses for PyTorch Scientist.

Contains all configuration structures for:
- LLM providers (Grok, Anthropic, OpenAI)
- Exa API settings
- Search parameters
- Experiment settings
- Pipeline configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROK = "grok"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class SearchStrategy(Enum):
    """Config search strategies."""
    EVOLUTIONARY = "evolutionary"
    MCTS = "mcts"
    RANDOM = "random"
    GRID = "grid"


class TargetOperation(Enum):
    """Target operations for optimization."""
    SOFTMAX = "softmax"
    ATTENTION = "attention"
    GEMM = "gemm"
    LAYERNORM = "layernorm"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: LLMProvider = LLMProvider.GROK
    model: str = "grok-3-mini"

    # API endpoints
    grok_base_url: str = "https://api.x.ai/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com/v1"

    # API keys (loaded from environment)
    grok_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95

    def __post_init__(self):
        """Load API keys from environment variables."""
        self.grok_api_key = self.grok_api_key or os.getenv("XAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")

    @property
    def active_api_key(self) -> str | None:
        """Get the API key for the current provider."""
        if self.provider == LLMProvider.GROK:
            return self.grok_api_key
        elif self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_api_key
        return None

    @property
    def active_base_url(self) -> str:
        """Get the base URL for the current provider."""
        if self.provider == LLMProvider.GROK:
            return self.grok_base_url
        elif self.provider == LLMProvider.ANTHROPIC:
            return self.anthropic_base_url
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_base_url
        return self.openai_base_url


@dataclass
class ExaConfig:
    """Configuration for Exa.ai API."""

    api_key: str | None = field(default=None, repr=False)
    num_results: int = 10
    use_autoprompt: bool = True
    include_domains: list[str] = field(default_factory=lambda: [
        "arxiv.org",
        "aclanthology.org",
        "openreview.net",
        "proceedings.neurips.cc",
        "proceedings.mlr.press",
    ])

    # Search categories
    default_category: str = "research paper"

    def __post_init__(self):
        """Load API key from environment."""
        self.api_key = self.api_key or os.getenv("EXA_API_KEY")


@dataclass
class XSearchConfig:
    """Configuration for X (Twitter) search via xdk."""

    api_key: Optional[str] = field(default=None, repr=False)
    enabled: bool = True

    # Author sources
    authors: list[str] = field(default_factory=lambda: [
        "iScienceLuvr",
        "arankomatsuzaki",
        "rohanpaul_ai",
        "omarsar0",
        "_akhaliq",
    ])
    authors_file: Optional[Path] = None

    # Query behavior
    query: str = ""
    include_retweets: bool = True
    max_results: int = 50
    use_full_archive: bool = False
    start_date: Optional[str] = None  # YYYY-MM-DD

    def __post_init__(self):
        """Load API key from environment if not provided."""
        self.api_key = self.api_key or os.getenv("X_API_KEY")

    def resolved_authors(self) -> list[str]:
        """Return combined authors list, including those from a file if provided."""
        authors = list(self.authors)
        if self.authors_file and Path(self.authors_file).exists():
            with open(self.authors_file, "r") as f:
                for line in f:
                    handle = line.strip().lstrip("@")
                    if handle:
                        authors.append(handle)
        # Deduplicate while preserving order
        seen = set()
        unique: list[str] = []
        for a in authors:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique


@dataclass
class SearchConfig:
    """Configuration for config search."""

    strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    max_evaluations: int = 50
    population_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_count: int = 2

    # Early stopping
    patience: int = 10
    min_improvement: float = 0.01

    # MCTS-specific
    mcts_exploration_weight: float = 1.414
    mcts_max_depth: int = 10
    mcts_simulations: int = 100


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    target_operation: TargetOperation = TargetOperation.SOFTMAX

    # Hardware settings
    device: str = "cuda"  # cuda, cpu, or specific like cuda:0
    backend: Literal["cuda", "rocm", "cpu"] = "cuda"

    # Benchmark settings
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    use_cuda_graphs: bool = True

    # Problem sizes (for transformer-style workloads)
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32])
    sequence_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    hidden_dims: list[int] = field(default_factory=lambda: [768, 1024, 2048])
    num_heads: list[int] = field(default_factory=lambda: [8, 12, 16])

    # Default problem size for quick experiments
    default_batch_size: int = 8
    default_seq_len: int = 1024
    default_hidden_dim: int = 1024
    default_num_heads: int = 8


@dataclass
class IdeationConfig:
    """Configuration for idea generation."""

    max_ideas: int = 5
    novelty_threshold: float = 0.5
    feasibility_threshold: float = 0.6

    # Idea generation parameters
    num_generation_rounds: int = 2
    ideas_per_round: int = 3

    # Novelty check
    novelty_search_limit: int = 5

    # Constraints for generated ideas
    constraints: list[str] = field(default_factory=lambda: [
        "Must be implementable within 24 hours",
        "Must target PyTorch >= 2.4 or Helion",
        "Must be benchmarkable with clear metrics",
        "Must focus on transformer-style workloads",
        "Must provide speedup over torch.compile baseline",
    ])


@dataclass
class ResearchConfig:
    """Top-level research configuration."""

    # Domain specification
    domain: str = "LLM guided PyTorch kernel optimization"

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    exa: ExaConfig = field(default_factory=ExaConfig)
    x: XSearchConfig = field(default_factory=XSearchConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    ideation: IdeationConfig = field(default_factory=IdeationConfig)

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    run_name: str | None = None
    save_intermediate: bool = True

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    def __post_init__(self):
        """Ensure output directory exists and generate run name if needed."""
        import datetime

        if self.run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """Get the directory for this run."""
        return self.output_dir / self.run_name

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(),
            exa=ExaConfig(),
            x=XSearchConfig(),
        )


# Default configuration singleton
DEFAULT_CONFIG = None


def get_default_config() -> ResearchConfig:
    """Get or create the default configuration."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = ResearchConfig.from_env()
    return DEFAULT_CONFIG


def set_default_config(config: ResearchConfig) -> None:
    """Set the default configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

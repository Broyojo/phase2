"""
Configuration search engine using evolutionary/MCTS algorithms.

Integrates with OpenEvolve patterns for:
- Evolutionary search over config spaces
- MCTS-based exploration
- Multi-fidelity optimization
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pytorch_scientist.config import SearchConfig, SearchStrategy
from pytorch_scientist.experiments import ExperimentResult, ExperimentRunner
from pytorch_scientist.ideation import Idea
from pytorch_scientist.search_space import (
    ConfigSpace,
    ConfigType,
    SoftmaxConfig,
    SoftmaxConfigSpace,
    TorchCompileConfig,
    TorchCompileConfigSpace,
)
from pytorch_scientist.utils.logging import get_logger

logger = get_logger("search_engine")

T = TypeVar("T")


@dataclass
class SearchEntry:
    """A single entry in the search history."""

    config: dict[str, Any]
    latency_ms: float
    speedup: float
    generation: int
    status: str
    parent_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "latency_ms": self.latency_ms,
            "speedup": self.speedup,
            "generation": self.generation,
            "status": self.status,
            "parent_id": self.parent_id,
        }


@dataclass
class SearchResult:
    """Result of a search run."""

    best_config: dict[str, Any]
    best_latency_ms: float
    best_speedup: float
    baseline_latency_ms: float
    history: list[SearchEntry]
    total_evaluations: int
    generations: int
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_config": self.best_config,
            "best_latency_ms": self.best_latency_ms,
            "best_speedup": self.best_speedup,
            "baseline_latency_ms": self.baseline_latency_ms,
            "history": [h.to_dict() for h in self.history],
            "total_evaluations": self.total_evaluations,
            "generations": self.generations,
            "status": self.status,
            "metadata": self.metadata,
        }

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        successful = [h for h in self.history if h.status == "success"]
        if not successful:
            return {"num_successful": 0}

        speedups = [h.speedup for h in successful]
        return {
            "num_successful": len(successful),
            "best_speedup": max(speedups),
            "worst_speedup": min(speedups),
            "avg_speedup": sum(speedups) / len(speedups),
            "total_evaluations": self.total_evaluations,
        }


class SearchAlgorithm(ABC, Generic[T]):
    """Abstract base class for search algorithms."""

    @abstractmethod
    def initialize(self, config_space: ConfigSpace[T]) -> list[T]:
        """Initialize the search with a population."""
        pass

    @abstractmethod
    def step(
        self,
        population: list[T],
        fitness: list[float],
        config_space: ConfigSpace[T],
    ) -> list[T]:
        """Perform one step of the search algorithm."""
        pass


class EvolutionarySearch(SearchAlgorithm[T]):
    """
    Evolutionary search algorithm.

    Implements a basic genetic algorithm with:
    - Tournament selection
    - Crossover and mutation
    - Elitism
    """

    def __init__(self, search_config: SearchConfig):
        """
        Initialize evolutionary search.

        Args:
            search_config: Search configuration
        """
        self.config = search_config

    def initialize(self, config_space: ConfigSpace[T]) -> list[T]:
        """Initialize population with random configs."""
        return [
            config_space.sample_random()
            for _ in range(self.config.population_size)
        ]

    def step(
        self,
        population: list[T],
        fitness: list[float],
        config_space: ConfigSpace[T],
    ) -> list[T]:
        """Perform one evolutionary step."""
        # Sort by fitness (higher is better)
        sorted_indices = sorted(
            range(len(fitness)),
            key=lambda i: fitness[i],
            reverse=True,
        )

        new_population: list[T] = []

        # Elitism - keep best individuals
        for i in range(self.config.elite_count):
            if i < len(sorted_indices):
                new_population.append(population[sorted_indices[i]])

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population, fitness)
            parent2 = self._tournament_select(population, fitness)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child = config_space.crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])

            # Mutation
            if random.random() < self.config.mutation_rate:
                child = config_space.mutate(child, self.config.mutation_rate)

            # Validate
            if config_space.validate(child):
                new_population.append(child)

        return new_population

    def _tournament_select(
        self,
        population: list[T],
        fitness: list[float],
        tournament_size: int = 3,
    ) -> T:
        """Select individual via tournament selection."""
        candidates = random.sample(
            list(zip(population, fitness)),
            min(tournament_size, len(population)),
        )
        return max(candidates, key=lambda x: x[1])[0]


class RandomSearch(SearchAlgorithm[T]):
    """Simple random search for baseline comparison."""

    def __init__(self, search_config: SearchConfig):
        """Initialize random search."""
        self.config = search_config

    def initialize(self, config_space: ConfigSpace[T]) -> list[T]:
        """Initialize with random configs."""
        return [
            config_space.sample_random()
            for _ in range(self.config.population_size)
        ]

    def step(
        self,
        population: list[T],
        fitness: list[float],
        config_space: ConfigSpace[T],
    ) -> list[T]:
        """Generate new random configs, keeping best."""
        # Keep best
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        best = population[best_idx]

        # Generate new random configs
        new_pop = [best]
        for _ in range(self.config.population_size - 1):
            new_pop.append(config_space.sample_random())

        return new_pop


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(
        self,
        config: Any,
        parent: "MCTSNode | None" = None,
    ):
        self.config = config
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """UCB1 score for selection."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value

        import math

        exploitation = self.value
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class MCTSSearch(SearchAlgorithm[T]):
    """
    Monte Carlo Tree Search for config optimization.

    Uses UCB1 for selection and random rollouts.
    """

    def __init__(self, search_config: SearchConfig):
        """Initialize MCTS."""
        self.config = search_config
        self.root: MCTSNode | None = None

    def initialize(self, config_space: ConfigSpace[T]) -> list[T]:
        """Initialize MCTS with root node and initial configs."""
        # Create root with a random config
        root_config = config_space.sample_random()
        self.root = MCTSNode(root_config)

        # Return initial population for evaluation
        return [
            config_space.sample_random()
            for _ in range(self.config.population_size)
        ]

    def step(
        self,
        population: list[T],
        fitness: list[float],
        config_space: ConfigSpace[T],
    ) -> list[T]:
        """Perform MCTS step."""
        if self.root is None:
            return self.initialize(config_space)

        # Update tree with fitness results
        for config, fit in zip(population, fitness):
            self._update_tree(config, fit)

        # Select and expand
        new_configs: list[T] = []
        for _ in range(self.config.population_size):
            # Select leaf node
            node = self._select(self.root)

            # Expand
            child_config = config_space.mutate(
                node.config,
                self.config.mutation_rate,
            )

            if config_space.validate(child_config):
                child_node = MCTSNode(child_config, parent=node)
                node.children.append(child_node)
                new_configs.append(child_config)
            else:
                new_configs.append(config_space.sample_random())

        return new_configs

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        while node.children:
            node = max(
                node.children,
                key=lambda n: n.ucb1(self.config.mcts_exploration_weight),
            )
        return node

    def _update_tree(self, config: Any, fitness: float) -> None:
        """Backpropagate fitness to tree."""
        # Find matching node and update
        # For simplicity, just update root for now
        if self.root:
            self.root.visits += 1
            self.root.total_reward += fitness


class ConfigSearchEngine:
    """
    Main search engine that orchestrates config optimization.
    """

    def __init__(
        self,
        search_config: SearchConfig,
        experiment_runner: ExperimentRunner,
    ):
        """
        Initialize search engine.

        Args:
            search_config: Search configuration
            experiment_runner: Experiment runner for evaluation
        """
        self.search_config = search_config
        self.experiment_runner = experiment_runner

        # Select algorithm
        self.algorithm = self._create_algorithm()

        # Search state
        self.history: list[SearchEntry] = []
        self.best_config: ConfigType | None = None
        self.best_fitness: float = float("-inf")
        self.generation = 0

        logger.info(f"Initialized ConfigSearchEngine with {search_config.strategy}")

    def _create_algorithm(self) -> SearchAlgorithm[Any]:
        """Create the search algorithm."""
        if self.search_config.strategy == SearchStrategy.EVOLUTIONARY:
            return EvolutionarySearch(self.search_config)
        elif self.search_config.strategy == SearchStrategy.MCTS:
            return MCTSSearch(self.search_config)
        elif self.search_config.strategy == SearchStrategy.RANDOM:
            return RandomSearch(self.search_config)
        else:
            logger.warning(f"Unknown strategy {self.search_config.strategy}, using evolutionary")
            return EvolutionarySearch(self.search_config)

    def run_search(
        self,
        idea: Idea,
        config_space: ConfigSpace[Any],
        max_evaluations: int | None = None,
        early_stop: bool = True,
    ) -> SearchResult:
        """
        Run the configuration search.

        Args:
            idea: The idea being tested
            config_space: Configuration space to search
            max_evaluations: Maximum evaluations (uses config default if None)
            early_stop: Whether to use early stopping

        Returns:
            SearchResult with best config and history
        """
        max_evals = max_evaluations or self.search_config.max_evaluations

        logger.info(
            f"Starting search for '{idea.title}' "
            f"(max_evals={max_evals}, strategy={self.search_config.strategy})"
        )

        # Reset state
        self.history = []
        self.best_config = None
        self.best_fitness = float("-inf")
        self.generation = 0
        no_improvement_count = 0

        # Initialize population
        population = self.algorithm.initialize(config_space)

        total_evals = 0

        while total_evals < max_evals:
            self.generation += 1

            # Evaluate population
            fitness_scores: list[float] = []
            for config in population:
                if total_evals >= max_evals:
                    break

                result = self.experiment_runner.evaluate_config(config)
                total_evals += 1

                # Use speedup as fitness
                fitness = result.speedup_vs_baseline
                fitness_scores.append(fitness)

                # Record history
                entry = SearchEntry(
                    config=result.config,
                    latency_ms=result.latency_ms,
                    speedup=result.speedup_vs_baseline,
                    generation=self.generation,
                    status=result.status,
                )
                self.history.append(entry)

                # Track best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_config = config
                    no_improvement_count = 0
                    logger.info(
                        f"New best: speedup={fitness:.4f}, "
                        f"latency={result.latency_ms:.3f}ms, "
                        f"gen={self.generation}"
                    )
                else:
                    no_improvement_count += 1

            # Early stopping
            if early_stop and no_improvement_count >= self.search_config.patience:
                logger.info(
                    f"Early stopping after {no_improvement_count} evaluations "
                    "without improvement"
                )
                break

            # Check if we have enough evaluations
            if total_evals >= max_evals:
                break

            # Evolve population
            if len(fitness_scores) == len(population):
                population = self.algorithm.step(population, fitness_scores, config_space)
            else:
                # Partial evaluation - just mutate existing
                population = [
                    config_space.mutate(config) for config in population
                ]

        # Get baseline for reporting
        baseline_metrics = self.experiment_runner.get_baseline_metrics()
        baseline_latency = baseline_metrics.get("compiled_latency_ms") or baseline_metrics.get(
            "eager_latency_ms"
        ) or 1.0

        # Best config dict
        best_config_dict = (
            self.best_config.to_dict()
            if hasattr(self.best_config, "to_dict")
            else {"config": str(self.best_config)}
        ) if self.best_config else {}

        # Find best entry for latency
        best_entry = max(self.history, key=lambda h: h.speedup) if self.history else None

        result = SearchResult(
            best_config=best_config_dict,
            best_latency_ms=best_entry.latency_ms if best_entry else baseline_latency,
            best_speedup=self.best_fitness if self.best_fitness > float("-inf") else 1.0,
            baseline_latency_ms=baseline_latency,
            history=self.history,
            total_evaluations=total_evals,
            generations=self.generation,
            metadata={
                "idea_title": idea.title,
                "strategy": self.search_config.strategy.value,
                "operation": baseline_metrics.get("operation"),
            },
        )

        logger.info(
            f"Search complete: {total_evals} evaluations, "
            f"best_speedup={result.best_speedup:.4f}x, "
            f"best_latency={result.best_latency_ms:.3f}ms"
        )

        return result


def run_search(
    idea: Idea,
    experiment_runner: ExperimentRunner,
    search_config: SearchConfig | None = None,
    max_evaluations: int = 50,
) -> SearchResult:
    """
    Convenience function to run a config search.

    Args:
        idea: The idea to test
        experiment_runner: Experiment runner
        search_config: Optional search configuration
        max_evaluations: Maximum evaluations

    Returns:
        SearchResult
    """
    if search_config is None:
        search_config = SearchConfig(max_evaluations=max_evaluations)

    # Determine config space from idea
    config_space: ConfigSpace[Any]
    if "torch.compile" in idea.config_space.lower():
        config_space = TorchCompileConfigSpace()
    elif "softmax" in idea.config_space.lower():
        config_space = SoftmaxConfigSpace()
    else:
        # Default to TorchCompile space
        config_space = TorchCompileConfigSpace()

    engine = ConfigSearchEngine(search_config, experiment_runner)
    return engine.run_search(idea, config_space, max_evaluations)

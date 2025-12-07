"""Tests for search engine module."""

from unittest.mock import MagicMock, patch

import pytest

from pytorch_scientist.config import SearchConfig, SearchStrategy
from pytorch_scientist.ideation import Idea
from pytorch_scientist.search_engine import (
    ConfigSearchEngine,
    EvolutionarySearch,
    MCTSNode,
    MCTSSearch,
    RandomSearch,
    SearchEntry,
    SearchResult,
)
from pytorch_scientist.search_space import (
    SoftmaxConfig,
    SoftmaxConfigSpace,
    TorchCompileConfigSpace,
)


class TestSearchEntry:
    """Tests for SearchEntry."""

    def test_to_dict(self):
        """Test serialization."""
        entry = SearchEntry(
            config={"block_size": 256},
            latency_ms=1.5,
            speedup=1.2,
            generation=1,
            status="success",
        )
        d = entry.to_dict()
        assert d["latency_ms"] == 1.5
        assert d["speedup"] == 1.2


class TestSearchResult:
    """Tests for SearchResult."""

    def test_to_dict(self):
        """Test serialization."""
        result = SearchResult(
            best_config={"block_size": 256},
            best_latency_ms=1.0,
            best_speedup=1.5,
            baseline_latency_ms=1.5,
            history=[],
            total_evaluations=10,
            generations=5,
        )
        d = result.to_dict()
        assert d["best_speedup"] == 1.5

    def test_get_summary_stats_empty(self):
        """Test summary stats with empty history."""
        result = SearchResult(
            best_config={},
            best_latency_ms=1.0,
            best_speedup=1.0,
            baseline_latency_ms=1.0,
            history=[],
            total_evaluations=0,
            generations=0,
        )
        stats = result.get_summary_stats()
        assert stats["num_successful"] == 0

    def test_get_summary_stats(self):
        """Test summary stats with history."""
        history = [
            SearchEntry(config={}, latency_ms=1.0, speedup=1.2, generation=1, status="success"),
            SearchEntry(config={}, latency_ms=1.1, speedup=1.1, generation=1, status="success"),
            SearchEntry(config={}, latency_ms=0.9, speedup=1.3, generation=2, status="success"),
        ]
        result = SearchResult(
            best_config={},
            best_latency_ms=0.9,
            best_speedup=1.3,
            baseline_latency_ms=1.2,
            history=history,
            total_evaluations=3,
            generations=2,
        )
        stats = result.get_summary_stats()
        assert stats["num_successful"] == 3
        assert stats["best_speedup"] == 1.3
        assert stats["worst_speedup"] == 1.1


class TestEvolutionarySearch:
    """Tests for EvolutionarySearch."""

    @pytest.fixture
    def search(self):
        """Create evolutionary search instance."""
        config = SearchConfig(population_size=5, elite_count=1)
        return EvolutionarySearch(config)

    @pytest.fixture
    def config_space(self):
        """Create config space."""
        return SoftmaxConfigSpace()

    def test_initialize(self, search, config_space):
        """Test population initialization."""
        population = search.initialize(config_space)
        assert len(population) == 5
        assert all(isinstance(c, SoftmaxConfig) for c in population)

    def test_step(self, search, config_space):
        """Test evolutionary step."""
        population = search.initialize(config_space)
        fitness = [1.0, 1.2, 0.8, 1.1, 0.9]
        new_pop = search.step(population, fitness, config_space)
        assert len(new_pop) == 5
        # Best individual should be preserved (elitism)


class TestRandomSearch:
    """Tests for RandomSearch."""

    @pytest.fixture
    def search(self):
        """Create random search instance."""
        config = SearchConfig(population_size=5)
        return RandomSearch(config)

    @pytest.fixture
    def config_space(self):
        """Create config space."""
        return SoftmaxConfigSpace()

    def test_initialize(self, search, config_space):
        """Test population initialization."""
        population = search.initialize(config_space)
        assert len(population) == 5

    def test_step_keeps_best(self, search, config_space):
        """Test step keeps best individual."""
        population = search.initialize(config_space)
        fitness = [0.5, 2.0, 0.3, 0.1, 0.4]  # Second is best
        new_pop = search.step(population, fitness, config_space)
        assert len(new_pop) == 5
        # Best config should be in new population
        assert population[1] in new_pop


class TestMCTSNode:
    """Tests for MCTSNode."""

    def test_value_no_visits(self):
        """Test value with no visits."""
        node = MCTSNode(config={})
        assert node.value == 0.0

    def test_value_with_visits(self):
        """Test value with visits."""
        node = MCTSNode(config={})
        node.visits = 10
        node.total_reward = 15.0
        assert node.value == 1.5

    def test_ucb1_no_visits(self):
        """Test UCB1 with no visits returns infinity."""
        node = MCTSNode(config={})
        assert node.ucb1() == float("inf")


class TestMCTSSearch:
    """Tests for MCTSSearch."""

    @pytest.fixture
    def search(self):
        """Create MCTS search instance."""
        config = SearchConfig(
            population_size=5,
            mcts_exploration_weight=1.414,
        )
        return MCTSSearch(config)

    @pytest.fixture
    def config_space(self):
        """Create config space."""
        return SoftmaxConfigSpace()

    def test_initialize(self, search, config_space):
        """Test MCTS initialization."""
        population = search.initialize(config_space)
        assert len(population) == 5
        assert search.root is not None


class TestConfigSearchEngine:
    """Tests for ConfigSearchEngine."""

    @pytest.fixture
    def mock_runner(self):
        """Create mock experiment runner."""
        runner = MagicMock()
        runner.evaluate_config.return_value = MagicMock(
            config={"test": True},
            latency_ms=1.0,
            speedup_vs_baseline=1.2,
            status="success",
        )
        runner.get_baseline_metrics.return_value = {
            "compiled_latency_ms": 1.2,
            "operation": "softmax",
        }
        return runner

    @pytest.fixture
    def idea(self):
        """Create test idea."""
        return Idea(
            title="Test Idea",
            description="Test description",
            novelty="Novel approach",
            implementation_sketch="Implement it",
            expected_outcome="Better performance",
            risk_level="medium",
            config_space="torch.compile options",
            feasibility_score=0.8,
            novelty_score=0.7,
        )

    def test_create_evolutionary_algorithm(self, mock_runner):
        """Test evolutionary algorithm creation."""
        config = SearchConfig(strategy=SearchStrategy.EVOLUTIONARY)
        engine = ConfigSearchEngine(config, mock_runner)
        assert isinstance(engine.algorithm, EvolutionarySearch)

    def test_create_mcts_algorithm(self, mock_runner):
        """Test MCTS algorithm creation."""
        config = SearchConfig(strategy=SearchStrategy.MCTS)
        engine = ConfigSearchEngine(config, mock_runner)
        assert isinstance(engine.algorithm, MCTSSearch)

    def test_run_search(self, mock_runner, idea):
        """Test search execution."""
        config = SearchConfig(
            strategy=SearchStrategy.RANDOM,
            max_evaluations=10,
            population_size=3,
        )
        engine = ConfigSearchEngine(config, mock_runner)
        config_space = TorchCompileConfigSpace()

        result = engine.run_search(idea, config_space, max_evaluations=5)

        assert isinstance(result, SearchResult)
        assert result.total_evaluations <= 5
        assert len(result.history) > 0

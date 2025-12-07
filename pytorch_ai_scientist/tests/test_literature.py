"""Tests for literature discovery module."""

import pytest

from pytorch_scientist.config import ExaConfig, LLMConfig
from pytorch_scientist.literature import (
    ExaSearchResult,
    LiteratureDiscovery,
    LiteratureSummary,
    MockExaClient,
)


class TestExaSearchResult:
    """Tests for ExaSearchResult."""

    def test_to_summary_string(self):
        """Test conversion to summary string."""
        result = ExaSearchResult(
            title="Test Paper",
            url="https://arxiv.org/test",
            score=0.9,
            text="This is a test paper about optimization.",
        )
        summary = result.to_summary_string()
        assert "Test Paper" in summary
        assert "https://arxiv.org/test" in summary
        assert "optimization" in summary


class TestMockExaClient:
    """Tests for MockExaClient."""

    def test_search_returns_results(self):
        """Test mock search returns results."""
        client = MockExaClient()
        response = client.search_and_contents("test query", num_results=5)
        assert len(response.results) == 5

    def test_results_have_required_fields(self):
        """Test mock results have required fields."""
        client = MockExaClient()
        response = client.search_and_contents("test query", num_results=1)
        result = response.results[0]
        assert hasattr(result, "title")
        assert hasattr(result, "url")
        assert hasattr(result, "text")


class TestLiteratureDiscovery:
    """Tests for LiteratureDiscovery."""

    @pytest.fixture
    def discovery(self):
        """Create a LiteratureDiscovery instance with mock client."""
        return LiteratureDiscovery(
            exa_config=ExaConfig(api_key=None),  # Will use mock
            llm_config=LLMConfig(),
        )

    def test_search_papers_returns_results(self, discovery):
        """Test paper search returns results."""
        results = discovery.search_papers("PyTorch optimization", num_results=5)
        assert len(results) > 0
        assert all(isinstance(r, ExaSearchResult) for r in results)

    def test_get_sota_methods(self, discovery):
        """Test SOTA methods returns results."""
        results = discovery.get_sota_methods()
        assert len(results) > 0

    def test_fallback_gap_analysis(self, discovery):
        """Test fallback gap analysis."""
        analysis = discovery._fallback_gap_analysis("test topic")
        assert len(analysis.key_papers) > 0
        assert len(analysis.open_problems) > 0
        assert len(analysis.research_gaps) > 0


class TestLiteratureSummary:
    """Tests for LiteratureSummary."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from pytorch_scientist.dspy_programs import Paper, ResearchGap

        summary = LiteratureSummary(
            domain="test domain",
            key_papers=[Paper(title="Test", summary="Summary", relevance="high")],
            open_problems=["Problem 1"],
            unexplored_directions=["Direction 1"],
            recent_trends=["Trend 1"],
            research_gaps=[ResearchGap(description="Gap", potential_impact="high", difficulty="medium")],
        )
        d = summary.to_dict()
        assert d["domain"] == "test domain"
        assert len(d["key_papers"]) == 1
        assert len(d["open_problems"]) == 1

    def test_to_summary_string(self):
        """Test conversion to summary string."""
        from pytorch_scientist.dspy_programs import Paper, ResearchGap

        summary = LiteratureSummary(
            domain="test domain",
            key_papers=[Paper(title="Test Paper", summary="Summary", relevance="high")],
            open_problems=["Problem 1"],
            unexplored_directions=["Direction 1"],
            recent_trends=["Trend 1"],
            research_gaps=[ResearchGap(description="Gap", potential_impact="high", difficulty="medium")],
        )
        s = summary.to_summary_string()
        assert "test domain" in s
        assert "Test Paper" in s
        assert "Problem 1" in s

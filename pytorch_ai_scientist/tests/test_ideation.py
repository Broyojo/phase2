"""Tests for ideation module."""

import pytest

from pytorch_scientist.ideation import Idea


class TestIdea:
    """Tests for Idea dataclass."""

    def test_creation(self):
        """Test idea creation."""
        idea = Idea(
            title="Test Idea",
            description="A test idea",
            novelty="Novel approach",
            implementation_sketch="Steps to implement",
            expected_outcome="Better performance",
            risk_level="medium",
            config_space="Tiling parameters",
            feasibility_score=0.8,
            novelty_score=0.7,
        )
        assert idea.title == "Test Idea"
        assert idea.feasibility_score == 0.8

    def test_combined_score(self):
        """Test combined score calculation."""
        idea = Idea(
            title="Test",
            description="Test",
            novelty="Test",
            implementation_sketch="Test",
            expected_outcome="Test",
            risk_level="low",
            config_space="Test",
            feasibility_score=0.8,
            novelty_score=0.6,
        )
        # Combined score is average of novelty and feasibility
        assert idea.combined_score == 0.7

    def test_to_dict(self):
        """Test serialization to dictionary."""
        idea = Idea(
            title="Test Idea",
            description="Description",
            novelty="Novel",
            implementation_sketch="Steps",
            expected_outcome="Outcome",
            risk_level="low",
            config_space="Config",
            feasibility_score=0.9,
            novelty_score=0.8,
            source="test",
        )
        d = idea.to_dict()
        assert d["title"] == "Test Idea"
        assert d["feasibility_score"] == 0.9
        assert d["source"] == "test"

    def test_from_optimization_idea(self):
        """Test creation from OptimizationIdea."""
        from pytorch_scientist.dspy_programs import OptimizationIdea

        opt_idea = OptimizationIdea(
            title="Opt Idea",
            description="Description",
            novelty="Novel",
            implementation_sketch="Steps",
            expected_outcome="Outcome",
            risk_level="high",
            config_space="Config",
            feasibility_score=0.7,
            novelty_score=0.9,
        )
        idea = Idea.from_optimization_idea(opt_idea, source="dspy")
        assert idea.title == "Opt Idea"
        assert idea.source == "dspy"
        assert idea.novelty_score == 0.9

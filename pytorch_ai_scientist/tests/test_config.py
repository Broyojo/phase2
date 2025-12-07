"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from pytorch_scientist.config import (
    ExaConfig,
    ExperimentConfig,
    IdeationConfig,
    LLMConfig,
    LLMProvider,
    ResearchConfig,
    SearchConfig,
    SearchStrategy,
    TargetOperation,
    XSearchConfig,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_provider(self):
        """Test default provider is Grok."""
        config = LLMConfig()
        assert config.provider == LLMProvider.GROK

    def test_active_base_url_grok(self):
        """Test active base URL for Grok."""
        config = LLMConfig(provider=LLMProvider.GROK)
        assert config.active_base_url == "https://api.x.ai/v1"

    def test_active_base_url_anthropic(self):
        """Test active base URL for Anthropic."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        assert config.active_base_url == "https://api.anthropic.com"

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment."""
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        config = LLMConfig()
        assert config.grok_api_key == "test-key"
        assert config.active_api_key == "test-key"


class TestExaConfig:
    """Tests for ExaConfig."""

    def test_default_domains(self):
        """Test default include domains."""
        config = ExaConfig()
        assert "arxiv.org" in config.include_domains

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment."""
        monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
        config = ExaConfig()
        assert config.api_key == "test-exa-key"


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_strategy(self):
        """Test default search strategy."""
        config = SearchConfig()
        assert config.strategy == SearchStrategy.EVOLUTIONARY

    def test_population_size(self):
        """Test default population size."""
        config = SearchConfig()
        assert config.population_size == 10


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_operation(self):
        """Test default target operation."""
        config = ExperimentConfig()
        assert config.target_operation == TargetOperation.SOFTMAX

    def test_default_device(self):
        """Test default device."""
        config = ExperimentConfig()
        assert config.device == "cuda"


class TestIdeationConfig:
    """Tests for IdeationConfig."""

    def test_constraints(self):
        """Test default constraints."""
        config = IdeationConfig()
        assert len(config.constraints) > 0
        assert any("PyTorch" in c for c in config.constraints)


class TestResearchConfig:
    """Tests for ResearchConfig."""

    def test_default_domain(self):
        """Test default domain."""
        config = ResearchConfig()
        assert "PyTorch" in config.domain

    def test_run_dir_creation(self, tmp_path):
        """Test run directory creation."""
        config = ResearchConfig(output_dir=tmp_path)
        assert config.run_dir.exists()

    def test_from_env(self, monkeypatch):
        """Test creation from environment."""
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
        config = ResearchConfig.from_env()
        assert config.llm.grok_api_key == "test-key"
        assert config.exa.api_key == "test-exa-key"


class TestXSearchConfig:
    """Tests for XSearchConfig."""

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("X_API_KEY", "test-x-key")
        cfg = XSearchConfig()
        assert cfg.api_key == "test-x-key"

    def test_resolved_authors_from_file(self, tmp_path):
        file = tmp_path / "authors.txt"
        file.write_text("@alice\nbob\n@alice\n")
        cfg = XSearchConfig(authors=["carol"], authors_file=file)
        assert cfg.resolved_authors() == ["carol", "alice", "bob"]

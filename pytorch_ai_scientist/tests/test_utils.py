"""Tests for utility modules."""

import json
import tempfile
from pathlib import Path

import pytest

from pytorch_scientist.utils.persistence import (
    ArtifactManager,
    EnhancedJSONEncoder,
    load_json,
    load_yaml,
    save_artifact,
    save_json,
    save_yaml,
)
from pytorch_scientist.utils.timing import Timer, TimingResult, benchmark_function


class TestTimingResult:
    """Tests for TimingResult."""

    def test_str_format(self):
        """Test string formatting."""
        result = TimingResult(
            mean_ms=1.5,
            std_ms=0.1,
            min_ms=1.2,
            max_ms=2.0,
            iterations=100,
            total_ms=150.0,
        )
        s = str(result)
        assert "1.500ms" in s
        assert "n=100" in s

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TimingResult(
            mean_ms=1.5,
            std_ms=0.1,
            min_ms=1.2,
            max_ms=2.0,
            iterations=100,
            total_ms=150.0,
        )
        d = result.to_dict()
        assert d["mean_ms"] == 1.5
        assert d["iterations"] == 100


class TestTimer:
    """Tests for Timer context manager."""

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        import time

        with Timer() as t:
            time.sleep(0.01)  # 10ms

        assert t.elapsed_ms >= 10
        assert t.elapsed_ms < 100  # Should not take more than 100ms


class TestBenchmarkFunction:
    """Tests for benchmark_function."""

    def test_simple_function(self):
        """Test benchmarking a simple function."""
        counter = [0]

        def simple_func():
            counter[0] += 1
            return counter[0]

        result = benchmark_function(simple_func, warmup=2, iterations=5, sync_cuda=False)

        assert result.iterations == 5
        assert result.mean_ms > 0
        assert counter[0] == 7  # 2 warmup + 5 iterations


class TestEnhancedJSONEncoder:
    """Tests for EnhancedJSONEncoder."""

    def test_encode_enum(self):
        """Test encoding enums."""
        from pytorch_scientist.config import LLMProvider

        data = {"provider": LLMProvider.GROK}
        encoded = json.dumps(data, cls=EnhancedJSONEncoder)
        assert '"grok"' in encoded

    def test_encode_path(self):
        """Test encoding Path objects."""
        data = {"path": Path("/tmp/test")}
        encoded = json.dumps(data, cls=EnhancedJSONEncoder)
        assert "/tmp/test" in encoded

    def test_encode_dataclass(self):
        """Test encoding dataclasses."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            value: int
            name: str

        data = TestClass(value=42, name="test")
        encoded = json.dumps(data, cls=EnhancedJSONEncoder)
        assert "42" in encoded
        assert "test" in encoded


class TestSaveLoadJson:
    """Tests for save_json and load_json."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test save and load roundtrip."""
        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data

    def test_save_creates_directory(self, tmp_path):
        """Test save creates parent directory."""
        data = {"test": True}
        path = tmp_path / "subdir" / "test.json"

        save_json(data, path)

        assert path.exists()


class TestSaveLoadYaml:
    """Tests for save_yaml and load_yaml."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test save and load roundtrip."""
        data = {"key": "value", "list": [1, 2, 3]}
        path = tmp_path / "test.yaml"

        save_yaml(data, path)
        loaded = load_yaml(path)

        assert loaded == data


class TestSaveArtifact:
    """Tests for save_artifact."""

    def test_save_json_artifact(self, tmp_path):
        """Test saving JSON artifact."""
        data = {"artifact": "data"}
        path = save_artifact(data, "test_artifact", tmp_path, format="json")

        assert path.suffix == ".json"
        assert path.exists()

    def test_save_yaml_artifact(self, tmp_path):
        """Test saving YAML artifact."""
        data = {"artifact": "data"}
        path = save_artifact(data, "test_artifact", tmp_path, format="yaml")

        assert path.suffix == ".yaml"
        assert path.exists()


class TestArtifactManager:
    """Tests for ArtifactManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create an artifact manager."""
        return ArtifactManager(tmp_path)

    def test_directory_creation(self, manager):
        """Test subdirectories are created."""
        assert manager.literature_dir.exists()
        assert manager.ideas_dir.exists()
        assert manager.search_dir.exists()
        assert manager.results_dir.exists()

    def test_save_literature_summary(self, manager):
        """Test saving literature summary."""
        summary = {"papers": [], "gaps": []}
        path = manager.save_literature_summary(summary)
        assert path.exists()

    def test_save_ideas(self, manager):
        """Test saving ideas."""
        ideas = [{"title": "Idea 1"}, {"title": "Idea 2"}]
        path = manager.save_ideas(ideas)
        assert path.exists()

    def test_save_summary(self, manager):
        """Test saving markdown summary."""
        summary = "# Test Summary\n\nThis is a test."
        path = manager.save_summary(summary)
        assert path.exists()
        assert path.read_text() == summary

    def test_load_nonexistent_returns_none(self, manager):
        """Test loading nonexistent file returns None."""
        result = manager.load_literature_summary()
        assert result is None

    def test_save_load_roundtrip(self, manager):
        """Test save and load roundtrip for ideas."""
        ideas = [{"title": "Test Idea", "score": 0.9}]
        manager.save_ideas(ideas)
        loaded = manager.load_ideas()
        assert loaded == ideas

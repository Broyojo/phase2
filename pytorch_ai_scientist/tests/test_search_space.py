"""Tests for search space module."""

import pytest

from pytorch_scientist.search_space import (
    AttentionConfig,
    AttentionConfigSpace,
    HelionConfig,
    HelionConfigSpace,
    SoftmaxConfig,
    SoftmaxConfigSpace,
    TilingStrategy,
    TorchCompileConfig,
    TorchCompileConfigSpace,
    TorchCompileMode,
    get_config_space,
)


class TestHelionConfig:
    """Tests for HelionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HelionConfig()
        assert config.block_m == 64
        assert config.block_n == 64
        assert config.num_warps == 4

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = HelionConfig()
        d = config.to_dict()
        assert "block_m" in d
        assert "tiling_strategy" in d
        assert d["tiling_strategy"] == TilingStrategy.BLOCK.value

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "block_m": 128,
            "block_n": 64,
            "block_k": 32,
            "num_warps": 8,
            "num_stages": 2,
            "tiling_strategy": "tile_2d",
            "use_shared_memory": True,
            "shared_memory_size": 49152,
            "use_transposed_b": False,
            "use_atomic_add": False,
            "fuse_epilogue": True,
            "fuse_softmax": False,
        }
        config = HelionConfig.from_dict(d)
        assert config.block_m == 128
        assert config.tiling_strategy == TilingStrategy.TILE_2D


class TestHelionConfigSpace:
    """Tests for HelionConfigSpace."""

    @pytest.fixture
    def space(self):
        """Create a config space."""
        return HelionConfigSpace()

    def test_sample_random(self, space):
        """Test random sampling."""
        config = space.sample_random()
        assert isinstance(config, HelionConfig)
        assert space.validate(config)

    def test_mutate(self, space):
        """Test mutation."""
        config = space.sample_random()
        mutated = space.mutate(config, mutation_rate=1.0)  # Force mutation
        assert isinstance(mutated, HelionConfig)
        assert space.validate(mutated)

    def test_crossover(self, space):
        """Test crossover."""
        parent1 = space.sample_random()
        parent2 = space.sample_random()
        child = space.crossover(parent1, parent2)
        assert isinstance(child, HelionConfig)
        assert space.validate(child)

    def test_validate_valid_config(self, space):
        """Test validation of valid config."""
        config = HelionConfig()
        assert space.validate(config)

    def test_serialize_deserialize(self, space):
        """Test serialization round trip."""
        config = space.sample_random()
        serialized = space.serialize(config)
        deserialized = space.deserialize(serialized)
        assert deserialized.block_m == config.block_m
        assert deserialized.num_warps == config.num_warps


class TestTorchCompileConfig:
    """Tests for TorchCompileConfig."""

    def test_default_mode(self):
        """Test default compile mode."""
        config = TorchCompileConfig()
        assert config.mode == TorchCompileMode.DEFAULT

    def test_to_compile_kwargs(self):
        """Test conversion to compile kwargs."""
        config = TorchCompileConfig(mode=TorchCompileMode.MAX_AUTOTUNE)
        kwargs = config.to_compile_kwargs()
        assert kwargs["mode"] == "max-autotune"
        assert kwargs["backend"] == "inductor"


class TestTorchCompileConfigSpace:
    """Tests for TorchCompileConfigSpace."""

    @pytest.fixture
    def space(self):
        """Create a config space."""
        return TorchCompileConfigSpace()

    def test_sample_random(self, space):
        """Test random sampling."""
        config = space.sample_random()
        assert isinstance(config, TorchCompileConfig)
        assert space.validate(config)

    def test_mutate(self, space):
        """Test mutation."""
        config = space.sample_random()
        mutated = space.mutate(config)
        assert isinstance(mutated, TorchCompileConfig)

    def test_crossover(self, space):
        """Test crossover."""
        parent1 = space.sample_random()
        parent2 = space.sample_random()
        child = space.crossover(parent1, parent2)
        assert isinstance(child, TorchCompileConfig)


class TestSoftmaxConfigSpace:
    """Tests for SoftmaxConfigSpace."""

    @pytest.fixture
    def space(self):
        """Create a config space."""
        return SoftmaxConfigSpace()

    def test_sample_random(self, space):
        """Test random sampling."""
        config = space.sample_random()
        assert isinstance(config, SoftmaxConfig)
        assert space.validate(config)

    def test_validate_valid_block_sizes(self, space):
        """Test validation accepts valid block sizes."""
        config = SoftmaxConfig(block_size=256, num_warps=4)
        assert space.validate(config)

    def test_validate_invalid_block_size(self, space):
        """Test validation rejects invalid block sizes."""
        config = SoftmaxConfig(block_size=100, num_warps=4)  # Invalid
        assert not space.validate(config)


class TestAttentionConfigSpace:
    """Tests for AttentionConfigSpace."""

    @pytest.fixture
    def space(self):
        """Create a config space."""
        return AttentionConfigSpace()

    def test_sample_random(self, space):
        """Test random sampling."""
        config = space.sample_random()
        assert isinstance(config, AttentionConfig)
        assert space.validate(config)

    def test_mutate(self, space):
        """Test mutation."""
        config = space.sample_random()
        mutated = space.mutate(config)
        assert isinstance(mutated, AttentionConfig)


class TestGetConfigSpace:
    """Tests for get_config_space factory function."""

    def test_get_helion_space(self):
        """Test getting Helion config space."""
        space = get_config_space("helion")
        assert isinstance(space, HelionConfigSpace)

    def test_get_torch_compile_space(self):
        """Test getting torch.compile config space."""
        space = get_config_space("torch_compile")
        assert isinstance(space, TorchCompileConfigSpace)

    def test_get_softmax_space(self):
        """Test getting softmax config space."""
        space = get_config_space("softmax")
        assert isinstance(space, SoftmaxConfigSpace)

    def test_invalid_operation(self):
        """Test invalid operation raises error."""
        with pytest.raises(ValueError):
            get_config_space("invalid")

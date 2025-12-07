"""
Search space definitions for PyTorch/Helion config optimization.

Provides structured configuration spaces for:
- Helion kernel parameters (tiling, block sizes, etc.)
- torch.compile options
- Algorithm variants
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from pytorch_scientist.utils.logging import get_logger

logger = get_logger("search_space")


# Type variable for config types
T = TypeVar("T")


class TorchCompileMode(Enum):
    """torch.compile backend modes."""

    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


class MemoryFormat(Enum):
    """Memory layout formats."""

    CONTIGUOUS = "contiguous"
    CHANNELS_LAST = "channels_last"


class TilingStrategy(Enum):
    """Tiling strategies for kernel optimization."""

    NONE = "none"
    BLOCK = "block"
    TILE_2D = "tile_2d"
    TILE_3D = "tile_3d"


@dataclass
class HelionConfig:
    """
    Configuration for Helion kernel parameters.

    Represents tunable parameters for custom GPU kernels.
    """

    # Block/tile dimensions
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32

    # Number of warps
    num_warps: int = 4

    # Number of stages for pipelining
    num_stages: int = 2

    # Tiling strategy
    tiling_strategy: TilingStrategy = TilingStrategy.BLOCK

    # Memory settings
    use_shared_memory: bool = True
    shared_memory_size: int = 49152  # 48KB default

    # Algorithm variants
    use_transposed_b: bool = False
    use_atomic_add: bool = False

    # Fusion options
    fuse_epilogue: bool = True
    fuse_softmax: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "tiling_strategy": self.tiling_strategy.value,
            "use_shared_memory": self.use_shared_memory,
            "shared_memory_size": self.shared_memory_size,
            "use_transposed_b": self.use_transposed_b,
            "use_atomic_add": self.use_atomic_add,
            "fuse_epilogue": self.fuse_epilogue,
            "fuse_softmax": self.fuse_softmax,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HelionConfig":
        """Create from dictionary."""
        data = data.copy()
        if "tiling_strategy" in data:
            data["tiling_strategy"] = TilingStrategy(data["tiling_strategy"])
        return cls(**data)


@dataclass
class TorchCompileConfig:
    """
    Configuration for torch.compile options.
    """

    # Compile mode
    mode: TorchCompileMode = TorchCompileMode.DEFAULT

    # Backend selection
    backend: str = "inductor"

    # Dynamic shapes
    dynamic: bool = False

    # Full graph compilation
    fullgraph: bool = False

    # Disable optimizations (for debugging)
    disable: bool = False

    # Memory format
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS

    # Inductor-specific options
    inductor_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "backend": self.backend,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "disable": self.disable,
            "memory_format": self.memory_format.value,
            "inductor_config": self.inductor_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TorchCompileConfig":
        """Create from dictionary."""
        data = data.copy()
        if "mode" in data:
            data["mode"] = TorchCompileMode(data["mode"])
        if "memory_format" in data:
            data["memory_format"] = MemoryFormat(data["memory_format"])
        return cls(**data)

    def to_compile_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for torch.compile()."""
        kwargs = {
            "mode": self.mode.value,
            "backend": self.backend,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
        }

        # Add inductor config via options dict
        if self.inductor_config:
            kwargs["options"] = self.inductor_config

        return kwargs


@dataclass
class SoftmaxConfig:
    """
    Configuration for softmax kernel optimization.
    """

    # Block size for reduction
    block_size: int = 256

    # Number of warps
    num_warps: int = 4

    # Use online softmax algorithm
    use_online_softmax: bool = True

    # Fuse with scale
    fuse_scale: bool = True
    scale_factor: float = 1.0

    # Memory settings
    use_float32_accum: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "block_size": self.block_size,
            "num_warps": self.num_warps,
            "use_online_softmax": self.use_online_softmax,
            "fuse_scale": self.fuse_scale,
            "scale_factor": self.scale_factor,
            "use_float32_accum": self.use_float32_accum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SoftmaxConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AttentionConfig:
    """
    Configuration for attention kernel optimization.
    """

    # Block sizes
    block_q: int = 64
    block_kv: int = 64

    # Number of warps
    num_warps: int = 4
    num_stages: int = 2

    # Flash attention options
    use_flash_attention: bool = True
    causal: bool = False

    # Memory efficiency
    use_recompute: bool = True

    # Numerical precision
    use_float32_accum: bool = True
    softmax_scale: float | None = None  # Auto-computed if None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "block_q": self.block_q,
            "block_kv": self.block_kv,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "use_flash_attention": self.use_flash_attention,
            "causal": self.causal,
            "use_recompute": self.use_recompute,
            "use_float32_accum": self.use_float32_accum,
            "softmax_scale": self.softmax_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttentionConfig":
        """Create from dictionary."""
        return cls(**data)


# Type alias for any config type
ConfigType = HelionConfig | TorchCompileConfig | SoftmaxConfig | AttentionConfig


class ConfigSpace(ABC, Generic[T]):
    """
    Abstract base class for configuration spaces.

    Defines the interface for sampling, mutating, and validating configs.
    """

    @abstractmethod
    def sample_random(self) -> T:
        """Sample a random valid configuration."""
        pass

    @abstractmethod
    def mutate(self, config: T, mutation_rate: float = 0.2) -> T:
        """
        Mutate a configuration.

        Args:
            config: Configuration to mutate
            mutation_rate: Probability of mutating each parameter

        Returns:
            Mutated configuration
        """
        pass

    @abstractmethod
    def crossover(self, parent1: T, parent2: T) -> T:
        """
        Create offspring from two parent configurations.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Offspring configuration
        """
        pass

    @abstractmethod
    def validate(self, config: T) -> bool:
        """
        Validate a configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def serialize(self, config: T) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        pass

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> T:
        """Deserialize configuration from dictionary."""
        pass


class HelionConfigSpace(ConfigSpace[HelionConfig]):
    """
    Configuration space for Helion kernel parameters.
    """

    # Valid values for discrete parameters
    BLOCK_SIZES = [16, 32, 64, 128, 256]
    NUM_WARPS = [1, 2, 4, 8]
    NUM_STAGES = [1, 2, 3, 4]
    SHARED_MEM_SIZES = [16384, 32768, 49152, 65536]  # 16KB to 64KB

    def sample_random(self) -> HelionConfig:
        """Sample a random Helion configuration."""
        config = HelionConfig(
            block_m=random.choice(self.BLOCK_SIZES),
            block_n=random.choice(self.BLOCK_SIZES),
            block_k=random.choice([16, 32, 64]),
            num_warps=random.choice(self.NUM_WARPS),
            num_stages=random.choice(self.NUM_STAGES),
            tiling_strategy=random.choice(list(TilingStrategy)),
            use_shared_memory=random.choice([True, False]),
            shared_memory_size=random.choice(self.SHARED_MEM_SIZES),
            use_transposed_b=random.choice([True, False]),
            use_atomic_add=random.choice([True, False]),
            fuse_epilogue=random.choice([True, False]),
            fuse_softmax=random.choice([True, False]),
        )

        # Validate and fix if needed
        return self._fix_config(config)

    def mutate(self, config: HelionConfig, mutation_rate: float = 0.2) -> HelionConfig:
        """Mutate a Helion configuration."""
        new_config = HelionConfig(
            block_m=self._maybe_mutate_discrete(
                config.block_m, self.BLOCK_SIZES, mutation_rate
            ),
            block_n=self._maybe_mutate_discrete(
                config.block_n, self.BLOCK_SIZES, mutation_rate
            ),
            block_k=self._maybe_mutate_discrete(
                config.block_k, [16, 32, 64], mutation_rate
            ),
            num_warps=self._maybe_mutate_discrete(
                config.num_warps, self.NUM_WARPS, mutation_rate
            ),
            num_stages=self._maybe_mutate_discrete(
                config.num_stages, self.NUM_STAGES, mutation_rate
            ),
            tiling_strategy=self._maybe_mutate_discrete(
                config.tiling_strategy, list(TilingStrategy), mutation_rate
            ),
            use_shared_memory=self._maybe_flip(config.use_shared_memory, mutation_rate),
            shared_memory_size=self._maybe_mutate_discrete(
                config.shared_memory_size, self.SHARED_MEM_SIZES, mutation_rate
            ),
            use_transposed_b=self._maybe_flip(config.use_transposed_b, mutation_rate),
            use_atomic_add=self._maybe_flip(config.use_atomic_add, mutation_rate),
            fuse_epilogue=self._maybe_flip(config.fuse_epilogue, mutation_rate),
            fuse_softmax=self._maybe_flip(config.fuse_softmax, mutation_rate),
        )

        return self._fix_config(new_config)

    def crossover(
        self,
        parent1: HelionConfig,
        parent2: HelionConfig,
    ) -> HelionConfig:
        """Create offspring from two parent configurations."""
        # Uniform crossover
        config = HelionConfig(
            block_m=random.choice([parent1.block_m, parent2.block_m]),
            block_n=random.choice([parent1.block_n, parent2.block_n]),
            block_k=random.choice([parent1.block_k, parent2.block_k]),
            num_warps=random.choice([parent1.num_warps, parent2.num_warps]),
            num_stages=random.choice([parent1.num_stages, parent2.num_stages]),
            tiling_strategy=random.choice(
                [parent1.tiling_strategy, parent2.tiling_strategy]
            ),
            use_shared_memory=random.choice(
                [parent1.use_shared_memory, parent2.use_shared_memory]
            ),
            shared_memory_size=random.choice(
                [parent1.shared_memory_size, parent2.shared_memory_size]
            ),
            use_transposed_b=random.choice(
                [parent1.use_transposed_b, parent2.use_transposed_b]
            ),
            use_atomic_add=random.choice(
                [parent1.use_atomic_add, parent2.use_atomic_add]
            ),
            fuse_epilogue=random.choice(
                [parent1.fuse_epilogue, parent2.fuse_epilogue]
            ),
            fuse_softmax=random.choice(
                [parent1.fuse_softmax, parent2.fuse_softmax]
            ),
        )

        return self._fix_config(config)

    def validate(self, config: HelionConfig) -> bool:
        """Validate a Helion configuration."""
        # Check block sizes are valid
        if config.block_m not in self.BLOCK_SIZES:
            return False
        if config.block_n not in self.BLOCK_SIZES:
            return False
        if config.block_k not in [16, 32, 64]:
            return False

        # Check warps and stages
        if config.num_warps not in self.NUM_WARPS:
            return False
        if config.num_stages not in self.NUM_STAGES:
            return False

        # Check shared memory constraints
        required_shared = config.block_m * config.block_k * 4  # 4 bytes per float32
        if config.use_shared_memory and required_shared > config.shared_memory_size:
            return False

        return True

    def serialize(self, config: HelionConfig) -> dict[str, Any]:
        """Serialize configuration."""
        return config.to_dict()

    def deserialize(self, data: dict[str, Any]) -> HelionConfig:
        """Deserialize configuration."""
        return HelionConfig.from_dict(data)

    def _fix_config(self, config: HelionConfig) -> HelionConfig:
        """Fix configuration to ensure validity."""
        # Ensure shared memory is sufficient
        required_shared = config.block_m * config.block_k * 4
        if config.use_shared_memory and required_shared > config.shared_memory_size:
            # Find smallest sufficient size
            for size in self.SHARED_MEM_SIZES:
                if size >= required_shared:
                    config.shared_memory_size = size
                    break
            else:
                # Reduce block sizes
                config.block_m = min(config.block_m, 64)
                config.block_k = min(config.block_k, 32)

        return config

    def _maybe_mutate_discrete(
        self,
        value: Any,
        choices: list[Any],
        mutation_rate: float,
    ) -> Any:
        """Maybe mutate a discrete value."""
        if random.random() < mutation_rate:
            return random.choice(choices)
        return value

    def _maybe_flip(self, value: bool, mutation_rate: float) -> bool:
        """Maybe flip a boolean value."""
        if random.random() < mutation_rate:
            return not value
        return value


class TorchCompileConfigSpace(ConfigSpace[TorchCompileConfig]):
    """
    Configuration space for torch.compile options.
    """

    def sample_random(self) -> TorchCompileConfig:
        """Sample a random torch.compile configuration."""
        # Sample inductor options
        inductor_config = {}
        if random.choice([True, False]):
            inductor_config["max_autotune"] = random.choice([True, False])
        if random.choice([True, False]):
            inductor_config["epilogue_fusion"] = random.choice([True, False])
        if random.choice([True, False]):
            inductor_config["coordinate_descent_tuning"] = random.choice([True, False])

        return TorchCompileConfig(
            mode=random.choice(list(TorchCompileMode)),
            backend="inductor",
            dynamic=random.choice([True, False]),
            fullgraph=random.choice([True, False]),
            memory_format=random.choice(list(MemoryFormat)),
            inductor_config=inductor_config,
        )

    def mutate(
        self,
        config: TorchCompileConfig,
        mutation_rate: float = 0.2,
    ) -> TorchCompileConfig:
        """Mutate a torch.compile configuration."""
        new_inductor = config.inductor_config.copy()

        # Mutate inductor options
        if random.random() < mutation_rate:
            new_inductor["max_autotune"] = random.choice([True, False])
        if random.random() < mutation_rate:
            new_inductor["epilogue_fusion"] = random.choice([True, False])

        return TorchCompileConfig(
            mode=(
                random.choice(list(TorchCompileMode))
                if random.random() < mutation_rate
                else config.mode
            ),
            backend=config.backend,
            dynamic=(
                not config.dynamic if random.random() < mutation_rate else config.dynamic
            ),
            fullgraph=(
                not config.fullgraph
                if random.random() < mutation_rate
                else config.fullgraph
            ),
            memory_format=(
                random.choice(list(MemoryFormat))
                if random.random() < mutation_rate
                else config.memory_format
            ),
            inductor_config=new_inductor,
        )

    def crossover(
        self,
        parent1: TorchCompileConfig,
        parent2: TorchCompileConfig,
    ) -> TorchCompileConfig:
        """Create offspring from two parent configurations."""
        # Merge inductor configs
        new_inductor = {}
        all_keys = set(parent1.inductor_config.keys()) | set(
            parent2.inductor_config.keys()
        )
        for key in all_keys:
            if key in parent1.inductor_config and key in parent2.inductor_config:
                new_inductor[key] = random.choice(
                    [parent1.inductor_config[key], parent2.inductor_config[key]]
                )
            elif key in parent1.inductor_config:
                new_inductor[key] = parent1.inductor_config[key]
            else:
                new_inductor[key] = parent2.inductor_config[key]

        return TorchCompileConfig(
            mode=random.choice([parent1.mode, parent2.mode]),
            backend=parent1.backend,
            dynamic=random.choice([parent1.dynamic, parent2.dynamic]),
            fullgraph=random.choice([parent1.fullgraph, parent2.fullgraph]),
            memory_format=random.choice([parent1.memory_format, parent2.memory_format]),
            inductor_config=new_inductor,
        )

    def validate(self, config: TorchCompileConfig) -> bool:
        """Validate a torch.compile configuration."""
        # All combinations are generally valid
        return True

    def serialize(self, config: TorchCompileConfig) -> dict[str, Any]:
        """Serialize configuration."""
        return config.to_dict()

    def deserialize(self, data: dict[str, Any]) -> TorchCompileConfig:
        """Deserialize configuration."""
        return TorchCompileConfig.from_dict(data)


class SoftmaxConfigSpace(ConfigSpace[SoftmaxConfig]):
    """
    Configuration space for softmax kernel parameters.
    """

    BLOCK_SIZES = [64, 128, 256, 512, 1024]
    NUM_WARPS = [1, 2, 4, 8]

    def sample_random(self) -> SoftmaxConfig:
        """Sample a random softmax configuration."""
        return SoftmaxConfig(
            block_size=random.choice(self.BLOCK_SIZES),
            num_warps=random.choice(self.NUM_WARPS),
            use_online_softmax=random.choice([True, False]),
            fuse_scale=random.choice([True, False]),
            scale_factor=random.uniform(0.5, 2.0) if random.random() < 0.3 else 1.0,
            use_float32_accum=random.choice([True, False]),
        )

    def mutate(
        self,
        config: SoftmaxConfig,
        mutation_rate: float = 0.2,
    ) -> SoftmaxConfig:
        """Mutate a softmax configuration."""
        return SoftmaxConfig(
            block_size=(
                random.choice(self.BLOCK_SIZES)
                if random.random() < mutation_rate
                else config.block_size
            ),
            num_warps=(
                random.choice(self.NUM_WARPS)
                if random.random() < mutation_rate
                else config.num_warps
            ),
            use_online_softmax=(
                not config.use_online_softmax
                if random.random() < mutation_rate
                else config.use_online_softmax
            ),
            fuse_scale=(
                not config.fuse_scale
                if random.random() < mutation_rate
                else config.fuse_scale
            ),
            scale_factor=(
                random.uniform(0.5, 2.0)
                if random.random() < mutation_rate
                else config.scale_factor
            ),
            use_float32_accum=(
                not config.use_float32_accum
                if random.random() < mutation_rate
                else config.use_float32_accum
            ),
        )

    def crossover(
        self,
        parent1: SoftmaxConfig,
        parent2: SoftmaxConfig,
    ) -> SoftmaxConfig:
        """Create offspring from two parent configurations."""
        return SoftmaxConfig(
            block_size=random.choice([parent1.block_size, parent2.block_size]),
            num_warps=random.choice([parent1.num_warps, parent2.num_warps]),
            use_online_softmax=random.choice(
                [parent1.use_online_softmax, parent2.use_online_softmax]
            ),
            fuse_scale=random.choice([parent1.fuse_scale, parent2.fuse_scale]),
            scale_factor=random.choice([parent1.scale_factor, parent2.scale_factor]),
            use_float32_accum=random.choice(
                [parent1.use_float32_accum, parent2.use_float32_accum]
            ),
        )

    def validate(self, config: SoftmaxConfig) -> bool:
        """Validate a softmax configuration."""
        if config.block_size not in self.BLOCK_SIZES:
            return False
        if config.num_warps not in self.NUM_WARPS:
            return False
        return True

    def serialize(self, config: SoftmaxConfig) -> dict[str, Any]:
        """Serialize configuration."""
        return config.to_dict()

    def deserialize(self, data: dict[str, Any]) -> SoftmaxConfig:
        """Deserialize configuration."""
        return SoftmaxConfig.from_dict(data)


class AttentionConfigSpace(ConfigSpace[AttentionConfig]):
    """
    Configuration space for attention kernel parameters.
    """

    BLOCK_SIZES = [32, 64, 128]
    NUM_WARPS = [2, 4, 8]
    NUM_STAGES = [1, 2, 3, 4]

    def sample_random(self) -> AttentionConfig:
        """Sample a random attention configuration."""
        return AttentionConfig(
            block_q=random.choice(self.BLOCK_SIZES),
            block_kv=random.choice(self.BLOCK_SIZES),
            num_warps=random.choice(self.NUM_WARPS),
            num_stages=random.choice(self.NUM_STAGES),
            use_flash_attention=random.choice([True, False]),
            causal=random.choice([True, False]),
            use_recompute=random.choice([True, False]),
            use_float32_accum=random.choice([True, False]),
            softmax_scale=None,  # Auto-compute
        )

    def mutate(
        self,
        config: AttentionConfig,
        mutation_rate: float = 0.2,
    ) -> AttentionConfig:
        """Mutate an attention configuration."""
        return AttentionConfig(
            block_q=(
                random.choice(self.BLOCK_SIZES)
                if random.random() < mutation_rate
                else config.block_q
            ),
            block_kv=(
                random.choice(self.BLOCK_SIZES)
                if random.random() < mutation_rate
                else config.block_kv
            ),
            num_warps=(
                random.choice(self.NUM_WARPS)
                if random.random() < mutation_rate
                else config.num_warps
            ),
            num_stages=(
                random.choice(self.NUM_STAGES)
                if random.random() < mutation_rate
                else config.num_stages
            ),
            use_flash_attention=(
                not config.use_flash_attention
                if random.random() < mutation_rate
                else config.use_flash_attention
            ),
            causal=(
                not config.causal if random.random() < mutation_rate else config.causal
            ),
            use_recompute=(
                not config.use_recompute
                if random.random() < mutation_rate
                else config.use_recompute
            ),
            use_float32_accum=(
                not config.use_float32_accum
                if random.random() < mutation_rate
                else config.use_float32_accum
            ),
            softmax_scale=config.softmax_scale,
        )

    def crossover(
        self,
        parent1: AttentionConfig,
        parent2: AttentionConfig,
    ) -> AttentionConfig:
        """Create offspring from two parent configurations."""
        return AttentionConfig(
            block_q=random.choice([parent1.block_q, parent2.block_q]),
            block_kv=random.choice([parent1.block_kv, parent2.block_kv]),
            num_warps=random.choice([parent1.num_warps, parent2.num_warps]),
            num_stages=random.choice([parent1.num_stages, parent2.num_stages]),
            use_flash_attention=random.choice(
                [parent1.use_flash_attention, parent2.use_flash_attention]
            ),
            causal=random.choice([parent1.causal, parent2.causal]),
            use_recompute=random.choice([parent1.use_recompute, parent2.use_recompute]),
            use_float32_accum=random.choice(
                [parent1.use_float32_accum, parent2.use_float32_accum]
            ),
            softmax_scale=random.choice([parent1.softmax_scale, parent2.softmax_scale]),
        )

    def validate(self, config: AttentionConfig) -> bool:
        """Validate an attention configuration."""
        if config.block_q not in self.BLOCK_SIZES:
            return False
        if config.block_kv not in self.BLOCK_SIZES:
            return False
        if config.num_warps not in self.NUM_WARPS:
            return False
        if config.num_stages not in self.NUM_STAGES:
            return False
        return True

    def serialize(self, config: AttentionConfig) -> dict[str, Any]:
        """Serialize configuration."""
        return config.to_dict()

    def deserialize(self, data: dict[str, Any]) -> AttentionConfig:
        """Deserialize configuration."""
        return AttentionConfig.from_dict(data)


def get_config_space(operation: str) -> ConfigSpace[Any]:
    """
    Get the appropriate config space for an operation.

    Args:
        operation: One of "helion", "torch_compile", "softmax", "attention"

    Returns:
        ConfigSpace instance
    """
    spaces = {
        "helion": HelionConfigSpace(),
        "torch_compile": TorchCompileConfigSpace(),
        "softmax": SoftmaxConfigSpace(),
        "attention": AttentionConfigSpace(),
    }

    if operation not in spaces:
        raise ValueError(f"Unknown operation: {operation}. Valid: {list(spaces.keys())}")

    return spaces[operation]

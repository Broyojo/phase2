"""
Experiment execution and benchmarking for PyTorch optimization.

Provides:
- Benchmark harness for transformer operations
- Baseline implementations (eager, torch.compile)
- Helion kernel integration
- Metric collection and comparison
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from pytorch_scientist.config import ExperimentConfig, TargetOperation
from pytorch_scientist.search_space import (
    AttentionConfig,
    ConfigType,
    HelionConfig,
    SoftmaxConfig,
    TorchCompileConfig,
)
from pytorch_scientist.utils.logging import get_logger
from pytorch_scientist.utils.timing import TimingResult, benchmark_pytorch_function

logger = get_logger("experiments")


# Try to import torch
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, running in mock mode")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    latency_ms: float
    throughput: float | None = None  # ops/sec or GFLOPS
    memory_mb: float | None = None
    timing: TimingResult | None = None
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # "success", "error", "timeout"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "timing": self.timing.to_dict() if self.timing else None,
            "config": self.config,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class ExperimentResult:
    """Complete experiment result with baselines and comparisons."""

    config: dict[str, Any]
    latency_ms: float
    speedup_vs_baseline: float
    baseline_latency_ms: float
    timing: TimingResult | None = None
    status: str = "success"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "latency_ms": self.latency_ms,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "baseline_latency_ms": self.baseline_latency_ms,
            "timing": self.timing.to_dict() if self.timing else None,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @abstractmethod
    def setup(self, batch_size: int, seq_len: int, hidden_dim: int) -> None:
        """Set up the benchmark with problem size."""
        pass

    @abstractmethod
    def run_eager(self) -> Any:
        """Run eager PyTorch implementation."""
        pass

    @abstractmethod
    def run_compiled(self, config: TorchCompileConfig) -> Any:
        """Run torch.compile implementation."""
        pass

    @abstractmethod
    def run_custom(self, config: ConfigType) -> Any:
        """Run custom/Helion implementation."""
        pass

    @abstractmethod
    def verify_output(self, output: Any, reference: Any) -> bool:
        """Verify output correctness."""
        pass


class SoftmaxBenchmark(Benchmark):
    """
    Benchmark for softmax operation.
    """

    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize softmax benchmark.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        self.input_tensor: Any = None
        self.reference_output: Any = None
        self._compiled_fn: Any = None

    def setup(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        """Set up the benchmark."""
        batch_size = batch_size or self.config.default_batch_size
        seq_len = seq_len or self.config.default_seq_len
        hidden_dim = hidden_dim or self.config.default_hidden_dim

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using mock setup")
            return

        # Create input tensor
        self.input_tensor = torch.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.config.device,
            dtype=torch.float32,
        )

        # Compute reference output
        self.reference_output = F.softmax(self.input_tensor, dim=-1)

        # Clear compiled function cache
        self._compiled_fn = None

        logger.info(
            f"Set up SoftmaxBenchmark: shape={self.input_tensor.shape}, "
            f"device={self.config.device}"
        )

    def run_eager(self) -> Any:
        """Run eager softmax."""
        if not TORCH_AVAILABLE:
            return None
        return F.softmax(self.input_tensor, dim=-1)

    def run_compiled(self, config: TorchCompileConfig) -> Any:
        """Run compiled softmax."""
        if not TORCH_AVAILABLE:
            return None

        # Create compiled function if needed
        if self._compiled_fn is None or True:  # Always recompile for different configs
            torch._dynamo.reset()

            @torch.compile(**config.to_compile_kwargs())
            def compiled_softmax(x: torch.Tensor) -> torch.Tensor:
                return F.softmax(x, dim=-1)

            self._compiled_fn = compiled_softmax

        return self._compiled_fn(self.input_tensor)

    def run_custom(self, config: ConfigType) -> Any:
        """
        Run custom softmax implementation.

        This is where Helion kernels would be integrated.
        For now, we simulate different configs affecting performance.
        """
        if not TORCH_AVAILABLE:
            return None

        if isinstance(config, SoftmaxConfig):
            # Simulate different algorithm choices
            if config.use_online_softmax:
                # Online (numerically stable) softmax
                x = self.input_tensor
                x_max = x.max(dim=-1, keepdim=True).values
                x_exp = torch.exp(x - x_max)
                return x_exp / x_exp.sum(dim=-1, keepdim=True)
            else:
                # Standard softmax
                return F.softmax(self.input_tensor, dim=-1)

        # Fall back to eager for unsupported configs
        return self.run_eager()

    def verify_output(self, output: Any, reference: Any) -> bool:
        """Verify softmax output."""
        if not TORCH_AVAILABLE or output is None or reference is None:
            return True

        return torch.allclose(output, reference, rtol=1e-4, atol=1e-4)


class GEMMBenchmark(Benchmark):
    """
    Benchmark for GEMM (matrix multiplication) operation.
    """

    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize GEMM benchmark.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        self.a_tensor: Any = None
        self.b_tensor: Any = None
        self.reference_output: Any = None
        self._compiled_fn: Any = None

    def setup(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        """Set up the benchmark."""
        batch_size = batch_size or self.config.default_batch_size
        seq_len = seq_len or self.config.default_seq_len
        hidden_dim = hidden_dim or self.config.default_hidden_dim

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using mock setup")
            return

        # Create input tensors for batched GEMM
        # Shape: (batch, seq, hidden) @ (hidden, hidden) = (batch, seq, hidden)
        self.a_tensor = torch.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.b_tensor = torch.randn(
            hidden_dim,
            hidden_dim,
            device=self.config.device,
            dtype=torch.float32,
        )

        # Compute reference output
        self.reference_output = torch.matmul(self.a_tensor, self.b_tensor)

        self._compiled_fn = None

        logger.info(
            f"Set up GEMMBenchmark: A={self.a_tensor.shape}, B={self.b_tensor.shape}"
        )

    def run_eager(self) -> Any:
        """Run eager GEMM."""
        if not TORCH_AVAILABLE:
            return None
        return torch.matmul(self.a_tensor, self.b_tensor)

    def run_compiled(self, config: TorchCompileConfig) -> Any:
        """Run compiled GEMM."""
        if not TORCH_AVAILABLE:
            return None

        torch._dynamo.reset()

        @torch.compile(**config.to_compile_kwargs())
        def compiled_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

        return compiled_gemm(self.a_tensor, self.b_tensor)

    def run_custom(self, config: ConfigType) -> Any:
        """
        Run custom GEMM implementation.

        TODO: Integrate Helion kernel here.
        """
        if not TORCH_AVAILABLE:
            return None

        if isinstance(config, HelionConfig):
            # Placeholder for Helion kernel integration
            # Would use config.block_m, config.block_n, config.block_k, etc.

            # For now, use different algorithms based on config
            if config.use_transposed_b:
                return torch.matmul(self.a_tensor, self.b_tensor.T.contiguous().T)
            else:
                return torch.matmul(self.a_tensor, self.b_tensor)

        return self.run_eager()

    def verify_output(self, output: Any, reference: Any) -> bool:
        """Verify GEMM output."""
        if not TORCH_AVAILABLE or output is None or reference is None:
            return True

        return torch.allclose(output, reference, rtol=1e-3, atol=1e-3)


class AttentionBenchmark(Benchmark):
    """
    Benchmark for attention operation.
    """

    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize attention benchmark.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        self.query: Any = None
        self.key: Any = None
        self.value: Any = None
        self.reference_output: Any = None
        self._compiled_fn: Any = None

    def setup(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
    ) -> None:
        """Set up the benchmark."""
        batch_size = batch_size or self.config.default_batch_size
        seq_len = seq_len or self.config.default_seq_len
        hidden_dim = hidden_dim or self.config.default_hidden_dim
        num_heads = num_heads or self.config.default_num_heads

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using mock setup")
            return

        head_dim = hidden_dim // num_heads

        # Create QKV tensors
        self.query = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.key = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.value = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.config.device,
            dtype=torch.float32,
        )

        # Store dimensions
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len

        # Compute reference output using scaled dot-product attention
        scale = head_dim**-0.5
        attn_weights = torch.matmul(self.query, self.key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        self.reference_output = torch.matmul(attn_weights, self.value)

        self._compiled_fn = None

        logger.info(
            f"Set up AttentionBenchmark: Q/K/V={self.query.shape}, "
            f"heads={num_heads}, head_dim={head_dim}"
        )

    def run_eager(self) -> Any:
        """Run eager attention."""
        if not TORCH_AVAILABLE:
            return None

        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(self.query, self.key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, self.value)

    def run_compiled(self, config: TorchCompileConfig) -> Any:
        """Run compiled attention."""
        if not TORCH_AVAILABLE:
            return None

        torch._dynamo.reset()

        head_dim = self.head_dim

        @torch.compile(**config.to_compile_kwargs())
        def compiled_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            scale = head_dim**-0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.matmul(attn_weights, v)

        return compiled_attention(self.query, self.key, self.value)

    def run_custom(self, config: ConfigType) -> Any:
        """
        Run custom attention implementation.

        TODO: Integrate Helion/FlashAttention-style kernel here.
        """
        if not TORCH_AVAILABLE:
            return None

        if isinstance(config, AttentionConfig):
            # Use scaled_dot_product_attention if available (PyTorch 2.0+)
            if hasattr(F, "scaled_dot_product_attention"):
                return F.scaled_dot_product_attention(
                    self.query,
                    self.key,
                    self.value,
                    is_causal=config.causal,
                )

        return self.run_eager()

    def verify_output(self, output: Any, reference: Any) -> bool:
        """Verify attention output."""
        if not TORCH_AVAILABLE or output is None or reference is None:
            return True

        return torch.allclose(output, reference, rtol=1e-3, atol=1e-3)


class ExperimentRunner:
    """
    Runs experiments and collects metrics.
    """

    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config

        # Create appropriate benchmark
        self.benchmark = self._create_benchmark()

        # Baseline results cache
        self._baseline_eager: BenchmarkResult | None = None
        self._baseline_compiled: BenchmarkResult | None = None

        logger.info(f"Initialized ExperimentRunner for {experiment_config.target_operation}")

    def _create_benchmark(self) -> Benchmark:
        """Create the appropriate benchmark for the target operation."""
        if self.config.target_operation == TargetOperation.SOFTMAX:
            return SoftmaxBenchmark(self.config)
        elif self.config.target_operation == TargetOperation.GEMM:
            return GEMMBenchmark(self.config)
        elif self.config.target_operation == TargetOperation.ATTENTION:
            return AttentionBenchmark(self.config)
        else:
            raise ValueError(f"Unsupported operation: {self.config.target_operation}")

    def setup(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        """
        Set up the benchmark with problem size.

        Also computes baseline results.
        """
        self.benchmark.setup(batch_size, seq_len, hidden_dim)

        # Compute baselines
        self._compute_baselines()

    def _compute_baselines(self) -> None:
        """Compute baseline performance."""
        logger.info("Computing baseline performance...")

        # Eager baseline
        self._baseline_eager = self._benchmark_function(
            self.benchmark.run_eager,
            "eager",
        )

        # torch.compile baseline with default config
        default_compile = TorchCompileConfig()
        self._baseline_compiled = self._benchmark_function(
            lambda: self.benchmark.run_compiled(default_compile),
            "compiled_default",
        )

        logger.info(
            f"Baselines - Eager: {self._baseline_eager.latency_ms:.3f}ms, "
            f"Compiled: {self._baseline_compiled.latency_ms:.3f}ms"
        )

    def _benchmark_function(
        self,
        func: Callable[[], Any],
        name: str,
    ) -> BenchmarkResult:
        """Benchmark a function and return results."""
        if not TORCH_AVAILABLE:
            # Return mock result
            return BenchmarkResult(
                latency_ms=1.0,
                status="mock",
            )

        try:
            # Clear cache
            gc.collect()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()

            # Run benchmark
            timing = benchmark_pytorch_function(
                func,
                warmup=self.config.warmup_iterations,
                iterations=self.config.benchmark_iterations,
            )

            # Verify output
            output = func()
            if hasattr(self.benchmark, "reference_output"):
                is_correct = self.benchmark.verify_output(
                    output, self.benchmark.reference_output
                )
                if not is_correct:
                    logger.warning(f"Output verification failed for {name}")

            return BenchmarkResult(
                latency_ms=timing.mean_ms,
                timing=timing,
                status="success",
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {name}: {e}")
            return BenchmarkResult(
                latency_ms=float("inf"),
                status="error",
                error_message=str(e),
            )

    def evaluate_config(self, config: ConfigType) -> ExperimentResult:
        """
        Evaluate a single configuration.

        Args:
            config: Configuration to evaluate

        Returns:
            Experiment result with metrics and comparison to baseline
        """
        config_dict = (
            config.to_dict() if hasattr(config, "to_dict") else {"config": str(config)}
        )

        # Determine which runner to use
        if isinstance(config, TorchCompileConfig):
            func = lambda: self.benchmark.run_compiled(config)
        else:
            func = lambda: self.benchmark.run_custom(config)

        # Benchmark
        result = self._benchmark_function(func, f"config_{id(config)}")

        # Calculate speedup vs baseline
        baseline = self._baseline_compiled or self._baseline_eager
        baseline_latency = baseline.latency_ms if baseline else 1.0

        if result.latency_ms > 0 and result.latency_ms != float("inf"):
            speedup = baseline_latency / result.latency_ms
        else:
            speedup = 0.0

        return ExperimentResult(
            config=config_dict,
            latency_ms=result.latency_ms,
            speedup_vs_baseline=speedup,
            baseline_latency_ms=baseline_latency,
            timing=result.timing,
            status=result.status,
            error_message=result.error_message,
            metadata={
                "operation": self.config.target_operation.value,
                "device": self.config.device,
            },
        )

    def get_baseline_metrics(self) -> dict[str, Any]:
        """Get baseline metrics for reporting."""
        return {
            "eager_latency_ms": (
                self._baseline_eager.latency_ms if self._baseline_eager else None
            ),
            "compiled_latency_ms": (
                self._baseline_compiled.latency_ms if self._baseline_compiled else None
            ),
            "operation": self.config.target_operation.value,
            "device": self.config.device,
        }


def create_experiment_runner(
    operation: TargetOperation | str = TargetOperation.SOFTMAX,
    device: str = "cuda",
) -> ExperimentRunner:
    """
    Create an experiment runner for the given operation.

    Args:
        operation: Target operation
        device: Target device

    Returns:
        ExperimentRunner instance
    """
    if isinstance(operation, str):
        operation = TargetOperation(operation)

    config = ExperimentConfig(
        target_operation=operation,
        device=device,
    )

    return ExperimentRunner(config)

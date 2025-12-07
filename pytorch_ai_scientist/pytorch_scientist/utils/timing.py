"""
Timing utilities for PyTorch Scientist.

Provides accurate timing for benchmarks and profiling.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, TypeVar

import numpy as np

from pytorch_scientist.utils.logging import get_logger

logger = get_logger("timing")

T = TypeVar("T")


@dataclass
class TimingResult:
    """Results from timing a function."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    total_ms: float

    def __str__(self) -> str:
        return (
            f"mean={self.mean_ms:.3f}ms, "
            f"std={self.std_ms:.3f}ms, "
            f"min={self.min_ms:.3f}ms, "
            f"max={self.max_ms:.3f}ms, "
            f"n={self.iterations}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "iterations": self.iterations,
            "total_ms": self.total_ms,
        }


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer("my_operation") as t:
            # code to time
        print(f"Took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self, name: str = "operation", log: bool = False):
        """
        Initialize timer.

        Args:
            name: Name for logging
            log: Whether to log timing on exit
        """
        self.name = name
        self.log = log
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        if self.log:
            logger.info(f"{self.name}: {self.elapsed_ms:.3f}ms")

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000


@contextmanager
def timer(name: str = "operation") -> Generator[Timer, None, None]:
    """
    Context manager for timing code blocks.

    Args:
        name: Name for the timed operation

    Yields:
        Timer instance
    """
    t = Timer(name)
    with t:
        yield t


def benchmark_function(
    func: Callable[[], T],
    warmup: int = 10,
    iterations: int = 100,
    sync_cuda: bool = True,
) -> TimingResult:
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark (should take no arguments)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        sync_cuda: Whether to synchronize CUDA before timing

    Returns:
        TimingResult with statistics
    """
    # Try to import torch for CUDA sync
    torch_available = False
    try:
        import torch

        torch_available = True
    except ImportError:
        pass

    def maybe_sync() -> None:
        if sync_cuda and torch_available:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        func()
        maybe_sync()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        maybe_sync()
        start = time.perf_counter()
        func()
        maybe_sync()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times_array = np.array(times)

    return TimingResult(
        mean_ms=float(np.mean(times_array)),
        std_ms=float(np.std(times_array)),
        min_ms=float(np.min(times_array)),
        max_ms=float(np.max(times_array)),
        iterations=iterations,
        total_ms=float(np.sum(times_array)),
    )


def benchmark_pytorch_function(
    func: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 100,
    use_cuda_events: bool = True,
) -> TimingResult:
    """
    Benchmark a PyTorch function using CUDA events for accurate GPU timing.

    Args:
        func: Function to benchmark
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        use_cuda_events: Whether to use CUDA events for timing

    Returns:
        TimingResult with statistics
    """
    try:
        import torch
    except ImportError:
        # Fall back to regular benchmarking
        return benchmark_function(func, warmup, iterations, sync_cuda=False)

    if not torch.cuda.is_available() or not use_cuda_events:
        return benchmark_function(func, warmup, iterations, sync_cuda=torch.cuda.is_available())

    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Timed iterations with CUDA events
    times: list[float] = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func()
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times_array = np.array(times)

    return TimingResult(
        mean_ms=float(np.mean(times_array)),
        std_ms=float(np.std(times_array)),
        min_ms=float(np.min(times_array)),
        max_ms=float(np.max(times_array)),
        iterations=iterations,
        total_ms=float(np.sum(times_array)),
    )

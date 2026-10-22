"""Small reproducible benchmark utilities using only the standard library."""
from __future__ import annotations
import platform
import statistics
import sys
import time
import numpy as np


def benchmark_callable(func, *, repeat=5, number=1, warmup=1):
    """Measure a zero-argument callable repeatedly and return timing metadata.
    
    Parameters
    ----------
    func : callable
        Workload with no required arguments; return values are ignored.
    repeat : int, default=5
        Number of timed rounds; must be positive.
    number : int, default=1
        Calls per timed round; must be positive.
    warmup : int, default=1
        Untimed warm-up calls; must be non-negative.
    
    Returns
    -------
    dict[str, object]
        Per-call samples, median/mean/min seconds, run counts, and Python/NumPy/platform
        metadata for reproducibility.
    
    Raises
    ------
    ValueError
        If repeat/number/warmup are invalid.
    
    Complexity: Runtime is O((warmup + repeat*number)*cost(func)); stores O(repeat) timings.
    """
    for name, value in (("repeat", repeat), ("number", number), ("warmup", warmup)):
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if name != "warmup" else 0):
            raise ValueError(f"{name} has an invalid value")
    for _ in range(warmup):
        func()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            func()
        samples.append((time.perf_counter() - start) / number)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "repeat": repeat,
        "number": number,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def compare_implementations(vectorized, loop, *, repeat=5, number=1, warmup=1):
    """Benchmark vectorized and loop implementations and calculate their speed ratio.
    
    Parameters
    ----------
    vectorized, loop : callable
        Zero-argument functions expected to compute equivalent results.
    repeat, number, warmup : int
        Timing parameters forwarded to ``benchmark_callable``.
    
    Returns
    -------
    dict[str, object]
        Timing metadata for both implementations and ``speedup`` defined as loop median
        seconds divided by vectorized median seconds. Correctness is not checked here.
    
    Raises
    ------
    ValueError
        If timing parameters are invalid.
    
    Complexity: Sum of the two workloads' benchmark costs; O(repeat) timing storage.
    """
    vec = benchmark_callable(vectorized, repeat=repeat, number=number, warmup=warmup)
    lp = benchmark_callable(loop, repeat=repeat, number=number, warmup=warmup)
    vec_time, loop_time = vec["median_seconds"], lp["median_seconds"]
    return {
        "vectorized": vec,
        "loop": lp,
        "speedup": float(loop_time / vec_time) if vec_time > 0 else float("inf"),
    }

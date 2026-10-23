
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from numcompute.benchmarking import compare_implementations  # noqa: E402

README_START = "<!-- BENCHMARK_RESULTS_START -->"
README_END = "<!-- BENCHMARK_RESULTS_END -->"


def _markdown_report(result, *, size, seed, repeat, number, warmup):
    vec = result["vectorized"]
    loop = result["loop"]
    return "\n".join([
        "## Benchmark Results",
        "",
        f"- **Workload:** population variance of {size:,} deterministic normally distributed values",
        f"- **Seed:** `{seed}`",
        f"- **Timing protocol:** {warmup} warm-up call(s), {repeat} timed rounds, {number} call(s) per round; median reported",
        f"- **Environment:** Python `{vec['python']}`, NumPy `{vec['numpy']}`, platform `{vec['platform']}`",
        "",
        "| Implementation | Median runtime (seconds) | Relative speed |",
        "|---|---:|---:|",
        f"| Vectorized NumPy | {vec['median_seconds']:.6f} | **{result['speedup']:.2f}× faster** |",
        f"| Python loop | {loop['median_seconds']:.6f} | 1.00× baseline |",
        "",
        "The outputs were compared with `numpy.testing.assert_allclose` before timing. Results are machine-dependent; rerun this script on the machine used for the assignment and use that result in the submitted report.",
        "",
    ])


def _update_readme(section):
    readme = PROJECT_ROOT / "README.md"
    content = readme.read_text(encoding="utf-8") if readme.exists() else "# NumCompute\n"
    if README_START not in content or README_END not in content:
        raise RuntimeError(
            "README.md is missing benchmark markers. Keep these exact lines in the README:\n"
            f"{README_START}\n{README_END}"
        )
    before, remainder = content.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    readme.write_text(before + README_START + "\n" + section + README_END + after, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=200_000, help="Number of generated values (default: 200000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--repeat", type=int, default=5, help="Timed rounds per implementation (default: 5)")
    parser.add_argument("--number", type=int, default=1, help="Calls per timed round (default: 1)")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warm-up calls (default: 1)")
    args = parser.parse_args(argv)
    if args.size < 1:
        parser.error("--size must be positive")

    values = np.random.default_rng(args.seed).normal(size=args.size)

    def vectorized():
        center = values.mean()
        return np.mean((values - center) ** 2)

    def loop_based():
        average = sum(float(value) for value in values) / len(values)
        total = 0.0
        for value in values:
            delta = float(value) - average
            total += delta * delta
        return total / len(values)

    # A benchmark is valid only if implementations compute the same quantity.
    vec_value, loop_value = vectorized(), loop_based()
    np.testing.assert_allclose(vec_value, loop_value, rtol=1e-10, atol=1e-10)
    result = compare_implementations(
        vectorized, loop_based, repeat=args.repeat, number=args.number, warmup=args.warmup
    )
    section = _markdown_report(
        result, size=args.size, seed=args.seed, repeat=args.repeat,
        number=args.number, warmup=args.warmup,
    )

    results_path = PROJECT_ROOT / "benchmark" / "results.md"
    results_path.write_text(section, encoding="utf-8")
    _update_readme(section)

    print(f"Workload: population variance of {args.size:,} values (seed {args.seed})")
    print(f"Python: {result['vectorized']['python']} | NumPy: {result['vectorized']['numpy']}")
    print(f"Platform: {result['vectorized']['platform']}")
    print(f"Vectorized NumPy median: {result['vectorized']['median_seconds']:.6f} s")
    print(f"Python loop median:      {result['loop']['median_seconds']:.6f} s")
    print(f"Measured speedup:        {result['speedup']:.2f}x")
    print(f"Updated: {results_path.relative_to(PROJECT_ROOT)} and README.md")
    print("Use these local results in the assignment report; timings vary by machine and system load.")
    return result


if __name__ == "__main__":
    main()

# Benchmarking

Run the reproducible loop-versus-vectorized comparison from the project root:

```bash
python benchmark/compare_vectorized.py
```

The script generates deterministic normally distributed values (default size
200,000 and seed 42), compares a vectorized population-variance calculation with
a Python loop, and checks numerical equivalence before timing. It reports Python,
NumPy, and platform information, uses one warm-up and five timed repetitions by
default, and reports median runtime. It then updates the marked benchmark table
in `README.md` and creates `benchmark/results.md`.

Run the script on the computer you will describe in your report. Measurements
are affected by hardware, operating system, Python/NumPy versions, system load,
and timing settings. Optional settings can be inspected with:

```bash
python benchmark/compare_vectorized.py --help
```
<!-- BENCHMARK_RESULTS_START -->
## Benchmark Results

- **Workload:** Population variance of 200,000 deterministic normally distributed values
- **Seed:** `42`
- **Timing protocol:** 1 warm-up call, 5 timed rounds, 1 call per round; median reported
- **Environment:** Python `3.12.10`, NumPy `2.5.4`, platform `Windows-11-10.0.26200-SP0`

| Implementation | Median runtime (seconds) | Relative speed |
|---|---:|---:|
| Vectorized NumPy | 0.000677 | **28.50× faster** |
| Python loop | 0.019299 | 1.00× baseline |

Both implementations were checked for numerical equivalence using `numpy.testing.assert_allclose` before timing. Results are machine-dependent; rerun the benchmark on the machine used for the assignment report.

<!-- BENCHMARK_RESULTS_END -->
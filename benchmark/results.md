## Benchmark Results

- **Workload:** population variance of 200,000 deterministic normally distributed values
- **Seed:** `42`
- **Timing protocol:** 1 warm-up call(s), 5 timed rounds, 1 call(s) per round; median reported
- **Environment:** Python `3.12.10`, NumPy `2.5.4`, platform `Windows-11-10.0.26200-SP0`

| Implementation | Median runtime (seconds) | Relative speed |
|---|---:|---:|
| Vectorized NumPy | 0.000677 | **28.50× faster** |
| Python loop | 0.019299 | 1.00× baseline |

The outputs were compared with `numpy.testing.assert_allclose` before timing. Results are machine-dependent; rerun this script on the machine used for the assignment and use that result in the submitted report.

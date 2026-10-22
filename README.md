# NumCompute

NumCompute is a student-built scientific computing toolkit using plain Python and NumPy for the Programming for AI assignment.

## Requirements

- Python 3.10 or later
- NumPy

## Package modules

- `io.py`: CSV loading and chunked reading
- `preprocessing.py`: standard scaling, min-max scaling, imputation, one-hot encoding
- `sort_search.py`: stable sort, multi-key sorting, top-k, quickselect, binary search
- `rank.py`: tie-aware ranks and percentiles
- `stats.py`: descriptive statistics, quantiles, histogram, Welford accumulator
- `metrics.py`: classification metrics, confusion matrix, MSE, ROC/AUC
- `optim.py`: finite-difference gradient and Jacobian
- `pipeline.py`: transformer/estimator protocols, Pipeline, Compose, FeatureUnion, ColumnSelector
- `utils.py`: distances, activations, logsumexp, batching
- `benchmarking.py`: reproducible timing helper

## Run tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

## Run benchmark example

```bash
python benchmark/compare_vectorized.py
```

Check numerical outputs for correctness before using benchmark timings in the assignment report. Timing results are machine-dependent and should be reported from your own environment.

## Example

```python
import numpy as np
from numcompute.preprocessing import StandardScaler
from numcompute.sort_search import topk
from numcompute.optim import grad

X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
X_scaled = StandardScaler().fit_transform(X)
values, indices = topk(np.array([4.0, 10.0, 7.0]), k=2)
derivative = grad(lambda x: x**2, 3.0)
```

## API conventions

The supplied assignment team should review and agree on API semantics before treating this starter implementation as final. In particular, check label ordering, NaN handling, ranking direction, percentile q units, axis behaviour, and edge cases against the course requirements.


<!-- BENCHMARK_RESULTS_START -->
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
<!-- BENCHMARK_RESULTS_END -->

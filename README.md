# NumCompute

NumCompute is a lightweight scientific computing toolkit built with Python and NumPy for the Programming for AI assignment. It brings together common data handling, preprocessing, sorting, ranking, statistics, evaluation, numerical differentiation, and pipeline utilities behind a modular API.

The project is intended for learning and experimentation. Core computations use NumPy, and the project does not depend on pandas, scikit-learn, or deep learning frameworks.

## Features

- Read CSV data into NumPy arrays, including support for missing values and chunked reading.
- Preprocess numeric and categorical data with standard scaling, min-max scaling, imputation, and one-hot encoding.
- Sort values, perform multi-key sorting, select top-k values, use quickselect, and search sorted arrays.
- Calculate ranks with tie handling and compute percentiles.
- Calculate descriptive statistics, quantiles, histograms, and streaming statistics.
- Evaluate predictions with classification and regression metrics, including confusion matrices and ROC/AUC utilities.
- Estimate gradients and Jacobians using finite differences.
- Combine transformers and estimators with lightweight pipeline abstractions.
- Use utility functions for distances, activations, stable log-sum-exp, and batching.
- Compare vectorized NumPy code with a Python-loop implementation using a reproducible benchmark script.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer

The package metadata in `pyproject.toml` declares NumPy as its runtime dependency.

## Installation

Open a terminal at the root of the repository, where `pyproject.toml` is located, and run:

```bash
python -m pip install -e .
```

This installs NumCompute in editable mode, so changes to the local package are available without reinstalling it. If you prefer not to install the package, run project commands from the repository root so Python can find the `numcompute` package.

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
├── demo/
│   ├── quickstart.ipynb
│   └── data/
│       └── numcompute_student_performance.csv
├── benchmark/
│   ├── compare_vectorized.py
│   └── results.md
├── API_CONVENTIONS.md
├── README.md
└── pyproject.toml
```

## Package Overview

| Module | Purpose |
|---|---|
| `io.py` | CSV loading and chunked reading into NumPy arrays |
| `preprocessing.py` | StandardScaler, MinMaxScaler, imputation, and OneHotEncoder |
| `sort_search.py` | Stable sorting, multi-key sorting, top-k, quickselect, and binary search |
| `rank.py` | Ranking methods with tie handling and percentile calculations |
| `stats.py` | Descriptive statistics, quantiles, histograms, and streaming statistics |
| `metrics.py` | Accuracy, precision, recall, F1, confusion matrix, MSE, ROC, and AUC utilities |
| `optim.py` | Finite-difference gradient and Jacobian estimation |
| `pipeline.py` | Transformer and estimator protocols, Pipeline, Compose, FeatureUnion, and ColumnSelector |
| `utils.py` | Distances, activation functions, log-sum-exp, top-k helpers, and batching utilities |
| `benchmarking.py` | Timing helpers for repeatable performance comparisons |

Public functions and classes include docstrings describing their intended behaviour. `API_CONVENTIONS.md` records shared conventions for array shapes, missing values, ranking, metrics, testing, and team integration.

## Quick Example

```python
import numpy as np
from numcompute.preprocessing import StandardScaler
from numcompute.sort_search import topk
from numcompute.optim import grad

# Standardize a two-column feature matrix.
X = np.array([
    [1.0, 10.0],
    [3.0, 30.0],
    [5.0, 50.0],
])
X_scaled = StandardScaler().fit_transform(X)

# Select the two largest values and their original indices.
values, indices = topk(np.array([4.0, 10.0, 7.0]), k=2)

# Estimate the derivative of f(x) = x^2 at x = 3.
derivative = grad(lambda x: x**2, 3.0)

print("Scaled features:\n", X_scaled)
print("Top values:", values)
print("Original indices:", indices)
print("Estimated derivative:", derivative)
```

For more complete examples, open `demo/quickstart.ipynb`. The notebook uses the supplied student-performance CSV to demonstrate data loading, missing-value handling, preprocessing, sorting and ranking, statistics, metrics, numerical differentiation, and benchmarking.

## Run the Tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

This discovers the project's unit tests and prints each test as it runs. The current test suite contains 92 tests, including checks for common edge cases such as empty inputs, missing values, ties, invalid arguments, numerical extremes, and non-contiguous arrays. Run the command in your own environment before submitting and confirm the final output ends with `OK`.

## Run the Benchmark

From the repository root, run:

```bash
python benchmark/compare_vectorized.py
```

The script creates deterministic normally distributed data, compares vectorized NumPy population variance with a Python-loop implementation, checks numerical equivalence before timing, and records runtime and environment details. It uses one warm-up and five timed rounds by default, and reports median timings. It updates the marked results section in this README and writes detailed output to `benchmark/results.md`.

To see optional command-line settings, run:

```bash
python benchmark/compare_vectorized.py --help
```

<!-- BENCHMARK_RESULTS_START -->
## Benchmark Results

The following measurements were produced on the computer used for the current benchmark run.

- **Workload:** Population variance of 200,000 deterministic normally distributed values
- **Seed:** `42`
- **Timing protocol:** 1 warm-up call, 5 timed rounds, 1 call per round, median reported
- **Environment:** Python `3.12.10`, NumPy `2.5.4`, Windows 11 (`Windows-11-10.0.26200-SP0`)

| Implementation | Median runtime (seconds) | Relative speed |
|---|---:|---:|
| Vectorized NumPy | 0.000677 | **28.50x faster** |
| Python loop | 0.019299 | 1.00x baseline |

The outputs were compared with `numpy.testing.assert_allclose` before timing. These measurements apply to this workload and environment only. Runtime can vary with hardware, operating system, software versions, and system load. Rerun the script on the machine used for the assignment report and use those measurements in the submitted report.
<!-- BENCHMARK_RESULTS_END -->

## Design Conventions

- Tabular input normally uses the shape `(n_samples, n_features)`.
- Transformer classes use `fit`, `transform`, and `fit_transform` where applicable.
- Functions document their expected shapes and behaviour for missing values.
- Numerical routines should favour vectorized NumPy operations and numerically stable calculations where appropriate.
- Sorting and ranking methods document tie behaviour and their output conventions.
- Run the test suite after changing implementation code, and record benchmark environment details when reporting timings.

## Assignment Deliverables

The repository contains the NumCompute package, unit tests, quickstart notebook and dataset, benchmark script and results, configuration, and documentation. The assignment also requires a demonstration video of no more than five minutes and a technical report of no more than four pages. Submit those separately through the course submission system unless your instructor requests them in this repository.

## License and Use

This project was created for an educational programming assignment. Review the course requirements and your team's contribution notes before redistributing or reusing the work.

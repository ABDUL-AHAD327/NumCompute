# NumCompute tests

Run the complete test suite from the NumCompute project root:

```bash
python -m unittest discover -s tests -v
```

The test suite uses Python's built-in `unittest` module and NumPy only. The
original test file checks the core expected functionality. `test_edge_cases.py`
adds cases for missing values, invalid arguments, ties, non-contiguous arrays,
empty inputs, unknown categories, numerical extremes, shape mismatches, pipeline
validation, and deterministic batching.

The suite currently contains **92 tests**. Re-run it after any code changes and
record the count and result from the submitted environment.

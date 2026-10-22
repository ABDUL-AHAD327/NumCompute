# NumCompute API Conventions

- Use only Python and NumPy for implementation.
- Tabular data generally uses `(n_samples, n_features)` shapes.
- Numeric missing values are represented by `numpy.nan`.
- Preprocessors implement `fit`, `transform`, and `fit_transform`; `fit` returns `self`.
- Standard deviation defaults to population standard deviation (`ddof=0`).
- Ranking is ascending by default; the smallest valid value receives rank 1.
- Percentile helper in `rank.py` accepts q on a 0–100 scale; `stats.quantile` accepts q on a 0–1 scale, matching NumPy.
- Binary search requires ascending sorted input and returns `(leftmost_insertion_index, exists)`.
- Binary classification metrics use the supplied `pos_label` (default `1`).
- Confusion matrix rows are true labels and columns are predicted labels.
- All public functions should retain docstrings and informative validation errors.
- Test empty inputs, ties, NaNs, invalid shapes, non-contiguous arrays, and edge values.

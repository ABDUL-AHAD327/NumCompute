"""Ranking and percentile operations with explicit tie semantics."""
from __future__ import annotations
import numpy as np


def rank(data, method="average"):
    """Assign ranks with tie handling, preserving the input shape.
    
    Parameters
    ----------
    data : array-like
        Numeric or sortable values. NaNs in numeric arrays receive NaN ranks.
    method : {'average', 'dense', 'ordinal'}, default='average'
        'average' gives tied values the average occupied rank; 'dense' increments
        only when a new value appears; 'ordinal' gives stable sequential ranks.
    
    Returns
    -------
    numpy.ndarray
        Floating-point ranks with the same shape as data.
    
    Raises
    ------
    ValueError
        If method is not supported.
    
    Complexity: O(n log n) time and O(n) auxiliary storage for n flattened values.
    """
    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'")
    arr = np.asarray(data)
    original_shape = arr.shape
    flat = arr.reshape(-1)
    if flat.size == 0:
        return np.empty(original_shape, dtype=float)
    if np.issubdtype(flat.dtype, np.number):
        valid_mask = ~np.isnan(flat) if np.issubdtype(flat.dtype, np.inexact) else np.ones(flat.size, bool)
    else:
        valid_mask = np.ones(flat.size, dtype=bool)
    valid = flat[valid_mask]
    result = np.full(flat.size, np.nan, dtype=float)
    if valid.size == 0:
        return result.reshape(original_shape)
    if method == "ordinal":
        order = np.argsort(valid, kind="stable")
        ordinal = np.empty(valid.size, dtype=float)
        ordinal[order] = np.arange(1, valid.size + 1, dtype=float)
        result[valid_mask] = ordinal
    else:
        unique, inverse, counts = np.unique(valid, return_inverse=True, return_counts=True)
        if method == "dense":
            result[valid_mask] = inverse.astype(float) + 1.0
        else:
            ends = np.cumsum(counts, dtype=float)
            starts = ends - counts + 1.0
            mean_ranks = (starts + ends) / 2.0
            result[valid_mask] = mean_ranks[inverse]
    return result.reshape(original_shape)


def percentile(data, q, interpolation="linear"):
    """Compute one or more percentiles while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Values from which to compute percentiles.
    q : float or array-like
        Percentile(s) on a 0–100 scale.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        Quantile interpolation rule.
    
    Returns
    -------
    float or numpy.ndarray
        Scalar for scalar q; otherwise an array matching q's shape.
    
    Raises
    ------
    ValueError
        If data has no non-NaN values, q is non-finite/outside 0..100, or method is invalid.
    
    Complexity: O(n log n) time in the general case.
    """
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("interpolation must be 'linear', 'lower', 'higher', or 'midpoint'")
    arr = np.asarray(data, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        raise ValueError("data must contain at least one non-NaN value")
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 100) | ~np.isfinite(q_arr)):
        raise ValueError("q must contain finite percentiles between 0 and 100")
    scalar_q = q_arr.ndim == 0
    result = np.nanquantile(arr, q_arr / 100.0, method=interpolation)
    return float(result) if scalar_q else result

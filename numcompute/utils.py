"""Reusable numerical helpers: distances, activations, logsumexp, and batching."""
from __future__ import annotations
from typing import Iterator
import numpy as np


def relu(x):
    """Apply the rectified-linear activation elementwise.
    
    Parameters
    ----------
    x : scalar or array-like
        Input values.
    
    Returns
    -------
    float or numpy.ndarray
        ``max(0, x)`` elementwise; scalar input returns a Python float.
    
    Raises
    ------
    None
        No explicit input validation is performed; NumPy conversion errors propagate.

    Complexity: O(n) time and O(n) output storage.
    """
    arr = np.asarray(x)
    out = np.maximum(arr, 0)
    return float(out) if out.ndim == 0 else out


def sigmoid(x):
    """Apply a numerically stable logistic sigmoid elementwise.
    
    Parameters
    ----------
    x : scalar or array-like
        Input values; large positive and negative magnitudes are handled without overflow.
    
    Returns
    -------
    float or numpy.ndarray
        Values in [0, 1], with scalar input returned as a Python float.
    
    Raises
    ------
    None
        No explicit input validation is performed; NumPy conversion errors propagate.

    Complexity: O(n) time and O(n) output storage.
    """
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr, dtype=float)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return float(out) if out.ndim == 0 else out


def tanh(x):
    """Apply the hyperbolic tangent activation elementwise.
    
    Parameters
    ----------
    x : scalar or array-like
        Input values.
    
    Returns
    -------
    float or numpy.ndarray
        Values in [-1, 1]; scalar input returns a Python float.
    
    Raises
    ------
    None
        No explicit input validation is performed; NumPy conversion errors propagate.

    Complexity: O(n) time and O(n) output storage.
    """
    out = np.tanh(np.asarray(x, dtype=float))
    return float(out) if out.ndim == 0 else out


def softmax(x, axis=-1):
    """Compute numerically stable softmax probabilities along an axis.
    
    Parameters
    ----------
    x : array-like
        Numeric input scores.
    axis : int, default=-1
        Axis along which probabilities are normalized.
    
    Returns
    -------
    numpy.ndarray
        Same shape as x; finite slices sum to one. NaNs propagate, positive
        infinities share probability equally, and all-negative-infinity slices yield NaNs.
    
    Raises
    ------
    numpy.AxisError / ValueError
        If the selected axis is invalid.
    
    Complexity: O(n) time and O(n) temporary/output storage for n elements.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.ones_like(arr)
    max_val = np.max(arr, axis=axis, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = arr - max_val
        exp_shifted = np.exp(shifted)
        denom = np.sum(exp_shifted, axis=axis, keepdims=True)
        result = exp_shifted / denom
    positive_inf = np.isposinf(max_val)
    if np.any(positive_inf):
        mask = np.isposinf(arr)
        count = np.sum(mask, axis=axis, keepdims=True)
        inf_result = np.divide(mask.astype(float), count, out=np.zeros_like(arr, dtype=float), where=count > 0)
        result = np.where(positive_inf, inf_result, result)
    return result


def logsumexp(x, axis=None, keepdims=False):
    """Compute log(sum(exp(x))) with max-shifting for numerical stability.
    
    Parameters
    ----------
    x : array-like
        Numeric values.
    axis : int or tuple of int, optional
        Reduction axis/axes; None reduces all elements.
    keepdims : bool, default=False
        Preserve reduced axes as length one.
    
    Returns
    -------
    float or numpy.ndarray
        Log-sum-exp reduction. Infinite extrema are handled explicitly and NaNs propagate.
    
    Raises
    ------
    ValueError
        If x is empty or axis is invalid.
    
    Complexity: O(n) time and O(n) temporary storage for n elements.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError("logsumexp cannot operate on an empty array")
    max_val = np.max(arr, axis=axis, keepdims=True)
    finite_max = np.isfinite(max_val)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = np.where(finite_max, arr - max_val, -np.inf)
        result = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + max_val
    result = np.where(np.isposinf(max_val), np.inf, result)
    result = np.where(np.isneginf(max_val), -np.inf, result)
    result = np.where(np.isnan(max_val), np.nan, result)
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    elif not keepdims and axis is None:
        result = np.asarray(result).reshape(())
    return float(result) if np.asarray(result).ndim == 0 else result


def euclidean_distance(a, b):
    """Compute Euclidean distance between two same-shaped arrays.
    
    Parameters
    ----------
    a, b : array-like
        Coordinate arrays with identical shapes.
    
    Returns
    -------
    float
        Square root of the sum of squared elementwise differences.
    
    Raises
    ------
    ValueError
        If input shapes differ.
    
    Complexity: O(n) time for n coordinates.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have identical shapes")
    return float(np.linalg.norm(a_arr - b_arr))


def manhattan_distance(a, b):
    """Compute Manhattan (L1) distance between two same-shaped arrays.
    
    Parameters
    ----------
    a, b : array-like
        Coordinate arrays with identical shapes.
    
    Returns
    -------
    float
        Sum of absolute elementwise differences.
    
    Raises
    ------
    ValueError
        If input shapes differ.
    
    Complexity: O(n) time and O(n) temporary storage.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have identical shapes")
    return float(np.sum(np.abs(a_arr - b_arr)))


def pairwise_distances(X, Y=None, metric="euclidean"):
    """Compute distances between every row of X and every row of Y.
    
    Parameters
    ----------
    X : array-like, shape (n_samples_X, n_features)
        First set of observations.
    Y : array-like, shape (n_samples_Y, n_features), optional
        Second set; when omitted, distances among rows of X are returned.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use.
    
    Returns
    -------
    numpy.ndarray, shape (n_samples_X, n_samples_Y)
        Pairwise distance matrix.
    
    Raises
    ------
    ValueError
        If either input is not 2D, feature counts differ, or metric is unsupported.
    
    Complexity: Euclidean O(n*m*d) time with O(n*m) output; Manhattan O(n*m*d)
    with O(n*m*d) broadcast temporary storage.
    """
    x = np.asarray(X, dtype=float)
    y = x if Y is None else np.asarray(Y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if metric == "euclidean":
        x2 = np.sum(x * x, axis=1)[:, None]
        y2 = np.sum(y * y, axis=1)[None, :]
        squared = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
        return np.sqrt(squared)
    if metric == "manhattan":
        return np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)
    raise ValueError("metric must be 'euclidean' or 'manhattan'")


def iter_batches(X, batch_size, y=None, *, shuffle=False, seed=None, drop_last=False):
    """Yield mini-batches from X, optionally paired with y.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, ...)
        Input samples.
    batch_size : int
        Positive number of samples per batch.
    y : array-like, optional
        Targets with the same first dimension as X.
    shuffle : bool, default=False
        Shuffle sample indices before batching.
    seed : int, optional
        Random seed used when shuffle is True, for reproducible order.
    drop_last : bool, default=False
        Drop the final batch if it is smaller than batch_size.
    
    Yields
    ------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        X batches alone, or aligned (X_batch, y_batch) pairs.
    
    Raises
    ------
    ValueError
        If batch_size is invalid, X is scalar, or y has an incompatible sample count.
    
    Complexity: O(n) index setup plus O(n) overall indexing; shuffled mode uses O(n) indices.
    """
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError("X must have a sample dimension")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    labels = None if y is None else np.asarray(y)
    if labels is not None and (labels.ndim == 0 or labels.shape[0] != arr.shape[0]):
        raise ValueError("y must have the same number of samples as X")
    indices = np.arange(arr.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        if drop_last and len(batch_idx) < batch_size:
            break
        xb = arr[batch_idx]
        if labels is None:
            yield xb
        else:
            yield xb, labels[batch_idx]

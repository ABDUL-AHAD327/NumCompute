"""Descriptive and streaming statistics implemented with NumPy."""
from __future__ import annotations
import numpy as np


def mean(data, axis=None, keepdims=False):
    """Compute the arithmetic mean while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces all values.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Mean values with NumPy-compatible reduction shape. Empty/all-NaN slices yield NaN.
    
    Raises
    ------
    ValueError
        If axis is invalid.
    
    Complexity: O(n) time for n elements.
    """
    arr = np.asarray(data, dtype=float)
    return np.nanmean(arr, axis=axis, keepdims=keepdims)


def median(data, axis=None, keepdims=False):
    """Compute the median while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces all values.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Median values following NumPy's reduction-shape rules. Empty/all-NaN slices yield NaN.
    
    Raises
    ------
    ValueError
        If axis is invalid.
    
    Complexity: O(n log n) typical time per reduced slice.
    """
    arr = np.asarray(data, dtype=float)
    return np.nanmedian(arr, axis=axis, keepdims=keepdims)


def std(data, axis=None, ddof=0, keepdims=False):
    """Compute NaN-aware standard deviation.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces all values.
    ddof : int, default=0
        Delta degrees of freedom; 0 gives population standard deviation.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Standard deviations. Slices with insufficient valid observations produce NaN.
    
    Raises
    ------
    ValueError
        If ddof is negative or axis is invalid.
    
    Complexity: O(n) time for n elements.
    """
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    return np.nanstd(np.asarray(data, dtype=float), axis=axis, ddof=ddof, keepdims=keepdims)


def variance(data, axis=None, ddof=0, keepdims=False):
    """Compute NaN-aware variance.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces all values.
    ddof : int, default=0
        Delta degrees of freedom; 0 gives population variance.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Variances. Slices with insufficient valid observations produce NaN.
    
    Raises
    ------
    ValueError
        If ddof is negative or axis is invalid.
    
    Complexity: O(n) time for n elements.
    """
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    return np.nanvar(np.asarray(data, dtype=float), axis=axis, ddof=ddof, keepdims=keepdims)


def minimum(data, axis=None, keepdims=False):
    """Compute the minimum while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces the entire array.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Minimum values with NumPy-compatible reduction shape. An all-NaN slice returns NaN and emits RuntimeWarning.
    
    Raises
    ------
    ValueError
        If input is empty or axis is invalid. An all-NaN slice returns NaN rather than raising ValueError.
    
    Complexity: O(n) time for n elements.
    """
    return np.nanmin(np.asarray(data, dtype=float), axis=axis, keepdims=keepdims)


def maximum(data, axis=None, keepdims=False):
    """Compute the maximum while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    axis : int or tuple of int, optional
        Axis or axes to reduce; None reduces the entire array.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Maximum values with NumPy-compatible reduction shape. An all-NaN slice returns NaN and emits RuntimeWarning.
    
    Raises
    ------
    ValueError
        If input is empty or axis is invalid. An all-NaN slice returns NaN rather than raising ValueError.
    
    Complexity: O(n) time for n elements.
    """
    return np.nanmax(np.asarray(data, dtype=float), axis=axis, keepdims=keepdims)


def quantile(data, q, axis=None, interpolation="linear", keepdims=False):
    """Compute quantiles while ignoring NaNs.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    q : float or array-like
        Quantile probabilities on a 0–1 scale, inclusive.
    axis : int or tuple of int, optional
        Axis or axes along which to compute quantiles.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default='linear'
        Quantile interpolation rule.
    keepdims : bool, default=False
        Retain reduced axes as length-one dimensions.
    
    Returns
    -------
    float or numpy.ndarray
        Quantile result following NumPy's q/axis/keepdims shape conventions.
    
    Raises
    ------
    ValueError
        If q is non-finite/outside 0..1, interpolation is unsupported, or axis is invalid.
    
    Complexity: O(n log n) typical time per reduced slice.
    """
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("interpolation must be 'linear', 'lower', 'higher', or 'midpoint'")
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1) | ~np.isfinite(q_arr)):
        raise ValueError("q must be between 0 and 1")
    arr = np.asarray(data, dtype=float)
    return np.nanquantile(arr, q_arr, axis=axis, method=interpolation, keepdims=keepdims)


def histogram(data, bins=10, range=None):
    """Compute a histogram over finite values; NaNs and infinities are excluded.
    
    Parameters
    ----------
    data : array-like
        Numeric observations.
    bins : int or sequence, default=10
        Number of equal-width bins or explicit bin edges.
    range : tuple[float, float], optional
        Lower and upper histogram range passed to ``numpy.histogram``.
    
    Returns
    -------
    (counts, edges) : tuple of numpy.ndarray
        Integer bin counts and bin edges of length ``len(counts) + 1``.
    
    Raises
    ------
    ValueError
        If no finite data remain or the bin specification is invalid.
    
    Complexity: O(n + b) time for n observations and b bins.
    """
    arr = np.asarray(data, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("histogram requires at least one finite value")
    if isinstance(bins, (int, np.integer)) and bins <= 0:
        raise ValueError("bins must be a positive integer")
    return np.histogram(valid, bins=bins, range=range)


def describe(data, axis=None):
    """Return a dictionary of common descriptive statistics for data.
    
    Parameters
    ----------
    data : array-like
        Numeric observations; NaNs are ignored in each statistic.
    axis : int or tuple of int, optional
        Reduction axis or axes; None computes statistics over the whole array.
    
    Returns
    -------
    dict[str, scalar or numpy.ndarray]
        Dictionary with count, mean, median, std, min, max, q25 and q75. Statistic
        values follow the selected axis's reduction shape.
    
    Raises
    ------
    ValueError
        If the input/axis cannot be reduced (e.g. empty min/max input).
    
    Complexity: O(n log n) dominated by median and quantiles.
    """
    arr = np.asarray(data, dtype=float)
    return {
        "count": np.sum(~np.isnan(arr), axis=axis),
        "mean": mean(arr, axis=axis),
        "median": median(arr, axis=axis),
        "std": std(arr, axis=axis),
        "min": minimum(arr, axis=axis),
        "max": maximum(arr, axis=axis),
        "q25": quantile(arr, 0.25, axis=axis),
        "q75": quantile(arr, 0.75, axis=axis),
    }


class WelfordAccumulator:
    """Incrementally compute per-feature mean and variance from data chunks.
    
    Parameters
    ----------
    None
        The constructor has no parameters.
    
    Returns
    -------
    WelfordAccumulator
        update returns self; mean, variance, std and result expose summaries.
    
    Raises
    ------
    RuntimeError
        If a summary is requested before any chunk is accumulated.
    ValueError
        If a chunk has invalid dimensionality, feature count changes, or ddof is negative.
    
    Complexity: O(n*d) time per chunk and O(d) persistent storage for d features.
    """
    def __init__(self):
        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        self.n_features_in_ = None

    def update(self, chunk):
        """Merge a new numeric chunk into this streaming accumulator.
        
        Parameters
        ----------
        chunk : array-like, shape (n_samples, n_features) or (n_samples,)
            New observations. NaNs are ignored independently per feature. Empty chunks
            are accepted and leave the current state unchanged.
        
        Returns
        -------
        self
            The updated accumulator, allowing chained calls.
        
        Raises
        ------
        ValueError
            If chunk dimensionality is invalid or feature count changes between updates.
        
        Complexity: O(n*d) time and O(d) persistent storage per d-feature chunk.
        """
        arr = np.asarray(chunk, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("chunk must be one- or two-dimensional")
        if arr.shape[0] == 0:
            return self
        if self.n_features_in_ is not None and arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        valid = ~np.isnan(arr)
        batch_count = valid.sum(axis=0).astype(np.int64)
        batch_sum = np.nansum(arr, axis=0)
        batch_mean = np.divide(batch_sum, batch_count, out=np.zeros(arr.shape[1], float), where=batch_count > 0)
        centered = np.where(valid, arr - batch_mean, 0.0)
        batch_M2 = np.sum(centered * centered, axis=0)

        if self.count_ is None:
            self.count_ = batch_count
            self.mean_ = batch_mean
            self.M2_ = batch_M2
            self.n_features_in_ = arr.shape[1]
            return self

        total = self.count_ + batch_count
        delta = batch_mean - self.mean_
        merged_mean = self.mean_ + np.divide(
            delta * batch_count, total, out=np.zeros_like(delta), where=total > 0
        )
        cross = np.divide(
            delta * delta * self.count_ * batch_count, total,
            out=np.zeros_like(delta), where=total > 0,
        )
        self.M2_ = self.M2_ + batch_M2 + cross
        self.mean_ = np.where(total > 0, merged_mean, 0.0)
        self.count_ = total
        return self

    @property
    def mean(self):
        """Return the accumulated per-feature mean as a defensive copy.
        
        Returns
        -------
        numpy.ndarray
            Shape (n_features,); all-missing features return NaN.
        
        Raises
        ------
        RuntimeError
            If no chunk has been accumulated.
        
        Complexity: O(d) time and output storage.
        """
        if self.count_ is None:
            raise RuntimeError("No data have been accumulated")
        return np.where(self.count_ > 0, self.mean_, np.nan).copy()

    def variance(self, ddof=0):
        """Return accumulated per-feature variance.
        
        Parameters
        ----------
        ddof : int, default=0
            Degrees-of-freedom adjustment; 0 gives population variance.
        
        Returns
        -------
        numpy.ndarray
            Shape (n_features,); slices with count <= ddof return NaN.
        
        Raises
        ------
        RuntimeError
            If no chunk has been accumulated.
        ValueError
            If ddof is negative.
        
        Complexity: O(d) time and output storage.
        """
        if self.count_ is None:
            raise RuntimeError("No data have been accumulated")
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        denom = self.count_ - ddof
        return np.divide(self.M2_, denom, out=np.full_like(self.M2_, np.nan), where=denom > 0)

    def std(self, ddof=0):
        """Return accumulated per-feature standard deviation.
        
        Parameters
        ----------
        ddof : int, default=0
            Degrees-of-freedom adjustment passed to ``variance``.
        
        Returns
        -------
        numpy.ndarray
            Square root of the variance values, shape (n_features,).
        
        Raises
        ------
        RuntimeError
            If no chunk has been accumulated.
        ValueError
            If ddof is negative.
        
        Complexity: O(d) time and output storage.
        """
        return np.sqrt(self.variance(ddof=ddof))

    def result(self):
        """Return a snapshot of the accumulator's count, mean, population variance, and std.
        
        Returns
        -------
        dict[str, numpy.ndarray]
            Copies of per-feature count, mean, variance, and standard deviation arrays.
        
        Raises
        ------
        RuntimeError
            If no chunk has been accumulated.
        
        Complexity: O(d) time and output storage.
        """
        if self.count_ is None:
            raise RuntimeError("No data have been accumulated")
        return {"count": self.count_.copy(), "mean": self.mean, "variance": self.variance(), "std": self.std()}

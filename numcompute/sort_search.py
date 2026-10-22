"""Sorting, partial selection, and searching algorithms using NumPy."""
from __future__ import annotations
import numpy as np


def stable_sort(values, axis=-1):
    """Return a stably sorted copy of an array along a selected axis.
    
    Parameters
    ----------
    values : array-like
        Input data.
    axis : int, default=-1
        Axis to sort; None flattens the input before sorting.
    
    Returns
    -------
    numpy.ndarray
        Sorted copy with the same shape as values.
    
    Raises
    ------
    ValueError
        If axis is out of bounds.
    
    Complexity: O(n log n) time per sorted slice and O(n) temporary/output storage.
    """
    arr = np.asarray(values)
    return np.sort(arr, axis=axis, kind="stable")


def multi_key_sort(X, keys, ascending=True, *, return_indices=False):
    """Sort the rows of a 2D array by one or more columns.
    
    Parameters
    ----------
    X : array-like, shape (n_rows, n_columns)
        Input rows.
    keys : sequence of int
        Column indices in priority order (first key is primary).
    ascending : bool or sequence of bool, default=True
        One direction for all keys or one direction per key.
    return_indices : bool, default=False
        If True, return row indices rather than sorted rows.
    
    Returns
    -------
    numpy.ndarray
        Sorted rows (same shape as X) or a 1D permutation of row indices.
    
    Raises
    ------
    ValueError
        If X is not 2D, keys are empty/out of bounds, or directions mismatch.
    TypeError
        If a descending key is non-numeric.
    
    Complexity: O(n log n) time and O(n) index storage.
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    keys = tuple(int(k) for k in keys)
    if not keys:
        raise ValueError("keys must contain at least one column index")
    if any(k < 0 or k >= arr.shape[1] for k in keys):
        raise ValueError("key column index out of range")
    if isinstance(ascending, (bool, np.bool_)):
        orders = [bool(ascending)] * len(keys)
    else:
        orders = list(ascending)
        if len(orders) != len(keys) or not all(isinstance(v, (bool, np.bool_)) for v in orders):
            raise ValueError("ascending must be a bool or a bool for each key")
    sort_keys = []
    for col, asc in zip(reversed(keys), reversed(orders)):
        values = arr[:, col]
        if asc:
            sort_keys.append(values)
        else:
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError("descending multi-key sort currently requires numeric columns")
            sort_keys.append(-values)
    idx = np.lexsort(tuple(sort_keys))
    return idx if return_indices else arr[idx]


def topk(values, k, largest=True, return_indices=True):
    """Select the k largest or smallest elements of a 1D array using partial sorting.
    
    Parameters
    ----------
    values : array-like, shape (n,)
        Input values.
    k : int
        Number of values to select, in the inclusive range 0..n.
    largest : bool, default=True
        Select largest values when True, otherwise smallest values.
    return_indices : bool, default=True
        Whether to return original indices alongside values.
    
    Returns
    -------
    values_out : numpy.ndarray
        Selected values ordered best-first (descending for largest, ascending for
        smallest).
    indices_out : numpy.ndarray, optional
        Original indices, returned only if return_indices is True.
    
    Raises
    ------
    TypeError
        If k is not an integer.
    ValueError
        If values is not 1D or k is outside 0..n.
    
    Complexity: O(n) expected partial selection plus O(k log k) selected-value sorting;
    output storage is O(k).
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 0 or k > arr.size:
        raise ValueError("k must satisfy 0 <= k <= len(values)")
    if k == 0:
        out_values = arr[:0].copy()
        out_indices = np.empty(0, dtype=int)
    elif k == arr.size:
        out_indices = np.arange(arr.size)
        order = np.argsort(arr, kind="stable")
        if largest:
            # For numeric arrays, sort descending stably by negating values.
            if np.issubdtype(arr.dtype, np.number):
                order = np.argsort(-arr, kind="stable")
            else:
                order = order[::-1]
        else:
            order = np.argsort(arr, kind="stable")
        out_indices = order
        out_values = arr[out_indices]
    else:
        if largest:
            part = np.argpartition(arr, arr.size - k)[arr.size - k:]
            local_order = np.argsort(-arr[part], kind="stable") if np.issubdtype(arr.dtype, np.number) else np.argsort(arr[part], kind="stable")[::-1]
        else:
            part = np.argpartition(arr, k - 1)[:k]
            local_order = np.argsort(arr[part], kind="stable")
        out_indices = part[local_order]
        out_values = arr[out_indices]
    if return_indices:
        return out_values, out_indices
    return out_values


def quickselect(values, k):
    """Return the kth-smallest value using a quickselect partition algorithm.
    
    Parameters
    ----------
    values : array-like, shape (n,)
        Input sequence, which is not modified.
    k : int
        Zero-based order statistic, with 0 selecting the minimum and n-1 the maximum.
    
    Returns
    -------
    scalar
        The kth-smallest value.
    
    Raises
    ------
    TypeError
        If k is not an integer.
    ValueError
        If values is not 1D, is empty, or k is out of range.
    
    Complexity: O(n) average time and O(n) copy storage; worst-case time O(n^2).
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError("quickselect cannot operate on an empty array")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 0 or k >= arr.size:
        raise ValueError("k must satisfy 0 <= k < len(values)")
    work = arr.copy()
    left, right = 0, work.size - 1
    while left <= right:
        pivot = work[(left + right) // 2]
        i, j = left, right
        while i <= j:
            while work[i] < pivot:
                i += 1
            while work[j] > pivot:
                j -= 1
            if i <= j:
                work[i], work[j] = work[j], work[i]
                i += 1
                j -= 1
        if k <= j:
            right = j
        elif k >= i:
            left = i
        else:
            return work[k]
    return work[k]


def binary_search(sorted_array, x):
    """Find the leftmost insertion point for x in an ascending sorted 1D array.
    
    Parameters
    ----------
    sorted_array : array-like, shape (n,)
        Values in nondecreasing order.
    x : scalar
        Target value to search for.
    
    Returns
    -------
    (index, exists) : tuple[int, bool]
        Leftmost insertion index and whether the target exactly exists.
    
    Raises
    ------
    ValueError
        If sorted_array is not 1D or is not sorted in ascending order.
    
    Complexity: O(n) to validate sortedness plus O(log n) search; O(1) extra space.
    """
    arr = np.asarray(sorted_array)
    if arr.ndim != 1:
        raise ValueError("sorted_array must be one-dimensional")
    if arr.size > 1 and np.any(arr[1:] < arr[:-1]):
        raise ValueError("sorted_array must be sorted in ascending order")
    idx = int(np.searchsorted(arr, x, side="left"))
    exists = idx < arr.size and bool(arr[idx] == x)
    return idx, exists

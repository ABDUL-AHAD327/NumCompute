"""CSV input helpers implemented with NumPy and the Python standard library."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterator, Sequence, Any
import numpy as np

_DEFAULT_MISSING = ("", "NA", "NaN", "nan", "null", "None")


def read_csv(
    path: str | Path,
    delimiter: str = ",",
    *,
    dtype: Any = str,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: Any = "",
    skip_header: int = 1,
    usecols: Sequence[int] | None = None,
    encoding: str = "utf-8",
    max_rows: int | None = None,
) -> np.ndarray:
    """Read a delimited text file into a NumPy array.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV/TSV file.
    delimiter : str, default=','
        One-character field separator, such as ',' or '\t'.
    dtype : data type, default=str
        NumPy dtype used for loaded values; use str for mixed data or float for numeric data.
    missing_values : sequence of str
        Tokens interpreted as missing by numpy.genfromtxt.
    filling_values : scalar
        Placeholder used when a field is missing. For numeric data, use numpy.nan.
    skip_header : int, default=1
        Number of leading header lines to skip.
    usecols : sequence of int, optional
        Column indices to read.
    encoding : str, default='utf-8'
        Text encoding.
    max_rows : int, optional
        Maximum number of data rows to read.
    
    Returns
    -------
    numpy.ndarray
        Two-dimensional array with shape (n_rows, n_selected_columns), including
        stable 2D shape for one-row and one-column files.
    
    Raises
    ------
    ValueError
        If delimiter or row limits are invalid, or the file contents cannot be parsed.
    OSError
        If the file cannot be opened.
    
    Complexity: O(n*d) time and O(n*d) memory for n rows and d selected columns.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")
    if skip_header < 0 or (max_rows is not None and max_rows < 0):
        raise ValueError("skip_header and max_rows must be non-negative")
    try:
        return np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            skip_header=skip_header,
            usecols=usecols,
            encoding=encoding,
            max_rows=max_rows,
            ndmin=2,
            autostrip=True,
            invalid_raise=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse delimited file {path!s}: {exc}") from exc


def read_numeric_csv(
    path: str | Path,
    delimiter: str = ",",
    *,
    skip_header: int = 1,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: float = np.nan,
    usecols: Sequence[int] | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    """Read a numeric CSV/TSV file as a float NumPy array.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    delimiter : str, default=','
        One-character field separator, such as a comma or tab.
    skip_header : int, default=1
        Number of leading header lines to skip.
    missing_values : sequence of str
        Text tokens interpreted as missing values.
    filling_values : float, default=numpy.nan
        Numeric placeholder used for missing fields.
    usecols : sequence of int, optional
        Optional columns to load.
    max_rows : int, optional
        Maximum number of data rows to return.
    
    Returns
    -------
    numpy.ndarray
        Float array with shape ``(n_rows, n_selected_columns)``; missing values
        are represented by the selected ``filling_values`` value.
    
    Raises
    ------
    ValueError
        If parsing arguments or file contents are invalid.
    OSError
        If the file cannot be opened.
    
    Complexity: O(n*d) time and memory for n rows and d selected columns.
    """
    return read_csv(
        path,
        delimiter,
        dtype=float,
        missing_values=missing_values,
        filling_values=filling_values,
        skip_header=skip_header,
        usecols=usecols,
        max_rows=max_rows,
    )


def iter_csv_chunks(
    path: str | Path,
    chunksize: int,
    delimiter: str = ",",
    *,
    dtype: Any = str,
    missing_values: Sequence[str] = _DEFAULT_MISSING,
    filling_values: Any = "",
    skip_header: int = 1,
    usecols: Sequence[int] | None = None,
    encoding: str = "utf-8",
) -> Iterator[np.ndarray]:
    """Yield a delimited text file as two-dimensional NumPy array chunks.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Input CSV/TSV path.
    chunksize : int
        Maximum number of data rows per yielded chunk; must be positive.
    delimiter : str, default=','
        One-character field separator.
    dtype : data-type, default=str
        NumPy dtype for loaded values.
    missing_values, filling_values : sequence and scalar
        Missing tokens and the replacement value passed to ``numpy.genfromtxt``.
    skip_header : int, default=1
        Number of header lines skipped once before chunking.
    usecols : sequence of int, optional
        Columns to load from each chunk.
    encoding : str, default='utf-8'
        Text encoding used to read the file.
    
    Yields
    ------
    numpy.ndarray
        A 2D chunk with at most ``chunksize`` rows and the selected columns.
    
    Raises
    ------
    ValueError
        If chunksize, delimiter, or skip_header is invalid, or a chunk is malformed.
    OSError
        If the file cannot be opened.
    
    Complexity: O(n*d) total time; auxiliary memory is O(chunksize*d).
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")
    if skip_header < 0:
        raise ValueError("skip_header must be non-negative")

    with open(path, "r", encoding=encoding, newline="") as handle:
        for _ in range(skip_header):
            next(handle, None)
        while True:
            lines = list(islice(handle, chunksize))
            if not lines:
                break
            chunk = np.genfromtxt(
                lines,
                delimiter=delimiter,
                dtype=dtype,
                missing_values=missing_values,
                filling_values=filling_values,
                usecols=usecols,
                encoding=encoding,
                ndmin=2,
                autostrip=True,
                invalid_raise=True,
            )
            if chunk.size:
                yield chunk

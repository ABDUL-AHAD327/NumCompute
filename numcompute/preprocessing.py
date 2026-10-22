"""NumPy-only data preprocessing transformers."""
from __future__ import annotations

import numpy as np


def _numeric_2d(X):
    arr = np.asarray(X, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a one- or two-dimensional array")
    if arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    return arr, was_1d


class StandardScaler:
    """Standardize numeric features with column-wise z-scores.
    
    Input and output are usually 2D with shape (n_samples, n_features). A 1D input
    is treated as a single feature and transformed back to 1D. NaNs are ignored when
    fitting and preserved during transform; constant columns map to zero for observed
    values.
    
    Parameters
    ----------
    None
        The constructor has no parameters.
    
    Returns
    -------
    StandardScaler
        A transformer whose fit method returns self.
    
    Raises
    ------
    ValueError
        If training input is empty/invalid or any feature consists entirely of NaNs.
    RuntimeError
        If transform is called before fit.
    
    Complexity: O(n*d) fit/transform time; fitted state uses O(d) storage.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn per-feature means and population standard deviations from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Numeric training data. NaNs are ignored independently per feature.
        
        Returns
        -------
        self
            The fitted transformer.
        
        Raises
        ------
        ValueError
            If X is empty, has invalid dimensionality, or a feature is all NaN.
        
        Complexity: O(n*d) time and O(d) fitted storage for n samples and d features.
        """
        arr, _ = _numeric_2d(X)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(arr, axis=0)
            scale = np.nanstd(arr, axis=0, ddof=0)
        if np.any(np.isnan(mean)):
            raise ValueError("Cannot fit StandardScaler: a feature contains only NaN values")
        scale = np.where((scale == 0) | np.isnan(scale), 1.0, scale)
        self.mean_, self.scale_ = mean, scale
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        """Standardize data using statistics learned by ``fit``.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Numeric input with the same feature count used during fitting.
        
        Returns
        -------
        numpy.ndarray
            Standardized values; 2D input returns shape (n_samples, n_features),
            while 1D input returns a 1D array. NaNs remain NaN.
        
        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        ValueError
            If X is empty, has invalid dimensionality, or has a different feature count.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        self._check_fitted()
        arr, was_1d = _numeric_2d(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        result = (arr - self.mean_) / self.scale_
        return result.reshape(-1) if was_1d else result

    def fit_transform(self, X):
        """Fit the scaler on X and return its standardized values.
        
        Parameters
        ----------
        X : array-like
            Numeric training data; NaNs are ignored while learning the statistics.
        
        Returns
        -------
        numpy.ndarray
            Standardized array with the same dimensionality convention as ``transform``.
        
        Raises
        ------
        ValueError
            If X is empty, invalid, or includes an all-NaN feature.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        return self.fit(X).transform(X)

    def _check_fitted(self):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")


class MinMaxScaler:
    """Scale numeric features to a configured interval, column by column.
    
    Parameters
    ----------
    feature_range : tuple of float, default=(0.0, 1.0)
        Requested output interval (low, high), with low < high.
    
    Returns
    -------
    MinMaxScaler
        A transformer whose fit method returns self.
    
    Raises
    ------
    ValueError
        If the range is invalid or any input feature has no finite training value.
    RuntimeError
        If transform is called before fit.
    
    Complexity: O(n*d) fit/transform time; fitted state uses O(d) storage.
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must contain (min, max)")
        low, high = map(float, feature_range)
        if not low < high:
            raise ValueError("feature_range must satisfy min < max")
        self.feature_range = (low, high)
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn per-feature minima and maxima from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Numeric training data; NaNs are ignored feature-wise.
        
        Returns
        -------
        self
            The fitted transformer.
        
        Raises
        ------
        ValueError
            If X is empty/invalid or any feature has no finite value.
        
        Complexity: O(n*d) time and O(d) fitted storage.
        """
        arr, _ = _numeric_2d(X)
        with np.errstate(invalid="ignore"):
            data_min = np.nanmin(arr, axis=0)
            data_max = np.nanmax(arr, axis=0)
        if np.any(~np.isfinite(data_min)) or np.any(~np.isfinite(data_max)):
            raise ValueError("Cannot fit MinMaxScaler: every feature must have at least one finite value")
        span = data_max - data_min
        scale = np.divide(
            self.feature_range[1] - self.feature_range[0], span,
            out=np.ones_like(span, dtype=float), where=span != 0,
        )
        self.data_min_, self.data_max_, self.scale_ = data_min, data_max, scale
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        """Scale X to the configured feature range using fitted minima and maxima.
        
        Parameters
        ----------
        X : array-like
            Numeric input with the same feature count as the training data.
        
        Returns
        -------
        numpy.ndarray
            Scaled array; constant columns map to the lower range bound and NaNs remain
            NaN. The output dimensionality follows the input dimensionality.
        
        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        ValueError
            If X is empty, invalid, or has a different feature count.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform")
        arr, was_1d = _numeric_2d(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        low = self.feature_range[0]
        result = (arr - self.data_min_) * self.scale_ + low
        constant = self.data_max_ == self.data_min_
        if np.any(constant):
            result[:, constant] = low
            result[np.isnan(arr)] = np.nan
        return result.reshape(-1) if was_1d else result

    def fit_transform(self, X):
        """Fit the scaler on X and return the scaled values.
        
        Parameters
        ----------
        X : array-like
            Numeric training data.
        
        Returns
        -------
        numpy.ndarray
            Scaled values using the same dimensionality convention as ``transform``.
        
        Raises
        ------
        ValueError
            If X is empty/invalid or any feature contains no finite value.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace missing numeric values using a constant or feature statistic.
    
    Parameters
    ----------
    strategy : {'constant', 'mean', 'median'}, default='constant'
        Replacement rule. Mean and median ignore NaNs feature-wise.
    fill_value : float, default=0.0
        Replacement value when strategy='constant'.
    
    Returns
    -------
    SimpleImputer
        A transformer whose fit method returns self.
    
    Raises
    ------
    ValueError
        If input is invalid or a mean/median feature has no observed values.
    RuntimeError
        If transform is called before fit.
    
    Complexity: O(n*d) fit/transform time; fitted replacement values use O(d) storage.
    """
    def __init__(self, strategy="constant", fill_value=0.0):
        if strategy not in {"constant", "mean", "median"}:
            raise ValueError("strategy must be 'constant', 'mean', or 'median'")
        self.strategy, self.fill_value = strategy, fill_value
        self.statistics_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Learn replacement values for missing entries in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Numeric data containing NaNs.
        
        Returns
        -------
        self
            The fitted imputer.
        
        Raises
        ------
        ValueError
            If X is empty/invalid or a mean/median feature has no observed values.
        
        Complexity: O(n*d) time and O(d) fitted storage.
        """
        arr, _ = _numeric_2d(X)
        if self.strategy == "constant":
            stats = np.full(arr.shape[1], float(self.fill_value))
        elif self.strategy == "mean":
            stats = np.nanmean(arr, axis=0)
        else:
            stats = np.nanmedian(arr, axis=0)
        if np.any(~np.isfinite(stats)):
            raise ValueError("Cannot impute a feature containing only missing values with this strategy")
        self.statistics_, self.n_features_in_ = stats, arr.shape[1]
        return self

    def transform(self, X):
        """Replace NaNs with the learned per-feature replacement values.
        
        Parameters
        ----------
        X : array-like
            Numeric input with the same feature count used during fitting.
        
        Returns
        -------
        numpy.ndarray
            Imputed values with the same shape convention as the input.
        
        Raises
        ------
        RuntimeError
            If the imputer has not been fitted.
        ValueError
            If X is empty/invalid or has a different feature count.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform")
        arr, was_1d = _numeric_2d(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        result = np.where(np.isnan(arr), self.statistics_, arr)
        return result.reshape(-1) if was_1d else result

    def fit_transform(self, X):
        """Fit the imputer and replace missing values in X.
        
        Parameters
        ----------
        X : array-like
            Numeric training data.
        
        Returns
        -------
        numpy.ndarray
            Imputed data with the same shape convention as the input.
        
        Raises
        ------
        ValueError
            If X is empty/invalid or a statistic-based feature is entirely missing.
        
        Complexity: O(n*d) time and O(n*d) output storage.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categorical columns as a dense one-hot NumPy matrix.
    
    Parameters
    ----------
    handle_unknown : {'ignore', 'error'}, default='ignore'
        Whether unknown categories become all-zero blocks or raise ValueError.
    categories : 'auto' or sequence of sequences, default='auto'
        Category lists for each input feature. Output categories are deterministic.
    
    Returns
    -------
    OneHotEncoder
        A transformer whose fit method returns self; transform produces shape
        (n_samples, total_number_of_fitted_categories).
    
    Raises
    ------
    ValueError
        If input or explicit category definitions are invalid, or an unknown value
        appears with handle_unknown='error'.
    RuntimeError
        If transform is called before fit.
    
    Complexity: O(n*d log(n*d)) for fitting categories; dense transform storage is
    O(n*c), where c is the total number of encoded categories.
    """
    def __init__(self, handle_unknown="ignore", categories="auto"):
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        if categories is None or (isinstance(categories, str) and categories != "auto"):
            raise ValueError("categories must be 'auto' or a sequence of category sequences")
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.categories_ = None
        self.n_features_in_ = None
        self.feature_offsets_ = None
        self._unique_pairs_ = None
        self._key_width_ = None

    @staticmethod
    def _as_2d(X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        return arr

    @staticmethod
    def _string_tokens(arr):
        tokens = arr.astype(str)
        width = max(1, int(np.max(np.char.str_len(tokens))))
        return tokens, width

    @staticmethod
    def _make_pair_keys(tokens, width):
        n_samples, n_features = tokens.shape
        dtype = np.dtype([("feature", np.int64), ("value", f"U{width}")])
        keys = np.empty(n_samples * n_features, dtype=dtype)
        keys["feature"] = np.tile(np.arange(n_features, dtype=np.int64), n_samples)
        keys["value"] = tokens.reshape(-1)
        return keys

    def fit(self, X):
        """Learn the categories for each categorical input feature.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Categorical training data. Values are converted to strings for category
            comparison; 1D inputs are treated as a single feature.
        
        Returns
        -------
        self
            The fitted encoder.
        
        Raises
        ------
        ValueError
            If X is empty/invalid or explicit categories are malformed or duplicated.
        
        Complexity: O(n*d log(n*d)) time for category discovery and O(c) fitted
        storage, where c is the total number of categories.
        """
        arr = self._as_2d(X)
        tokens, width = self._string_tokens(arr)
        n_samples, n_features = arr.shape

        if isinstance(self.categories, str) and self.categories == "auto":
            pairs = self._make_pair_keys(tokens, width)
            unique_pairs = np.unique(pairs)
        else:
            if len(self.categories) != n_features:
                raise ValueError("categories must provide one category sequence per feature")
            feature_parts = []
            for j, cats in enumerate(self.categories):
                values = np.asarray(cats)
                if values.ndim != 1 or values.size == 0:
                    raise ValueError(f"categories for feature {j} must be a non-empty 1D sequence")
                token_values = values.astype(str)
                if np.unique(token_values).size != token_values.size:
                    raise ValueError(f"categories for feature {j} contain duplicates")
                feature_parts.append(np.sort(token_values))
            explicit_width = max(width, max(int(np.max(np.char.str_len(c))) for c in feature_parts))
            dtype = np.dtype([("feature", np.int64), ("value", f"U{explicit_width}")])
            total = sum(c.size for c in feature_parts)
            pairs = np.empty(total, dtype=dtype)
            cursor = 0
            for j, cats in enumerate(feature_parts):
                end = cursor + cats.size
                pairs["feature"][cursor:end] = j
                pairs["value"][cursor:end] = cats
                cursor = end
            unique_pairs = np.unique(pairs)
            width = explicit_width

        counts = np.bincount(unique_pairs["feature"], minlength=n_features)
        if np.any(counts == 0):
            raise ValueError("every input feature must have at least one category")
        offsets = np.r_[0, np.cumsum(counts[:-1])].astype(int)
        self._unique_pairs_ = unique_pairs
        self._key_width_ = width
        self.categories_ = [unique_pairs["value"][unique_pairs["feature"] == j] for j in range(n_features)]
        self.feature_offsets_ = offsets
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Convert categorical values into a dense one-hot encoded feature matrix.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Input with the same number of features used during fitting.
        
        Returns
        -------
        numpy.ndarray
            Dense float matrix with shape (n_samples, total_fitted_categories). Unknown
            values produce zeros in their feature block when handle_unknown='ignore'.
        
        Raises
        ------
        RuntimeError
            If the encoder has not been fitted.
        ValueError
            If X is empty/invalid, has a different feature count, or contains unknown
            categories while handle_unknown='error'.
        
        Complexity: O(n*d log(c) + n*c) time and O(n*c) output storage, where c is the
        number of fitted categories.
        """
        if self._unique_pairs_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform")
        arr = self._as_2d(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        tokens, width = self._string_tokens(arr)
        width = max(width, self._key_width_)
        dtype = np.dtype([("feature", np.int64), ("value", f"U{width}")])
        fitted_pairs = self._unique_pairs_.astype(dtype, copy=False)
        input_pairs = self._make_pair_keys(tokens, width)
        positions = np.searchsorted(fitted_pairs, input_pairs)
        valid_positions = positions < fitted_pairs.size
        matched = np.zeros(positions.size, dtype=bool)
        matched[valid_positions] = fitted_pairs[positions[valid_positions]] == input_pairs[valid_positions]
        if self.handle_unknown == "error" and not np.all(matched):
            unknown = np.unique(tokens.reshape(-1)[~matched])
            raise ValueError(f"Unknown categories: {unknown!r}")
        n_samples = arr.shape[0]
        output = np.zeros((n_samples, fitted_pairs.size), dtype=float)
        rows = np.repeat(np.arange(n_samples), self.n_features_in_)
        output[rows[matched], positions[matched]] = 1.0
        return output

    def fit_transform(self, X):
        """Fit category vocabularies and encode X in one call.
        
        Parameters
        ----------
        X : array-like
            Categorical training data.
        
        Returns
        -------
        numpy.ndarray
            Dense one-hot matrix with one row per sample.
        
        Raises
        ------
        ValueError
            If X or explicit categories are invalid.
        
        Complexity: O(n*d log(n*d) + n*c) time and O(n*c) output storage.
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return output-column names matching the encoder's one-hot matrix.
        
        Parameters
        ----------
        input_features : sequence of str, optional
            Names for original input columns. Defaults to x0, x1, and so on.
        
        Returns
        -------
        numpy.ndarray of str
            One name per encoded output column.
        
        Raises
        ------
        RuntimeError
            If the encoder has not been fitted.
        ValueError
            If input_features is not 1D or its length differs from the input feature count.
        
        Complexity: O(c) time and output storage for c encoded categories.
        """
        if self._unique_pairs_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before requesting feature names")
        if input_features is None:
            input_features = np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=str)
        else:
            input_features = np.asarray(input_features, dtype=str)
        if input_features.ndim != 1 or input_features.size != self.n_features_in_:
            raise ValueError("input_features length must match number of input columns")
        names = np.char.add(
            np.char.add(input_features[self._unique_pairs_["feature"]], "_"),
            self._unique_pairs_["value"],
        )
        return names

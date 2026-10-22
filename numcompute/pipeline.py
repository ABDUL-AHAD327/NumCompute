"""Lightweight transformer, estimator, and pipeline abstractions."""
from __future__ import annotations
import numpy as np


class Transformer:
    """Base protocol for objects implementing fit, transform, and fit_transform.
    
    Subclass this class when a component learns state from training data and maps
    input arrays to transformed outputs. X is typically shaped (n_samples, n_features).
    
    Parameters
    ----------
    None
        The base constructor has no parameters.
    
    Returns
    -------
    Transformer
        The base fit method returns self; subclasses define transformed output shapes.
    
    Raises
    ------
    NotImplementedError
        The base transform method must be overridden.
    
    Complexity: Depends on the transformer implementation.
    """
    def fit(self, X, y=None):
        """Fit transformer state using X and return self.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like, optional
            Optional target; base implementation ignores it.
        
        Returns
        -------
        self
            The transformer instance.

        Raises
        ------
        None
            The base fit implementation performs no validation.

        Complexity: O(1) time and storage in the base implementation.
        """
        return self

    def transform(self, X):
        """Transform X using fitted state.
        
        Parameters
        ----------
        X : array-like
            Input features.
        
        Returns
        -------
        array-like
            Transformed features; subclasses must implement this method.
        
        Raises
        ------
        NotImplementedError
            Always in the abstract-style base implementation.

        Complexity: O(1) before raising the exception.
        """
        raise NotImplementedError("Transformer subclasses must implement transform")

    def fit_transform(self, X, y=None):
        """Fit the transformer on X and transform X.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like, optional
            Optional target forwarded to fit.
        
        Returns
        -------
        array-like
            Transformed training features.

        Raises
        ------
        NotImplementedError or transformer-specific exceptions
            If transform is not implemented or the input is rejected.

        Complexity: Depends on the concrete transformer.
        """
        return self.fit(X, y).transform(X)


class Estimator:
    """Minimal base protocol for objects implementing fit and predict.
    
    X normally has shape (n_samples, n_features); target and prediction shapes depend
    on the estimator's task.
    
    Parameters
    ----------
    None
        The base constructor has no parameters.
    
    Returns
    -------
    Estimator
        Concrete fit methods should return self; predict returns task-specific outputs.
    
    Raises
    ------
    NotImplementedError
        Base fit and predict must be overridden.
    
    Complexity: Depends on the estimator implementation.
    """
    def fit(self, X, y):
        """Fit estimator parameters from X and y.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target labels or values.
        
        Returns
        -------
        self
            Subclasses should return the fitted estimator.
        
        Raises
        ------
        NotImplementedError
            Always in the base implementation.

        Complexity: O(1) before raising the exception.
        """
        raise NotImplementedError("Estimator subclasses must implement fit")

    def predict(self, X):
        """Predict targets for X using fitted estimator state.
        
        Parameters
        ----------
        X : array-like
            Input features.
        
        Returns
        -------
        array-like
            Predicted labels or numeric values.
        
        Raises
        ------
        NotImplementedError
            Always in the base implementation.

        Complexity: O(1) before raising the exception.
        """
        raise NotImplementedError("Estimator subclasses must implement predict")


def _check_steps(steps):
    steps = list(steps)
    if not steps:
        raise ValueError("steps must contain at least one (name, object) pair")
    names = []
    for step in steps:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            raise ValueError("each step must be a (name, object) pair")
        name, obj = step
        if not isinstance(name, str) or not name:
            raise ValueError("step names must be non-empty strings")
        if obj is None:
            raise ValueError(f"step {name!r} cannot be None")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("step names must be unique")
    return [(name, obj) for name, obj in steps]


class Pipeline:
    """Chain named transformations and optionally a final estimator.
    
    Parameters
    ----------
    steps : sequence of (str, object)
        Non-empty sequence of uniquely named transformer/estimator steps. Intermediate
        steps need fit and transform; the final step needs fit and either transform or predict.
    
    Returns
    -------
    Pipeline
        A chain object. fit returns self; transform/fit_transform return transformed arrays;
        predict returns the final estimator's predictions.
    
    Raises
    ------
    ValueError
        For empty/duplicate/invalid steps or when a final estimator is fitted without y.
    TypeError
        If a step does not implement the required methods.
    
    Complexity: Sum of each step's runtime; intermediate arrays may be held in memory.
    """
    def __init__(self, steps):
        self.steps = _check_steps(steps)
        self.named_steps = dict(self.steps)

    def fit(self, X, y=None):
        """Fit each intermediate transformer and then the final pipeline step.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like, optional
            Required when the final step exposes ``predict``; passed only to that estimator.
        
        Returns
        -------
        self
            Fitted pipeline.
        
        Raises
        ------
        TypeError
            If steps do not implement the required methods.
        ValueError
            If an estimator final step is used without y.
        
        Complexity: Sum of the fitted steps' complexities; intermediate outputs are held in memory.
        """
        current = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"Intermediate step {name!r} must implement fit and transform")
            step.fit(current)
            current = step.transform(current)
        final = self.steps[-1][1]
        if not hasattr(final, "fit"):
            raise TypeError("Final pipeline step must implement fit")
        if hasattr(final, "predict"):
            if y is None:
                raise ValueError("y is required when the final pipeline step is an estimator")
            final.fit(current, y)
        elif hasattr(final, "transform"):
            final.fit(current)
        else:
            raise TypeError("Final step must implement transform or predict")
        return self

    def transform(self, X):
        """Apply each pipeline step's transform method in order.
        
        Parameters
        ----------
        X : array-like
            Input data.
        
        Returns
        -------
        array-like
            Data after all pipeline transformations.
        
        Raises
        ------
        TypeError
            If any step lacks a transform method.
        
        Complexity: Sum of each transformer's runtime.
        """
        current = X
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step {name!r} does not implement transform")
            current = step.transform(current)
        return current

    def fit_transform(self, X, y=None):
        """Fit and transform every step in a transformer-only pipeline.
        
        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Accepted for API compatibility; transformations are fit without targets.
        
        Returns
        -------
        array-like
            Transformed training data.
        
        Raises
        ------
        TypeError
            If any step lacks fit or transform.
        
        Complexity: Sum of all step fit/transform runtimes.
        """
        current = X
        for name, step in self.steps:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(f"Step {name!r} must implement fit and transform for fit_transform")
            step.fit(current)
            current = step.transform(current)
        return current

    def predict(self, X):
        """Transform X through intermediate steps and call predict on the final estimator.
        
        Parameters
        ----------
        X : array-like
            Data to predict.
        
        Returns
        -------
        array-like
            Predictions returned by the final estimator.
        
        Raises
        ------
        TypeError
            If an intermediate step lacks transform or the final step lacks predict.
        
        Complexity: Sum of intermediate transform times and final predict time.
        """
        current = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"Intermediate step {name!r} does not implement transform")
            current = step.transform(current)
        final = self.steps[-1][1]
        if not hasattr(final, "predict"):
            raise TypeError("The final pipeline step does not implement predict")
        return final.predict(current)


class Compose(Transformer):
    """Compose a sequence of transformers into a single sequential transformer.
    
    Parameters
    ----------
    transformers : sequence of transformer-like objects
        Non-empty sequence; each object must support fit and transform.
    
    Returns
    -------
    Compose
        A composition whose output has the shape returned by the last transformer.
    
    Raises
    ------
    ValueError
        If the sequence is empty.
    TypeError
        If a member lacks fit/transform.
    
    Complexity: Sum of all component fit/transform runtimes.
    """
    def __init__(self, transformers):
        self.transformers = list(transformers)
        if not self.transformers:
            raise ValueError("transformers must not be empty")

    def fit(self, X, y=None):
        """Fit each transformer sequentially, passing each output to the next fit call.
        
        Parameters
        ----------
        X : array-like
            Training input.
        y : array-like, optional
            Accepted for API compatibility; individual transformer fits receive X only.
        
        Returns
        -------
        self
            Fitted composition.
        
        Raises
        ------
        TypeError
            If a member does not implement fit and transform.
        Complexity: Sum of each transformer fit/transform cost over the sequential data flow.
        """
        current = X
        for transformer in self.transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("Every Compose item must implement fit and transform")
            transformer.fit(current)
            current = transformer.transform(current)
        return self

    def transform(self, X):
        """Apply the fitted transformers sequentially to X.
        
        Parameters
        ----------
        X : array-like
            Input features.
        
        Returns
        -------
        array-like
            Output of the last transformer.
        
        Raises
        ------
        AttributeError or transformer-specific errors
            If a transformer cannot transform the supplied data.
        Complexity: Sum of transformer transform costs; intermediate arrays are reused sequentially.
        """
        current = X
        for transformer in self.transformers:
            current = transformer.transform(current)
        return current

    def fit_transform(self, X, y=None):
        """Fit and apply each transformer in sequence to X.
        
        Parameters
        ----------
        X : array-like
            Training input.
        y : array-like, optional
            Accepted for API compatibility.
        
        Returns
        -------
        array-like
            Fully transformed training input.
        
        Raises
        ------
        Transformer-specific errors
            If a transformer cannot fit or transform the current data.
        Complexity: Sum of all transformer fit/transform costs over the sequential data flow.
        """
        current = X
        for transformer in self.transformers:
            transformer.fit(current)
            current = transformer.transform(current)
        return current


class FeatureUnion(Transformer):
    """Fit branches in parallel on the same input and concatenate their output columns.
    
    Parameters
    ----------
    transformer_list : sequence of (str, transformer)
        Non-empty named transformer branches. Each output must be one- or two-dimensional
        and contain the same number of samples.
    
    Returns
    -------
    FeatureUnion
        A transformer producing a 2D matrix with all branch feature columns concatenated.
    
    Raises
    ------
    ValueError
        If branch outputs have incompatible dimensionality or sample counts.
    TypeError
        If a branch lacks fit/transform.
    
    Complexity: Sum of branch runtimes plus O(n*c) concatenation storage for n rows and c output columns.
    """
    def __init__(self, transformer_list):
        self.transformer_list = _check_steps(transformer_list)

    def fit(self, X, y=None):
        """Fit every branch transformer on the same input X.
        
        Parameters
        ----------
        X : array-like
            Shared input features.
        y : array-like, optional
            Accepted for API compatibility.
        
        Returns
        -------
        self
            Fitted feature union.
        
        Raises
        ------
        TypeError
            If a branch lacks fit or transform.
        Complexity: Sum of the fit costs of all branch transformers.
        """
        for name, transformer in self.transformer_list:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"FeatureUnion item {name!r} must implement fit and transform")
            transformer.fit(X)
        return self

    def transform(self, X):
        """Transform X through each branch and concatenate branch outputs by columns.
        
        Parameters
        ----------
        X : array-like
            Shared input features.
        
        Returns
        -------
        numpy.ndarray, shape (n_samples, sum(output_features))
            Concatenated 2D feature matrix.
        
        Raises
        ------
        ValueError
            If branch outputs are not 1D/2D or have inconsistent row counts.
        
        Complexity: Sum of branch transform costs plus concatenation/output storage.
        """
        outputs = []
        n_rows = None
        for name, transformer in self.transformer_list:
            out = np.asarray(transformer.transform(X))
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            if out.ndim != 2:
                raise ValueError(f"FeatureUnion item {name!r} must return 1D or 2D data")
            if n_rows is None:
                n_rows = out.shape[0]
            elif out.shape[0] != n_rows:
                raise ValueError("All FeatureUnion outputs must have the same number of rows")
            outputs.append(out)
        return np.concatenate(outputs, axis=1)

    def fit_transform(self, X, y=None):
        """Fit all branches on X and return their concatenated transformed outputs.
        
        Parameters
        ----------
        X : array-like
            Shared input features.
        y : array-like, optional
            Accepted for API compatibility.
        
        Returns
        -------
        numpy.ndarray
            2D matrix formed by concatenating each branch's output columns.
        
        Raises
        ------
        TypeError or ValueError
            If branch interfaces or output shapes are invalid.
        Complexity: Sum of branch fit/transform costs plus O(n*c) concatenation storage.
        """
        self.fit(X, y)
        return self.transform(X)


class ColumnSelector(Transformer):
    """Select specified columns from a 2D feature array.
    
    Parameters
    ----------
    columns : sequence of int
        Non-empty sequence of column indices, in desired output order.
    
    Returns
    -------
    ColumnSelector
        A transformer returning shape (n_samples, len(columns)).
    
    Raises
    ------
    ValueError
        If columns are empty, not 1D, out of bounds, or input is not 2D.
    RuntimeError
        If transform is called before fit.
    
    Complexity: O(n*k) output construction for n rows and k selected columns.
    """
    def __init__(self, columns):
        self.columns = np.asarray(columns, dtype=int)
        if self.columns.ndim != 1 or self.columns.size == 0:
            raise ValueError("columns must be a non-empty one-dimensional sequence")

    def fit(self, X, y=None):
        """Validate X and record its feature count for later transforms.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            2D input matrix.
        y : array-like, optional
            Ignored; present for transformer API compatibility.
        
        Returns
        -------
        self
            Fitted selector.
        
        Raises
        ------
        ValueError
            If X is not 2D or a selected column index is out of range.
        Complexity: O(k) time to validate k selected columns and O(1) additional storage.
        """
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if np.any(self.columns < 0) or np.any(self.columns >= arr.shape[1]):
            raise ValueError("column index out of range")
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        """Return only the configured columns from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input matrix with the same feature count observed during fit.
        
        Returns
        -------
        numpy.ndarray, shape (n_samples, len(columns))
            Selected columns, preserving the configured column order.
        
        Raises
        ------
        RuntimeError
            If the selector has not been fitted.
        ValueError
            If X is not 2D or its feature count differs from fit data.
        
        Complexity: O(n*k) output construction for n rows and k selected columns.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("ColumnSelector must be fitted before transform")
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.shape[1] != self.n_features_in_:
            raise ValueError("X has incompatible shape")
        return arr[:, self.columns]

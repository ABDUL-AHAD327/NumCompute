"""NumCompute: a NumPy-based scientific computing toolkit for Programming for AI."""
from .io import read_csv, read_numeric_csv, iter_csv_chunks
from .preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from .sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search
from .rank import rank, percentile
from .stats import mean, median, std, variance, minimum, maximum, quantile, histogram, describe, WelfordAccumulator
from .metrics import accuracy, precision, recall, f1, confusion_matrix, mse, roc_curve, auc
from .optim import grad, jacobian
from .pipeline import Transformer, Estimator, Pipeline, Compose, FeatureUnion, ColumnSelector
from .utils import relu, sigmoid, tanh, softmax, logsumexp, euclidean_distance, manhattan_distance, pairwise_distances, iter_batches

__version__ = "0.1.0"

__all__ = [
    "read_csv", "read_numeric_csv", "iter_csv_chunks",
    "StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer",
    "stable_sort", "multi_key_sort", "topk", "quickselect", "binary_search",
    "rank", "percentile", "mean", "median", "std", "variance", "minimum", "maximum", "quantile", "histogram", "describe", "WelfordAccumulator",
    "accuracy", "precision", "recall", "f1", "confusion_matrix", "mse", "roc_curve", "auc",
    "grad", "jacobian", "Transformer", "Estimator", "Pipeline", "Compose", "FeatureUnion", "ColumnSelector",
    "relu", "sigmoid", "tanh", "softmax", "logsumexp", "euclidean_distance", "manhattan_distance", "pairwise_distances", "iter_batches",
]

"""Additional edge-case tests required by the NumCompute assignment.

Run all tests from the project root with:
    python -m unittest discover -s tests -v
"""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from numcompute.io import read_csv, read_numeric_csv, iter_csv_chunks
from numcompute.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search
from numcompute.rank import rank, percentile
from numcompute.stats import mean, median, std, variance, minimum, maximum, quantile, histogram, WelfordAccumulator
from numcompute.metrics import accuracy, precision, recall, f1, confusion_matrix, mse, roc_curve, auc
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline, ColumnSelector, FeatureUnion
from numcompute.utils import softmax, logsumexp, pairwise_distances, iter_batches


class TestIOEdgeCases(unittest.TestCase):
    def test_tab_delimiter_and_single_row_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tab.tsv"
            path.write_text("name\tvalue\nA\t3\n", encoding="utf-8")
            data = read_csv(path, delimiter="\t", dtype=str)
        self.assertEqual(data.shape, (1, 2))
        self.assertEqual(data[0, 1], "3")

    def test_invalid_delimiter_is_rejected(self):
        with self.assertRaises(ValueError):
            read_csv(Path("missing.csv"), delimiter="||")

    def test_numeric_missing_values_are_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "numeric.csv"
            path.write_text("a,b\n1,\n2,4\n", encoding="utf-8")
            data = read_numeric_csv(path)
        self.assertEqual(data.shape, (2, 2))
        self.assertTrue(np.isnan(data[0, 1]))

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            list(iter_csv_chunks("missing.csv", 0))


class TestPreprocessingEdgeCases(unittest.TestCase):
    def test_scaler_preserves_nan_and_scales_other_values(self):
        X = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]])
        out = StandardScaler().fit_transform(X)
        self.assertTrue(np.isnan(out[0, 1]))
        np.testing.assert_allclose(out[:, 0], [-1.224744871, 0, 1.224744871], atol=1e-8)

    def test_standard_scaler_rejects_all_nan_feature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                StandardScaler().fit([[np.nan], [np.nan]])

    def test_minmax_rejects_all_nan_feature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                MinMaxScaler().fit([[np.nan], [np.nan]])

    def test_minmax_constant_column_and_nan(self):
        out = MinMaxScaler(feature_range=(-2.0, 2.0)).fit_transform([[4.0], [np.nan], [4.0]])
        np.testing.assert_allclose(out[[0, 2], 0], [-2.0, -2.0])
        self.assertTrue(np.isnan(out[1, 0]))

    def test_mean_imputer_uses_training_column_means(self):
        imputer = SimpleImputer(strategy="mean").fit([[1.0, 10.0], [np.nan, 30.0]])
        out = imputer.transform([[np.nan, 20.0]])
        np.testing.assert_allclose(out, [[1.0, 20.0]])

    def test_statistical_imputer_rejects_all_missing_feature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                SimpleImputer(strategy="median").fit([[np.nan], [np.nan]])

    def test_onehot_unknown_error_policy(self):
        enc = OneHotEncoder(handle_unknown="error").fit([["red"], ["blue"]])
        with self.assertRaises(ValueError):
            enc.transform([["green"]])

    def test_onehot_rejects_wrong_feature_count(self):
        enc = OneHotEncoder().fit([["red", "S"], ["blue", "M"]])
        with self.assertRaises(ValueError):
            enc.transform([["red"]])

    def test_onehot_feature_names_match_output_width(self):
        enc = OneHotEncoder().fit([["cat", "S"], ["dog", "M"], ["cat", "M"]])
        names = enc.get_feature_names_out(["animal", "size"])
        out = enc.transform([["cat", "S"], ["dog", "M"]])
        self.assertEqual(names.size, out.shape[1])
        self.assertEqual(names.ndim, 1)


class TestSortAndRankEdgeCases(unittest.TestCase):
    def test_topk_smallest_and_values_only(self):
        np.testing.assert_array_equal(topk([5, 1, 7, 4], 2, largest=False, return_indices=False), [1, 4])

    def test_topk_k_equals_length(self):
        values, indices = topk([3, 1, 2], 3)
        np.testing.assert_array_equal(values, [3, 2, 1])
        np.testing.assert_array_equal(indices, [0, 2, 1])

    def test_topk_rejects_boolean_k(self):
        with self.assertRaises(TypeError):
            topk([1, 2], True)

    def test_quickselect_rejects_out_of_range_k(self):
        with self.assertRaises(ValueError):
            quickselect([1, 2, 3], 3)

    def test_multikey_descending(self):
        X = np.array([[1, 3], [2, 1], [1, 2]])
        out = multi_key_sort(X, [0, 1], ascending=[False, True])
        np.testing.assert_array_equal(out, [[2, 1], [1, 2], [1, 3]])

    def test_rank_empty_and_invalid_method(self):
        self.assertEqual(rank([]).size, 0)
        with self.assertRaises(ValueError):
            rank([1, 2], method="unknown")

    def test_percentile_vector_q_shape(self):
        out = percentile([1, 2, 3, 4], [0, 50, 100])
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, [1, 2.5, 4])

    def test_percentile_rejects_q_outside_range(self):
        with self.assertRaises(ValueError):
            percentile([1, 2, 3], 101)


class TestStatisticsEdgeCases(unittest.TestCase):
    def test_axis_keepdims(self):
        out = mean([[1.0, 2.0], [3.0, 4.0]], axis=0, keepdims=True)
        self.assertEqual(out.shape, (1, 2))
        np.testing.assert_allclose(out, [[2.0, 3.0]])

    def test_quantile_rejects_probability_outside_zero_one(self):
        with self.assertRaises(ValueError):
            quantile([1, 2, 3], 1.1)

    def test_histogram_rejects_no_finite_values(self):
        with self.assertRaises(ValueError):
            histogram([np.nan, np.inf, -np.inf])

    def test_welford_methods_require_data(self):
        acc = WelfordAccumulator()
        with self.assertRaises(RuntimeError):
            _ = acc.mean
        with self.assertRaises(RuntimeError):
            acc.variance()

    def test_welford_rejects_feature_count_change(self):
        acc = WelfordAccumulator().update([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            acc.update([[1.0, 2.0, 3.0]])

    def test_non_contiguous_statistics_and_reductions(self):
        view = np.arange(24, dtype=float).reshape(4, 6)[:, ::2]
        self.assertFalse(view.flags.c_contiguous)
        np.testing.assert_allclose(mean(view, axis=0), [9.0, 11.0, 13.0])
        np.testing.assert_allclose(minimum(view, axis=0), [0.0, 2.0, 4.0])
        np.testing.assert_allclose(maximum(view, axis=0), [18.0, 20.0, 22.0])
        self.assertTrue(np.all(variance(view, axis=0) >= 0))
        self.assertTrue(np.all(std(view, axis=0) >= 0))
        self.assertTrue(np.isfinite(median(view)))


class TestMetricsEdgeCases(unittest.TestCase):
    def test_accuracy_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            accuracy([1, 0], [1])

    def test_zero_denominator_policy(self):
        self.assertEqual(precision([0, 0], [0, 0], pos_label=1, zero_division=0.25), 0.25)
        self.assertEqual(recall([0, 0], [0, 0], pos_label=1, zero_division=0.5), 0.5)
        self.assertEqual(f1([0, 0], [0, 0], pos_label=1, zero_division=0), 0.0)

    def test_confusion_matrix_rejects_unlisted_label(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0, 1], labels=[0])

    def test_mse_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            mse([], [])

    def test_roc_requires_both_classes(self):
        with self.assertRaises(ValueError):
            roc_curve([1, 1, 1], [0.2, 0.5, 0.8])

    def test_auc_requires_two_points(self):
        with self.assertRaises(ValueError):
            auc([0.0], [1.0])


class TestOptimUtilsAndPipelineEdgeCases(unittest.TestCase):
    def test_forward_gradient(self):
        self.assertAlmostEqual(grad(lambda x: x * x, 3.0, method="forward"), 6.0, places=4)

    def test_gradient_rejects_invalid_step(self):
        with self.assertRaises(ValueError):
            grad(lambda x: x * x, 2.0, h=0)

    def test_jacobian_output_shape(self):
        J = jacobian(lambda x: np.array([x[0] + x[1], x[0] * x[1], x[0] ** 2]), [2.0, 3.0])
        self.assertEqual(J.shape, (3, 2))
        np.testing.assert_allclose(J, [[1, 1], [3, 2], [4, 0]], atol=1e-4)

    def test_softmax_of_all_negative_infinity_is_undefined(self):
        result = softmax([-np.inf, -np.inf])
        self.assertTrue(np.all(np.isnan(result)))

    def test_logsumexp_infinite_extremes(self):
        self.assertTrue(np.isposinf(logsumexp([np.inf, 3.0])))
        self.assertTrue(np.isneginf(logsumexp([-np.inf, -np.inf])))

    def test_manhattan_pairwise_distance(self):
        out = pairwise_distances([[0, 0], [1, 2]], [[3, 0]], metric="manhattan")
        np.testing.assert_allclose(out, [[3], [4]])

    def test_pairwise_distances_reject_feature_mismatch(self):
        with self.assertRaises(ValueError):
            pairwise_distances([[1, 2]], [[1, 2, 3]])

    def test_batches_shuffle_is_reproducible_and_drop_last(self):
        X = np.arange(11)
        run1 = list(iter_batches(X, 4, shuffle=True, seed=7, drop_last=True))
        run2 = list(iter_batches(X, 4, shuffle=True, seed=7, drop_last=True))
        self.assertEqual([b.tolist() for b in run1], [b.tolist() for b in run2])
        self.assertEqual(sum(len(b) for b in run1), 8)

    def test_batches_reject_nonpositive_size(self):
        with self.assertRaises(ValueError):
            list(iter_batches(np.arange(3), -1))

    def test_column_selector_requires_fit(self):
        selector = ColumnSelector([0])
        with self.assertRaises(RuntimeError):
            selector.transform([[1.0, 2.0]])

    def test_pipeline_rejects_empty_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_feature_union_requires_matching_rows(self):
        class First:
            def fit(self, X): return self
            def transform(self, X): return np.asarray(X)[:2, :1]
        class Second:
            def fit(self, X): return self
            def transform(self, X): return np.asarray(X)[:1, :1]
        union = FeatureUnion([("a", First()), ("b", Second())])
        union.fit(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            union.transform(np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()

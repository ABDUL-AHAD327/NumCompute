"""Unit tests for the core NumCompute modules. Run with: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from numcompute.io import read_csv, read_numeric_csv, iter_csv_chunks
from numcompute.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search
from numcompute.rank import rank, percentile
from numcompute.stats import mean, median, std, variance, histogram, quantile, WelfordAccumulator
from numcompute.metrics import accuracy, precision, recall, f1, confusion_matrix, mse, roc_curve, auc
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline, ColumnSelector, FeatureUnion
from numcompute.utils import sigmoid, softmax, logsumexp, pairwise_distances, iter_batches


class TestIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.csv"
        self.path.write_text("name,value\nA,1\nB,\nC,3\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_mixed_csv(self):
        a = read_csv(self.path, dtype=str)
        self.assertEqual(a.shape, (3, 2))
        self.assertEqual(a[0, 0], "A")

    def test_read_numeric_missing(self):
        a = read_numeric_csv(self.path, usecols=[1])
        self.assertEqual(a.shape, (3, 1))
        self.assertTrue(np.isnan(a[1, 0]))

    def test_chunk_reader(self):
        chunks = list(iter_csv_chunks(self.path, 2, dtype=str))
        self.assertEqual([c.shape[0] for c in chunks], [2, 1])


class TestPreprocessing(unittest.TestCase):
    def test_standard_scaler_mean_zero(self):
        x = np.array([[1., 10.], [3., 30.], [5., 50.]])
        out = StandardScaler().fit_transform(x)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)

    def test_standard_scaler_constant_column(self):
        out = StandardScaler().fit_transform([[2., 5.], [2., 7.]])
        np.testing.assert_allclose(out[:, 0], 0)

    def test_scaler_rejects_wrong_feature_count(self):
        scaler = StandardScaler().fit([[1., 2.]])
        with self.assertRaises(ValueError):
            scaler.transform([[1., 2., 3.]])

    def test_minmax_range(self):
        out = MinMaxScaler((-1, 1)).fit_transform([[0.], [5.], [10.]])
        np.testing.assert_allclose(out[:, 0], [-1, 0, 1])

    def test_minmax_constant(self):
        out = MinMaxScaler().fit_transform([[7.], [7.]])
        np.testing.assert_allclose(out, 0)

    def test_imputer_constant(self):
        out = SimpleImputer(fill_value=-1).fit_transform([[1.], [np.nan]])
        np.testing.assert_allclose(out[:, 0], [1, -1])

    def test_onehot(self):
        out = OneHotEncoder().fit_transform(np.array([["A"], ["B"], ["A"]]))
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(out.sum(axis=1), 1)

    def test_onehot_unknown_ignored(self):
        enc = OneHotEncoder().fit([["A"], ["B"]])
        self.assertEqual(enc.transform([["C"]]).sum(), 0)

    def test_onehot_explicit_unsorted_categories(self):
        enc = OneHotEncoder(categories=[["B", "A"]]).fit([["A"], ["B"]])
        out = enc.transform([["A"], ["B"]])
        np.testing.assert_array_equal(out, [[1, 0], [0, 1]])


class TestSortSearch(unittest.TestCase):
    def test_stable_sort(self):
        np.testing.assert_array_equal(stable_sort([3, 1, 2]), [1, 2, 3])

    def test_multi_key_sort(self):
        x = np.array([[1, 2], [0, 3], [1, 1]])
        np.testing.assert_array_equal(multi_key_sort(x, [0, 1]), [[0, 3], [1, 1], [1, 2]])

    def test_topk_values_and_indices(self):
        values, indices = topk([5, 1, 7, 4], 2)
        np.testing.assert_array_equal(values, [7, 5])
        np.testing.assert_array_equal(indices, [2, 0])

    def test_topk_zero(self):
        values, indices = topk([1, 2], 0)
        self.assertEqual(values.size, 0)
        self.assertEqual(indices.size, 0)

    def test_topk_invalid_k(self):
        with self.assertRaises(ValueError):
            topk([1, 2], 3)

    def test_quickselect(self):
        self.assertEqual(quickselect([9, 2, 4, 1, 7], 2), 4)

    def test_binary_search_found_and_missing(self):
        self.assertEqual(binary_search([1, 2, 2, 4], 2), (1, True))
        self.assertEqual(binary_search([1, 2, 4], 3), (2, False))

    def test_binary_search_requires_sorted(self):
        with self.assertRaises(ValueError):
            binary_search([2, 1], 1)

    def test_binary_search_empty(self):
        self.assertEqual(binary_search([], 9), (0, False))

    def test_topk_all_equal(self):
        values, indices = topk([4, 4, 4, 4], 3)
        np.testing.assert_array_equal(values, [4, 4, 4])
        self.assertEqual(len(indices), 3)

    def test_non_contiguous_sort_and_topk(self):
        base = np.arange(20)
        view = base[::2]
        self.assertFalse(view.flags.c_contiguous)
        np.testing.assert_array_equal(stable_sort(view), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        values, _ = topk(view, 2)
        np.testing.assert_array_equal(values, [18, 16])


class TestRankAndStats(unittest.TestCase):
    def test_average_rank_ties(self):
        np.testing.assert_allclose(rank([10, 20, 20, 30]), [1, 2.5, 2.5, 4])

    def test_dense_rank(self):
        np.testing.assert_allclose(rank([10, 20, 20, 30], "dense"), [1, 2, 2, 3])

    def test_ordinal_rank(self):
        np.testing.assert_allclose(rank([20, 20, 10], "ordinal"), [2, 3, 1])

    def test_rank_nan_preserved(self):
        result = rank([3.0, np.nan, 1.0])
        np.testing.assert_allclose(result[[0, 2]], [2, 1])
        self.assertTrue(np.isnan(result[1]))

    def test_statistics_non_contiguous_input(self):
        view = np.arange(20, dtype=float)[::2]
        self.assertFalse(view.flags.c_contiguous)
        self.assertEqual(float(mean(view)), 9.0)

    def test_percentile_lower(self):
        self.assertEqual(percentile([1, 3, 5, 7], 50, "lower"), 3)

    def test_stats_axis(self):
        np.testing.assert_allclose(mean([[1, 2], [3, 4]], axis=0), [2, 3])
        self.assertEqual(median([1, 3, 5]), 3)
        self.assertAlmostEqual(float(std([1, 3, 5])), np.sqrt(8 / 3))

    def test_quantile_ignores_nan(self):
        self.assertEqual(float(quantile([1, np.nan, 3], 0.5)), 2)

    def test_histogram_ignores_nan(self):
        counts, edges = histogram([1, 2, np.nan, 3], bins=3)
        self.assertEqual(counts.sum(), 3)
        self.assertEqual(edges.size, 4)

    def test_welford_across_chunks(self):
        acc = WelfordAccumulator().update([[1., 10.], [2., 20.]])
        acc.update([[3., 30.]])
        np.testing.assert_allclose(acc.mean, [2., 20.])
        np.testing.assert_allclose(acc.variance(), [2 / 3, 200 / 3])


class TestMetricsOptimAndUtils(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(accuracy([1, 0, 1], [1, 1, 1]), 2 / 3)

    def test_binary_metrics(self):
        self.assertAlmostEqual(precision(["Yes", "No", "Yes"], ["Yes", "Yes", "No"], pos_label="Yes"), 0.5)
        self.assertAlmostEqual(recall(["Yes", "No", "Yes"], ["Yes", "Yes", "No"], pos_label="Yes"), 0.5)
        self.assertAlmostEqual(f1([1, 0], [1, 0]), 1.0)

    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 1, 1], [0, 0, 1], labels=[0, 1])
        np.testing.assert_array_equal(cm, [[1, 0], [1, 1]])

    def test_mse(self):
        self.assertEqual(mse([1, 2], [2, 4]), 2.5)

    def test_roc_and_auc(self):
        fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        area = auc(fpr, tpr)
        self.assertGreaterEqual(area, 0)
        self.assertLessEqual(area, 1)
        self.assertEqual(len(fpr), len(thresholds))

    def test_gradient(self):
        self.assertAlmostEqual(grad(lambda x: x * x, 3.0), 6.0, places=4)

    def test_jacobian(self):
        out = jacobian(lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1]]), np.array([2., 3.]))
        np.testing.assert_allclose(out, [[4, 1], [3, 2]], atol=1e-4)

    def test_sigmoid_stable(self):
        values = sigmoid(np.array([-1000., 0., 1000.]))
        np.testing.assert_allclose(values, [0, 0.5, 1], atol=1e-12)

    def test_softmax_sums_to_one(self):
        out = softmax([[1000., 1001.]])
        np.testing.assert_allclose(out.sum(axis=1), [1.])

    def test_softmax_positive_infinity(self):
        out = softmax([np.inf, 1.0, np.inf])
        np.testing.assert_allclose(out, [0.5, 0.0, 0.5])

    def test_logsumexp_large_values(self):
        value = logsumexp([1000., 1000.])
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 1000 + np.log(2))

    def test_pairwise_distances(self):
        out = pairwise_distances([[0, 0], [3, 4]])
        np.testing.assert_allclose(out, [[0, 5], [5, 0]])

    def test_batches(self):
        batches = list(iter_batches(np.arange(5), 2))
        self.assertEqual([len(b) for b in batches], [2, 2, 1])

    def test_pipeline_with_column_selector(self):
        X = np.array([[1., 10.], [3., 20.], [5., 30.]])
        union = FeatureUnion([
            ("first", Pipeline([("select", ColumnSelector([0])), ("scale", StandardScaler())])),
            ("second", Pipeline([("select", ColumnSelector([1])), ("scale", MinMaxScaler())])),
        ])
        out = union.fit_transform(X)
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(out[:, 0].mean(), 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

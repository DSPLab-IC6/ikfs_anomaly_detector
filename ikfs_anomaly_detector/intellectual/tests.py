import unittest

import numpy as np

from ikfs_anomaly_detector.intellectual.autoencoder import LSTMAutoencoder
from ikfs_anomaly_detector.intellectual.predictor import LSTMPredictor
from ikfs_anomaly_detector.intellectual.utils import (
    z_normalization,
    calculate_mean,
    find_anomaly_points,
    calculate_covariance_matrix,
    mahalanobis_distance,
    squared_error,
    ewma,
)


class TestAutoencoder(unittest.TestCase):

    def test_init(self) -> None:
        LSTMAutoencoder(signals_count=5)


class TestPredictor(unittest.TestCase):

    def test_init(self) -> None:
        LSTMPredictor()


class TestUtils(unittest.TestCase):

    def test_z_normalization(self) -> None:
        arr = np.array([1, 2, 3, 30, 30, 3, 2, 1])
        self.assertListEqual(
            list(z_normalization(arr)), [
                -0.6587095756740946,
                -0.5763708787148328,
                -0.49403218175557095,
                1.7291126361444984,
                1.7291126361444984,
                -0.49403218175557095,
                -0.5763708787148328,
                -0.6587095756740946,
            ])

    def test_calculate_mean(self) -> None:
        arr = np.array([1, 2, 5, 10])
        self.assertAlmostEqual(calculate_mean(arr), arr.mean(axis=0))

    def test_calculate_covariance_matrix(self) -> None:
        arr = np.array([[1., 1.5], [2., 2.], [3., 1.], [1., 5.]])
        mean, matrix = calculate_covariance_matrix(arr)

        self.assertAlmostEqual(mean, 2.0625)
        self.assertListEqual([list(v) for v in matrix], [[0.78515625, -0.87890625],
                                                         [-0.87890625, 2.51953125]])

    def test_mahalanobis_distance(self) -> None:
        arr = np.array([[1., 1.5], [2., 2.], [3., 1.], [1., 5.]])
        mean, matrix = calculate_covariance_matrix(arr)

        self.assertAlmostEqual(mahalanobis_distance(arr[0], mean, matrix), 3.43629, places=5)

    def test_squared_error(self) -> None:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        self.assertListEqual(list(squared_error(a, b)), [3.0, 3.0, 3.0])

    def test_ewma(self) -> None:
        arr = np.array(range(10))
        self.assertListEqual(list(ewma(arr, window=4)), [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    def test_find_anomaly_points(self) -> None:
        arr = np.array([1, 10, 20, 30, 10, 20, 2, 2, 1, 10])
        self.assertListEqual(find_anomaly_points(arr, threshold=5, offset=2), [1, 3, 5, 9])

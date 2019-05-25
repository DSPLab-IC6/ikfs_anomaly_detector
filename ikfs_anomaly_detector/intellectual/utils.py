from typing import Tuple, cast, Optional, List

import numpy as np


def z_normalization(data: np.ndarray) -> np.ndarray:
    std = data.std()
    if std == 0:
        return data

    result = data.copy()
    mean = result.mean()
    return (result - mean) / std


def calculate_mean(data: np.ndarray) -> float:
    return sum(data) / len(data)


def calculate_covariance_matrix(data: np.ndarray) -> Tuple[float, np.ndarray]:
    data_mean = data.mean()

    cov_matrix = 0.
    for vec in data:
        cov_matrix += np.dot((vec - data_mean).reshape(len(vec), 1), (vec - data_mean).reshape(1, len(vec)))
    cov_matrix /= len(data)

    return data_mean, cov_matrix


def mahalanobis_distance(x: np.ndarray, mean: float, cov_matrix: np.ndarray) -> float:
    distance = np.dot(x - mean, np.linalg.inv(cov_matrix))
    distance = np.dot(distance, (x - mean).T)
    return cast(float, distance)


def squared_error(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.sqrt((predictions - targets) ** 2)


def ewma(data: np.ndarray, window: int, alpha: Optional[float] = None) -> np.ndarray:
    """Exponential weighted moving average"""
    alpha = alpha or (2. / (window + 1))
    t_ = window - 1
    ema = np.zeros_like(data, dtype=float)

    for i in range(len(data)):
        # Special Case
        if i > t_:
            ema[i] = (data[i] - ema[i - 1]) * alpha + ema[i - 1]
        else:
            ema[i] = np.mean(data[:i + 1])
    return ema


def find_anomaly_points(data: np.ndarray, offset: int = 500, threshold: float = None) -> List[int]:
    assert offset >= 1

    threshold = threshold or (data.mean() + data.std() * 3)
    anomaly_points = []

    i = 0
    while i <= len(data) - 1:
        if data[i] >= threshold:
            anomaly_points.append(i)
            i += offset
        else:
            i += 1

    return anomaly_points


def find_anomaly_points_by_median(data: np.array, offset: int = 500, threshold: int = 10) -> List[int]:
    """https://stackoverflow.com/a/45399188"""
    assert offset >= 1

    distances = np.abs(data - np.median(data))
    mdev = np.median(distances)

    anomaly_points = []

    i = 0
    while i <= len(data) - 1:
        if (distances[i] / (mdev or 1.)) >= threshold:
            anomaly_points.append(i)
            i += offset
        else:
            i += 1

    return anomaly_points

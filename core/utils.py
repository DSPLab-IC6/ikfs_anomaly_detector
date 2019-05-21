import collections
import os
from typing import Iterable

import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def as_set(cls: type) -> set:
    result = set()

    for attr in dir(cls):
        if not attr.startswith('__'):
            data = getattr(cls, attr)
            if isinstance(data, collections.Hashable):
                result.add(data)

    return result


def minmax_normalize(data: np.ndarray) -> np.ndarray:
    denominator = data.max() - data.min()
    if denominator == 0:
        raise ZeroDivisionError
    return (data - data.min()) / denominator


def fill_zeros_with_previous(data: np.ndarray) -> np.ndarray:
    if np.array_equal(data, data.astype(bool)):
        return data

    result = data.copy()
    for i, x in enumerate(result):
        if isinstance(x, Iterable):
            continue
        if x == 0:
            result[i] = result[i - 1]
    return result

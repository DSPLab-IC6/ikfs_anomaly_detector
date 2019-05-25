import collections
import os
from typing import Iterable, TextIO

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


def roll_up_points(data: list) -> list:
    result = []

    i = 0
    while i < len(data):
        k = i
        start = data[k]
        end = None
        while (k + 1 < len(data)) and (data[k + 1] - data[k] == 1):
            end = data[k + 1]
            k += 1
            i += 1

        if end is None:
            result.append(start)
        else:
            result.append((start, end))

        i += 1

    return result


def write_rolluped_points(points: list, file: TextIO) -> None:
    for point in points:
        if isinstance(point, tuple):
            file.write(f'\t{point[0]} - {point[1]}\n')
        else:
            file.write(f'\t{point}\n')

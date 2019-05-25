import unittest

import numpy as np
from mock import patch, MagicMock, Mock

from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.core.utils import as_set, minmax_normalize, fill_zeros_with_previous


class TestReader(unittest.TestCase):

    def setUp(self) -> None:
        self.reader_file_mock = MagicMock(keys=Mock(return_value=['a', 'b', 'c']))

        with patch('ikfs_anomaly_detector.core.reader.h5py.File') as FileMock:
            FileMock.return_value = self.reader_file_mock
            self.reader = TelemetryReader('path')

    def test_init(self) -> None:
        self.assertIs(self.reader._file, self.reader_file_mock)
        self.assertFalse(self.reader._cache)

    def test_close(self) -> None:
        self.reader.close()
        self.reader_file_mock.close.assert_called_once()

    def test_list_signals(self) -> None:
        self.assertListEqual(self.reader.list_signals(), ['a', 'b', 'c'])


class TestUtils(unittest.TestCase):

    def test_as_set(self) -> None:
        class State:
            a = 1
            b = 2
            c = 3

        self.assertSetEqual(as_set(State), {1, 2, 3})

    def test_minmax_normalize(self) -> None:
        arr = np.array([1, 2, 3, 4, 5])
        self.assertListEqual(list(minmax_normalize(arr)), [0., 0.25, 0.5, 0.75, 1.])

        arr = np.array([0])
        with self.assertRaises(ZeroDivisionError):
            minmax_normalize(arr)

    def test_fill_zeros_with_previous(self) -> None:
        arr = np.array([0, 1, 1, 0, 1])
        self.assertListEqual(list(fill_zeros_with_previous(arr)), [0, 1, 1, 0, 1])

        arr = np.array([0, 0, 20, 21, 0, 19])
        self.assertListEqual(list(fill_zeros_with_previous(arr)), [19, 19, 20, 21, 21, 19])

        # Ignore iterables inside
        arr = np.array([[100, 0, 200], [200, 0]])
        self.assertListEqual(list(fill_zeros_with_previous(arr)), [[100, 0, 200], [200, 0]])


class TestPrinter(unittest.TestCase):

    def test_import(self) -> None:
        # noinspection PyUnresolvedReferences
        from ikfs_anomaly_detector.core.printer import plot_telemetry

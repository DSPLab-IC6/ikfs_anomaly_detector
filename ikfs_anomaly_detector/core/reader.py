from typing import List

import h5py
import numpy as np

from ikfs_anomaly_detector.core.custom_types import Signals


class TelemetryReader:

    def __init__(self, path: str) -> None:
        self._file = h5py.File(path, 'r')
        self._cache = {}

    def __enter__(self) -> 'TelemetryReader':
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def list_signals(self) -> List[str]:
        return list(self._file.keys())

    def get_signals(self, *names) -> Signals:
        result = {}

        for name in names:
            if name in self._cache:
                result[name] = self._cache[name]
            else:
                try:
                    result[name] = self._cache[name] = self._file[name][()]
                except KeyError:
                    raise KeyError(f'Сигнала {name} нет в файле')

        return result

    def get_all_signals(self) -> Signals:
        return self.get_signals(*self.list_signals())

    def get_signal(self, name: str) -> np.ndarray:
        return self.get_signals(name)[name]

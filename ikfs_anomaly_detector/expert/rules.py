import abc
from dataclasses import dataclass
from typing import Union, Callable

from ikfs_anomaly_detector.core.custom_types import SignalDetails, Signals, PointIndex
from ikfs_anomaly_detector.core.utils import fill_zeros_with_previous


class Weight:
    normal = 0.
    warning = 5.
    error = 10.
    critical = 20.


@dataclass
class BaseRule(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def apply_to_signals(self, signals: Signals, error_rate: dict, details: dict) -> None:
        pass

    @staticmethod
    def _get_point_index(idx: int) -> str:
        return f'{idx:06}_Point'


@dataclass
class StatedTelemetry(BaseRule):
    signal: str  # Название сигнала
    state_weights: Union[dict, set]  # Состояние - вес

    def apply_to_signals(self, signals: Signals, error_rate: dict, details: dict) -> None:
        values = signals[self.signal]

        for i, state in enumerate(values):
            if isinstance(self.state_weights, set):
                weight = (
                    Weight.normal
                    if state in self.state_weights else
                    Weight.critical
                )
            else:
                weight = self.state_weights.get(state, Weight.critical)

            point_index = self._get_point_index(i)

            sum_weight = error_rate.setdefault(point_index, 0.)
            error_rate[point_index] = sum_weight + weight

            if weight:
                details.setdefault(point_index, {})[self.signal] = SignalDetails(value=state, weight=weight)


@dataclass
class NumericalTelemetry(BaseRule):
    signal: str  # Название сигнала
    low: float  # Нижняя граница
    expected: float  # Ожидаемое значение
    high: float  # Верхняя граница

    def apply_to_signals(self, signals: Signals, error_rate: dict, details: dict) -> None:
        values = fill_zeros_with_previous(signals[self.signal])

        for i, value in enumerate(values):
            if not (self.low <= value <= self.high):
                weight = Weight.critical
            else:
                weight = abs(value - self.expected) / Weight.error

            point_index = self._get_point_index(i)

            sum_weight = error_rate.setdefault(point_index, 0.)
            error_rate[point_index] = sum_weight + weight

            if weight:
                details.setdefault(point_index, {})[self.signal] = SignalDetails(value=value, weight=weight)


@dataclass
class RelatedTelemetry(BaseRule):
    signal: str  # Название сигнала
    expected: Callable[[Signals, PointIndex], float]  # Ожидаемое значение
    error: float  # Погрешность

    def apply_to_signals(self, signals: Signals, error_rate: dict, details: dict) -> None:
        values = fill_zeros_with_previous(signals[self.signal])

        for i, value in enumerate(values):
            expected = self.expected(signals, i)

            if not (expected - self.error <= value <= expected + self.error):
                weight = Weight.critical
            else:
                weight = abs(value - expected) / Weight.error

            point_index = self._get_point_index(i)

            sum_weight = error_rate.setdefault(point_index, 0.)
            error_rate[point_index] = sum_weight + weight

            if weight:
                details.setdefault(point_index, {})[self.signal] = SignalDetails(value=value, weight=weight)

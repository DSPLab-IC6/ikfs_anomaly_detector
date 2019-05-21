import os

import numpy as np

from core.custom_types import Signals
from core.format.telemetry import TelemetryAttrs
from core.printer import plot_telemetry, Subplot, Signal
from core.reader import TelemetryReader
from intellectual.autoencoder import LSTMAutoencoder
from intellectual.predictor import LSTMPredictor, SIGNALS_FOR_TRAINING
from intellectual.signal_groups import SIGNALS_GROUPS

DATA_PATH = '/Users/anthony/Desktop/best_diploma/data/'

SPEED_PER_POINT = 2000
POINTS_COUNT = 5000


def _flat_speed_signal(data: np.ndarray) -> np.ndarray:
    return data[:, 1]


def join_signals_from_files(*signal_names, kind: str = 'good') -> Signals:
    path = DATA_PATH + kind
    signals = {}

    for file_name in os.listdir(path):
        if not file_name.endswith('.h5'):
            continue

        print(f'Чтение телеметрии из файла "{file_name}"')

        with TelemetryReader(f'{path}/{file_name}') as reader:
            signals_in_file = reader.get_signals(*signal_names)
            signals.update({
                name: np.concatenate((
                    signals.get(name, np.array([])),
                    (
                        _flat_speed_signal(signal)
                        if name == TelemetryAttrs.speed else
                        signal
                    )
                ))
                for name, signal in signals_in_file.items()
            })

    return signals


def train_predictor(postfix: str = '') -> None:
    predictor = LSTMPredictor()

    for signal_name in SIGNALS_FOR_TRAINING:
        signals_data = join_signals_from_files(signal_name)
        # noinspection PyBroadException
        try:
            print(f'Обучение предиктора для сигнала "{signal_name}"...')
            predictor.train(signal_name + postfix, signals_data[signal_name])
        except Exception as exc:
            print(f'Ошибка: {exc}')


def train_autoencoder() -> None:
    for name, group in SIGNALS_GROUPS.items():
        signals_data = join_signals_from_files(*group.signals)
        group.signals_data = signals_data

        encoder = LSTMAutoencoder(len(group.signals))
        # noinspection PyBroadException
        try:
            print(f'Обучение автокодировщика для группы сигналов "{group.name}"...')
            encoder.train(group)
        except Exception as exc:
            print(f'Ошибка: {exc}')


# TODO(a.telyshev): Нормальные данные содержат аномалии?
def train_predictor_for_speed() -> None:
    signal = join_signals_from_files(TelemetryAttrs.speed)[TelemetryAttrs.speed]
    plot_telemetry(*[
        Subplot(
            signals=[
                Signal('Speed', signal)
            ],
        )
    ])


if __name__ == '__main__':
    # train_predictor()
    # train_predictor_for_speed()
    # train_autoencoder()
    pass

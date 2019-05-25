import os
from typing import List

import numpy as np

from ikfs_anomaly_detector.core.custom_types import Signals
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.intellectual.autoencoder import LSTMAutoencoder, SignalsGroup
from ikfs_anomaly_detector.intellectual.predictor import LSTMPredictor


def join_signals_from_files(*signal_names, telemetry_dir: str) -> Signals:
    signals = {}

    for file_name in os.listdir(telemetry_dir):
        if not file_name.endswith('.h5'):
            continue

        print(f'Чтение телеметрии из файла "{file_name}"...')

        with TelemetryReader(f'{telemetry_dir}/{file_name}') as reader:
            signals_in_file = reader.get_signals(*signal_names)
            signals.update({
                name: np.concatenate((
                    signals.get(name, np.array([])), signal
                ))
                for name, signal in signals_in_file.items()
            })

    return signals


def train_predictor(
        signals: List[str],
        telemetry_dir: str,
        models_dir: str,
        tensorboard_dir: str,
) -> None:
    predictor = LSTMPredictor(models_dir, tensorboard_dir)

    for signal_name in signals:
        signals_data = join_signals_from_files(signal_name, telemetry_dir=telemetry_dir)
        predictor.train(signal_name, signals_data[signal_name])


def train_autoencoder(
        signal_groups: List[SignalsGroup],
        telemetry_dir: str,
        models_dir: str,
        tensorboard_dir: str,
) -> None:
    for group in signal_groups:
        signals_data = join_signals_from_files(*group.signals, telemetry_dir=telemetry_dir)
        group.signals_data = signals_data

        encoder = LSTMAutoencoder(len(group.signals), models_dir, tensorboard_dir)
        encoder.train(group)

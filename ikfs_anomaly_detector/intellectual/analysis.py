from typing import List, Dict

import numpy as np

from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs
from ikfs_anomaly_detector.core.printer import Subplot, Signal, plot_telemetry, Label, Colours, Ticks
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.core.utils import roll_up_points
from ikfs_anomaly_detector.intellectual.autoencoder import SignalsGroup, LSTMAutoencoder
from ikfs_anomaly_detector.intellectual.predictor import LSTMPredictor
from ikfs_anomaly_detector.intellectual.utils import find_anomaly_points


def run_predictor(
        reader: TelemetryReader,
        models_dir: str,
        signals: List[str],
        thresholds: Dict[str, float],
        results_path: str,
        anomalies_points_storage: dict,
        full_report: bool = False,
) -> None:
    predictor = LSTMPredictor(models_dir=models_dir)

    for signal in signals:
        print(f'LSTM-предиктор: анализ сигнала "{signal}"...')

        result = predictor.analyze(reader.get_signals(signal))

        threshold = thresholds[signal]
        subplots = []

        if full_report:
            subplots.append(
                Subplot(
                    signals=[
                        Signal(signal, result.data, color=Colours.blue, alpha=.5),
                        Signal(f'{signal}__predicted', result.predicted_data, color=Colours.green, alpha=.5)
                    ],
                    xlabel=Label('Индекс точки измерения'),
                )
            )
            anomaly_points = find_anomaly_points(result.mahalanobis_distance, offset=1, threshold=threshold)
            anomalies_points_storage[f'predicted__{signal}'] = anomaly_points
        else:
            anomaly_points = []

        subplots.append(
            Subplot(
                signals=[
                    Signal(f'Расстояние Махаланобиса', result.mahalanobis_distance, color=Colours.red),
                    Signal('Граница аномалии', np.array([threshold] * len(result.data)), color=Colours.green),
                ],
                xlabel=Label('Индекс точки измерения'),
                ylim=(0, 1000),
            ),
        )

        print(f'LSTM-предиктор: печать изображений"...')
        plot_telemetry(
            *subplots,
            img_path=f'{results_path}/predicted__{signal}.png',
            anomaly_points=anomaly_points,
        )

        if full_report and anomaly_points and (signal == TelemetryAttrs.scanner_angle):
            for anomaly in roll_up_points(anomaly_points):
                data = reader.get_signal(TelemetryAttrs.scanner_angle)

                if isinstance(anomaly, tuple):
                    data = data[anomaly[0] - 250: anomaly[1] + 250]
                    ticks = Ticks(start=anomaly[0] - 250, period=50)
                    path = f'{results_path}/predicted__{signal}__{anomaly[0]}_{anomaly[1]}.png'
                    selections = range(250, 250 + anomaly[1] - anomaly[0])
                else:
                    data = data[anomaly - 250: anomaly + 250]
                    ticks = Ticks(start=anomaly - 250, period=50)
                    path = f'{results_path}/predicted__{signal}__{anomaly}.png'
                    selections = [250]

                print(f'LSTM-предиктор: печать увеличенных фрагментов сигнала {TelemetryAttrs.scanner_angle}"...')
                plot_telemetry(
                    Subplot(
                        signals=[Signal(TelemetryAttrs.scanner_angle, data)],
                        xlabel=Label('Индекс точки измерения'),
                        ticks=ticks,
                    ),
                    img_path=path,
                    anomaly_points=selections,
                    anomaly_selection_width=10,
                )


def run_autoencoder(
        reader: TelemetryReader,
        models_dir: str,
        signal_groups: List[SignalsGroup],
        thresholds: Dict[str, float],
        results_path: str,
        anomalies_points_storage: dict,
        full_report: bool = False,
) -> None:
    for group in signal_groups:
        print(f'LSTM-автокодировщик: анализ группы сигналов "{group.name}" {group.signals}...')

        encoder = LSTMAutoencoder(len(group.signals), models_dir)

        group.signals_data = reader.get_signals(*group.signals)
        result = encoder.analyze(group)

        threshold = thresholds[group.name]
        subplots = [
            Subplot(
                signals=[
                    Signal('EWMA MSE', result.ewma_mse, color=Colours.red),
                    Signal('Граница аномалии', np.array([threshold] * len(result.ewma_mse)), color=Colours.green),
                ],
                xlabel=Label('Индекс точки измерения'),
                ylabel=Label(''),
            ),
        ]

        if full_report:
            anomaly_points = find_anomaly_points(result.ewma_mse, offset=1, threshold=threshold)
            anomalies_points_storage[f'group__{group.name}'] = anomaly_points

            subplots.extend([
                Subplot(
                    signals=[
                        Signal(group.signals[i], data, color=Colours.black),
                        Signal(f'{group.signals[i]}__decoded', decoded, color=Colours.green),
                    ],
                    xlabel=Label('Индекс точки измерения'),
                )
                for i, (data, decoded) in enumerate(zip(result.signals, result.decoded_signals))
            ])

        print(f'LSTM-автокодировщик: печать изображений...')
        plot_telemetry(
            *subplots,
            img_path=f'{results_path}/group__{group.name}.png',
        )

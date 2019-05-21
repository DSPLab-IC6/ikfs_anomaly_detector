import os

import numpy as np
import yaml

from core.format.telemetry import TelemetryAttrs
from core.printer import plot_telemetry, Subplot, Label, Signal, Colours, Ticks
from core.reader import TelemetryReader
from expert.analyzer import ExpertAnalyzer
from intellectual import signal_groups
from intellectual.autoencoder import LSTMAutoencoder
from intellectual.predictor import LSTMPredictor, SIGNALS_FOR_TRAINING
from intellectual.utils import find_anomaly_points

KIND = 'bad'
PRINT_ERRORS_ONLY = False


def load_thresholds() -> dict:
    with open('thresholds.yml', 'r') as stream:
        try:
            thresholds = yaml.safe_load(stream)
            assert 'default' in thresholds, 'В конфигурации границ не найдена секция "default"'
            return thresholds

        except yaml.YAMLError as exc:
            print('Ошибка при загрузке файла конфигурации: ' + str(exc))


THRESHOLDS = load_thresholds()


def get_threshold(fname: str, signal_name: str) -> float:
    try:
        return THRESHOLDS.get(fname, {}).get(signal_name, THRESHOLDS['default'][signal_name])
    except KeyError:
        print(f'Для сигнала "{signal_name}" не определена аномальная граница')


ANOMALIES_POINTS = {}


def _roll_up_points(data: list) -> list:
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


def _expert_analysis(reader: TelemetryReader, fname: str) -> None:
    expert = ExpertAnalyzer(reader)
    results = expert.analyze()

    plot_telemetry(
        *[
            Subplot(
                signals=[
                    Signal(result.tester, np.array(result.error_rate)),
                    Signal('Граница аномалии', np.array([get_threshold('', 'rules')] * len(result.error_rate)),
                           color=Colours.red),
                ],
                xlabel=Label('Точка телеметрии'),
                ylabel=Label('Показатель ошибки'),
            )
            for result in results
        ],
        img_path=f'results/{KIND}/{fname}/rules.png',
    )


def _run_predictor(reader: TelemetryReader, fname: str) -> None:
    predictor = LSTMPredictor()

    for signal in SIGNALS_FOR_TRAINING:
        print(f'Анализ предиктором сигнала "{signal}"...')

        try:
            result = predictor.analyze(reader.get_signals(signal))
        except Exception:
            print('err for ', signal)
            continue

        threshold = get_threshold(fname, signal)
        anomaly_points = find_anomaly_points(result.mahalanobis_distance, offset=1, threshold=threshold)
        ANOMALIES_POINTS[f'predicted__{signal}'] = anomaly_points

        subplots = [] if PRINT_ERRORS_ONLY else [
            Subplot(
                signals=[
                    Signal(signal, result.data, color=Colours.blue, alpha=.5),
                    Signal(f'{signal}__predicted', result.predicted_data, color=Colours.green, alpha=.5)
                ],
                xlabel=Label('Индекс точки измерения'),
            ),
        ]
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

        plot_telemetry(
            *subplots,
            img_path=f'results/{KIND}/{fname}/predicted__{signal}.png',
            anomaly_points=anomaly_points,
        )

        if signal == TelemetryAttrs.scanner_angle and anomaly_points:
            for anomaly in _roll_up_points(anomaly_points):
                data = reader.get_signal(TelemetryAttrs.scanner_angle)

                if isinstance(anomaly, tuple):
                    data = data[anomaly[0] - 250: anomaly[1] + 250]
                    ticks = Ticks(start=anomaly[0] - 250, period=50)
                    path = f'results/{KIND}/{fname}/predicted__{signal}__{anomaly[0]}_{anomaly[1]}.png'
                    selections = range(250, 250 + anomaly[1] - anomaly[0])
                else:
                    data = data[anomaly - 250: anomaly + 250]
                    ticks = Ticks(start=anomaly - 250, period=50)
                    path = f'results/{KIND}/{fname}/predicted__{signal}__{anomaly}.png'
                    selections = [250]

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


def _run_autoencoder(reader: TelemetryReader, fname: str) -> None:
    for group_name, group in signal_groups.SIGNALS_GROUPS.items():
        print(f'Анализ автокодировщиком группы сигналов "{group_name}"...')

        encoder = LSTMAutoencoder(len(group.signals))

        group.signals_data = reader.get_signals(*group.signals)
        result = encoder.analyze(group)

        res = group.signals_data.copy()
        res['err'] = result.ewma_mse

        signals = list(group.signals_data.keys())

        threshold = get_threshold(fname, group_name)
        anomaly_points = find_anomaly_points(result.ewma_mse, offset=1, threshold=threshold)
        ANOMALIES_POINTS[f'group__{group_name}'] = anomaly_points

        subplots = [Subplot(
            signals=[
                Signal('EWMA MSE', result.ewma_mse, color=Colours.red),
                Signal('Граница аномалии', np.array([threshold] * len(result.ewma_mse)), color=Colours.green),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label(''),
        )]

        if not PRINT_ERRORS_ONLY:
            subplots.extend([
                Subplot(
                    signals=[
                        Signal(signals[i], data, color=Colours.black),
                        Signal(f'{signals[i]}__decoded', decoded, color=Colours.green),
                    ],
                    xlabel=Label('Индекс точки измерения'),
                )
                for i, (data, decoded) in enumerate(zip(result.signals, result.decoded_signals))
            ])

        plot_telemetry(
            *subplots,
            img_path=f'results/{KIND}/{fname}/group__{group_name}.png',
        )


def main() -> None:
    # path = f'/Users/anthony/Desktop/best_diploma/data/{KIND}/'
    path = f'/home/anton/ikfs_anomaly/data/{KIND}/'

    for file_name in os.listdir(path):
        if not file_name.endswith('.h5'):
            continue

        fname = file_name.split(".")[0]
        if os.path.exists(f'results/{KIND}/{fname}'):
            continue

        os.mkdir(f'results/{KIND}/{fname}')

        print(f'\nProcess "{file_name}"')
        telemetry_file = f'{path}{file_name}'

        with TelemetryReader(telemetry_file) as reader:
            _expert_analysis(reader, fname)
            _run_predictor(reader, fname)
            _run_autoencoder(reader, fname)

        print('Печать выходного файла...')
        anomalies_counter = [0] * 100_000

        with open(f'results/{KIND}/{fname}/anomalies.txt', 'w') as f:
            f.write('Аномальные участки:\n')

            for signal, points in ANOMALIES_POINTS.items():
                if not points:
                    continue

                for i in points:
                    anomalies_counter[i] += 1

                f.write(f'- {signal}\n')
                for p in _roll_up_points(points):
                    if isinstance(p, tuple):
                        f.write(f'\t{p[0]} - {p[1]}\n')
                    else:
                        f.write(f'\t{p}\n')

            for min_cnt in range(2, 5 + 1):
                anomaly_points = _roll_up_points([i for i, cnt in enumerate(anomalies_counter) if cnt > min_cnt])

                f.write(f'\nАномальные участки, встречающиеся среди анализаторов более {min_cnt} раз:\n')
                if anomaly_points:
                    for p in anomaly_points:
                        if isinstance(p, tuple):
                            f.write(f'\t{p[0]} - {p[1]}\n')
                        else:
                            f.write(f'\t{p}\n')
                else:
                    f.write('отсутствуют.\n')


if __name__ == '__main__':
    # PRINT_ERRORS_ONLY = True
    main()

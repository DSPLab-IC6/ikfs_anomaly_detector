import os

import numpy as np

from core.format.telemetry import TelemetryAttrs
from core.printer import plot_telemetry, Subplot, Label, Signal, Colours
from core.reader import TelemetryReader
from expert.analyzer import ExpertAnalyzer
from intellectual import signal_groups
from intellectual.autoencoder import LSTMAutoencoder
from intellectual.predictor import LSTMPredictor, SIGNALS_FOR_TRAINING
from intellectual.utils import find_anomaly_points

KIND = 'bad'
THRESHOLDS = {
    'rules': 0.55,

    TelemetryAttrs.ppt_ripple: 100,
    TelemetryAttrs.ppt_sample_count: 100,
    TelemetryAttrs.scanner_angle: 420,
    TelemetryAttrs.str_power: 200,

    signal_groups.BFK_GROUP: 0.225,
    signal_groups.BPOP_GROUP: 0.1,
    signal_groups.BUD_BOARD_GROUP: 15.5,
    signal_groups.BUD_GROUP: 3.5,
    signal_groups.FP_GROUP: 0.1,
    signal_groups.MI_GROUP: 0.1,
    signal_groups.MK_GROUP: 0.55,
    signal_groups.PPT_DIRECTION_GROUP: 0.1,
    signal_groups.PPT_GROUP: 1,
    signal_groups.STR_GROUP: 0.04,
}

ANOMALIES_POINTS = {}


def _expert_analysis(reader: TelemetryReader, fname: str) -> None:
    expert = ExpertAnalyzer(reader)
    results = expert.analyze()

    plot_telemetry(
        *[
            Subplot(
                signals=[
                    Signal(result.tester, np.array(result.error_rate)),
                    Signal('Граница аномалии', np.array([THRESHOLDS['rules']] * len(result.error_rate)), color=Colours.red),
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

        threshold = THRESHOLDS[signal]
        anomaly_points = find_anomaly_points(result.mahalanobis_distance, offset=1, threshold=threshold)
        ANOMALIES_POINTS[signal] = anomaly_points

        plot_telemetry(
            Subplot(
                signals=[
                    Signal(signal, result.data, color=Colours.blue, alpha=.5),
                    Signal(f'{signal}__predicted', result.predicted_data, color=Colours.green, alpha=.5)
                ],
                xlabel=Label('Индекс точки измерения'),
            ),
            Subplot(
                signals=[
                    Signal(f'Расстояние Махаланобиса', result.mahalanobis_distance, color=Colours.red),
                    Signal('Граница аномалии', np.array([threshold] * len(result.data)), color=Colours.green),
                ],
                xlabel=Label('Индекс точки измерения'),
                ylim=(0, 1000),
            ),
            img_path=f'results/{KIND}/{fname}/predicted__{signal}.png',
            anomaly_points=anomaly_points,
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

        subplots = [
            Subplot(
                signals=[
                    Signal(signals[i], data, color=Colours.black),
                    Signal(f'{signals[i]}__decoded', decoded, color=Colours.green),
                ],
                xlabel=Label('Индекс точки измерения'),
            )
            for i, (data, decoded) in enumerate(zip(result.signals, result.decoded_signals))
        ]

        threshold = THRESHOLDS[group_name]
        anomaly_points = find_anomaly_points(result.ewma_mse, offset=1, threshold=threshold)
        ANOMALIES_POINTS[group_name] = anomaly_points

        subplots.append(Subplot(
            signals=[
                Signal('EWMA MSE', result.ewma_mse, color=Colours.red),
                Signal('Граница аномалии', np.array([threshold] * len(result.ewma_mse)), color=Colours.green),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label(''),
        ))

        plot_telemetry(
            *subplots,
            # anomaly_points=anomaly_points[::100],
            img_path=f'results/{KIND}/{fname}/group__{group_name}.png',
        )


def main() -> None:
    path = f'/Users/anthony/Desktop/best_diploma/data/{KIND}/'

    for file_name in os.listdir(path):
        if not file_name.endswith('.h5'):
            continue

        fname = file_name.split(".")[0]
        os.mkdir(f'results/{KIND}/{fname}')

        print(f'\nProcess "{file_name}"')
        telemetry_file = f'{path}{file_name}'

        with TelemetryReader(telemetry_file) as reader:
            _expert_analysis(reader, fname)
            _run_predictor(reader, fname)
            _run_autoencoder(reader, fname)

        # Напечатать кусочек
        # plot_telemetry(*[
        #     Subplot(
        #         signals=[
        #             Signal('ScannerAngle', reader.get_signal(TelemetryAttrs.scanner_angle)[16500:17000]),
        #         ],
        #         xlabel=Label('Points'),
        #         ylabel=Label(''),
        #         ticks=Ticks(start=16500, period=25),
        #     )
        # ])

        anomalies_counter = [0] * 100_000

        with open(f'results/{KIND}/{fname}/anomalies.txt', 'w') as f:
            f.write('Аномальные точки:\n')

            for signal, points in ANOMALIES_POINTS.items():
                pts = '\t\n'.join(map(str, points))
                f.write(f"{signal}:\n\t{pts}\n")

                for i in points:
                    anomalies_counter[i] += 1

            pts = '\t\n'.join(map(str, [i for i, cnt in enumerate(anomalies_counter) if cnt > 2]))
            f.write(f'Аномальные точки, «угаданные» больше 3-х раз:\n\t{pts}\n')

        return


if __name__ == '__main__':
    main()

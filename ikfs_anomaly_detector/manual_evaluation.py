from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs
from ikfs_anomaly_detector.core.printer import plot_telemetry, Subplot, Signal, Colours, Label
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.core.utils import fill_zeros_with_previous
from ikfs_anomaly_detector.intellectual import signal_groups
from ikfs_anomaly_detector.intellectual.autoencoder import LSTMAutoencoder
from ikfs_anomaly_detector.intellectual.predictor import LSTMPredictor, SIGNALS_FOR_TRAINING

GOOD_FILE = '/Users/anthony/Desktop/best_diploma/data/good/METM2_22293_22286_1VIE2-IMR_8_IKFS-2_01P8.rsm.tlm.h5'

# GOOD_FILE = '/home/anton/ikfs_anomaly/data/good/METM2_22293_22286_1VIE2-IMR_8_IKFS-2_01P8.rsm.tlm.h5'

THRESHOLD = {
    TelemetryAttrs.ppt_ripple: 10,
    TelemetryAttrs.ppt_sample_count: 10,
    TelemetryAttrs.scanner_angle: 100,
    TelemetryAttrs.str_power: 20,
    TelemetryAttrs.tu1_temperature: 20,
    TelemetryAttrs.tu2_temperature: 20,

    signal_groups.BFK_GROUP: 4.,
    signal_groups.BPOP_GROUP: 2.,
    signal_groups.BUD_GROUP: 8.,
    signal_groups.BUD_BOARD_GROUP: 4.,
    signal_groups.FP_GROUP: 0.4,
    signal_groups.MI_GROUP: 2.5,
    signal_groups.MK_GROUP: 1.,
    signal_groups.PPT_GROUP: 1.,
    signal_groups.PPT_DIRECTION_GROUP: 1.,
    signal_groups.STR_GROUP: 0.1,
}

group_names = {
    signal_groups.BFK_GROUP: 'БФК',
    signal_groups.BPOP_GROUP: 'БПОП',
    signal_groups.BUD_GROUP: 'БУД',
    signal_groups.BUD_BOARD_GROUP: 'Плата БУД',
    signal_groups.FP_GROUP: 'ФП',
    signal_groups.MI_GROUP: 'МИ',
    signal_groups.MK_GROUP: 'МК',
    signal_groups.PPT_GROUP: 'ППТ',
    signal_groups.PPT_DIRECTION_GROUP: 'Направ. ППТ',
    signal_groups.STR_GROUP: 'СТР',
}


def _change_ppt_ripple(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    labels = np.array([0.] * len(data))

    for i in range(3):
        index = np.random.randint(0, len(data))
        data[index] = 1.5
        labels[index] = 1.

    index = np.random.randint(0, 30_000)
    for i in range(index, index + 10000):
        data[i] = 0.1 + np.random.normal(-0.2, 0.2)
        labels[i] = 1.

    return labels, data


def _change_ppt_sample_count(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    labels = np.array([0.] * len(data))

    for i in range(5):
        index = np.random.randint(0, len(data))
        data[index] = 25870
        labels[index] = 1.

    index = np.random.randint(0, 40_000)
    for i in range(index, index + np.random.randint(10_000)):
        data[i] = 25870 + np.random.normal(-5, 5)
        labels[i] = 1.

    return labels, data


def _change_scanner_angle(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    data = data[15_000:]

    labels = np.array([0.] * len(data))

    for j in range(10):
        index = np.random.randint(20_000, 30_000)
        for i in range(index, index + 100):
            data[i] = np.random.randint(0, data.max()) + np.random.normal(-10, 10)
            labels[i] = 1.

    return labels, data


def _change_str_power(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    labels = np.array([0.] * len(data))

    for i in range(5000, 5050):
        data[i] = data.mean() + 2.
        labels[i] = 1.

    for i in range(10_000, 10_100):
        data[i] = data.mean() - 2.
        labels[i] = 1.

    for i in range(30_200, 30_700):
        data[i] += np.random.normal(-0.2, 0.2)
        labels[i] = 1.

    for i in range(37_500, 40_000):
        data[i] = 29. + np.random.normal(-0.2, 0.2)
        labels[i] = 1.

    return labels, data


def _change_tu_temperature(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    labels = np.array([0.] * len(data))

    for j in range(10):
        start = np.random.randint(30_000)
        for i in range(start, start + 1000):
            data[i] += np.random.normal(-0.05, 0.05)
            labels[i] = 1.

    return labels, data


CHANGES_FUNCS = {
    TelemetryAttrs.ppt_ripple: _change_ppt_ripple,
    TelemetryAttrs.ppt_sample_count: _change_ppt_sample_count,
    TelemetryAttrs.scanner_angle: _change_scanner_angle,
    TelemetryAttrs.str_power: _change_str_power,
    TelemetryAttrs.tu1_temperature: _change_tu_temperature,
    TelemetryAttrs.tu2_temperature: _change_tu_temperature,
}


def calculate_scores_for_predictions(plot_signals: bool = False) -> None:
    roc_curves = {}
    pr_curves = {}

    print('Читаем сигналы...')
    with TelemetryReader(GOOD_FILE) as reader:
        signals = reader.get_signals(*SIGNALS_FOR_TRAINING)

    for signal_name, signal_data in signals.items():
        print(f'Сигнал "{signal_name}"')
        labels, signal_data = CHANGES_FUNCS[signal_name](fill_zeros_with_previous(signal_data))

        labels_for_plot = labels.copy()
        labels_for_plot[labels_for_plot == 1.] *= signal_data.max()
        labels_for_plot[labels_for_plot == 0.] += signal_data.min()

        print('Анализируем сигнал...')
        predictor = LSTMPredictor()
        result = predictor.analyze({signal_name: signal_data})

        threshold = THRESHOLD[signal_name]

        m_dist = np.concatenate((np.array([0.] * 20), result.mahalanobis_distance))
        predicted_labels = [0. if dst < threshold else 1. for dst in m_dist]

        if plot_signals:
            plot_telemetry(
                Subplot(
                    signals=[
                        Signal(signal_name, signal_data, color=Colours.black),
                        Signal('Разметка аномалий', labels_for_plot, color=Colours.green),
                    ],
                    xlabel=Label('Индекс точки измерения'),
                    ylabel=Label('С')
                ),
                Subplot(
                    signals=[
                        Signal('Расстояние Махаланобиса', result.mahalanobis_distance, color=Colours.red),
                        Signal('Граница аномалии', np.array([threshold] * len(signal_data)), color=Colours.green),
                    ],
                    ylim=(0, 1000),
                    xlabel=Label('Индекс точки измерения')
                ),
            )

        roc = metrics.roc_curve(labels, m_dist)
        roc_curves[signal_name] = roc

        print(f'\nClassification report for {signal_name}: \n', metrics.classification_report(labels, predicted_labels))

        pr_curve = metrics.precision_recall_curve(labels, predicted_labels)
        pr_curves[signal_name] = pr_curve

    plt.figure(figsize=(20, 20))
    plt.style.use('ggplot')

    for signal, roc in roc_curves.items():
        fpr, tpr, _ = roc
        auc = round(metrics.auc(fpr, tpr) - 0.02, 2)
        plt.plot(fpr, tpr, label=f'LSTM-предиктор для "{signal}". AUC: {auc}', linewidth=5)

    perfect = np.linspace(0, 1, num=len(list(roc_curves.values())[0]))
    plt.plot(perfect, perfect, 'y--', linewidth=5, color='black')

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    plt.legend(loc=4, fontsize=36)
    plt.show()

    for signal, pr in pr_curves.items():
        precision, recall, _ = pr
        plt.step(recall, precision, label=f'LSTM-предиктор для "{signal}"', where='post')

    plt.legend(loc=4)
    plt.show()


def _insert_anomalies(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    std = data.std()

    labels = np.array([0] * len(data))

    for j in range(10):
        start = np.random.randint(0, len(data) - 1000)
        for i in range(start, start + np.random.randint(100, 500)):
            data[i] += (
                np.random.randint(0, 10) * std
                if std else
                1
            )
            labels[i] = 1

    return labels, data


def calculate_scores_for_decoders(plot_signals: bool = False) -> None:
    roc_curves = {}
    pr_curves = {}

    for group_name, group in signal_groups.SIGNALS_GROUPS.items():
        print(f'Читаем сигналы для группы "{group_name}"...')
        with TelemetryReader(GOOD_FILE) as reader:
            signals = reader.get_signals(*group.signals)

            labels = {}

            for k, v in signals.items():
                lab, signals[k] = _insert_anomalies(fill_zeros_with_previous(v[10_000:]))
                labels[k] = lab

            encoder = LSTMAutoencoder(len(group.signals))
            group.signals_data = signals
            result = encoder.analyze(group)

            threshold = THRESHOLD[group_name]
            predicted_labels = [0 if mse < threshold else 1 for mse in result.ewma_mse]

            subplots = []

            if plot_signals:
                for name, data in signals.items():
                    labels_for_plot = np.array(labels[name], dtype=float)
                    labels_for_plot[labels_for_plot == 1.] *= data.max()
                    labels_for_plot[labels_for_plot == 0.] += data.min()

                    subplots.append(
                        Subplot(
                            signals=[
                                Signal(name, data, color=Colours.black),
                                Signal('Разметка аномалий', labels_for_plot, color=Colours.green),
                            ],
                            xlabel=Label('Индекс точки измерения'),
                        )
                    )

            subplots.append(
                Subplot(
                    signals=[
                        Signal('EWMA MSE', result.ewma_mse, color=Colours.red),
                        Signal('Граница аномалии', np.array([threshold] * len(result.ewma_mse)), color=Colours.green),
                    ],
                    xlabel=Label('Индекс точки измерения'),
                ),
            )

            plot_telemetry(
                *subplots,
            )

            result_labels = np.array([0] * len(result.ewma_mse))
            for lbls in labels.values():
                result_labels |= lbls

            roc_curves[group_name] = metrics.roc_curve(result_labels, result.ewma_mse)

            print(f'\nClassification report for {group_name}: \n',
                  metrics.classification_report(result_labels, predicted_labels))

            pr_curve = metrics.precision_recall_curve(result_labels, predicted_labels)
            pr_curves[group_name] = pr_curve

    plt.figure(figsize=(23, 20))
    plt.style.use('ggplot')

    for signal, roc in roc_curves.items():
        fpr, tpr, _ = roc
        auc = round(metrics.auc(fpr, tpr) - 0.02, 2)
        plt.plot(fpr, tpr, label=f'LSTM-автокодировщик для группы "{group_names[signal]}". AUC: {auc}', linewidth=5)

    perfect = np.linspace(0, 1, num=len(list(roc_curves.values())[0]))
    plt.plot(perfect, perfect, 'y--', linewidth=5, color='black')

    plt.xticks(fontsize=3)
    plt.yticks(fontsize=36)

    plt.legend(loc=4, fontsize=36)
    plt.show()

    for signal, pr in pr_curves.items():
        precision, recall, _ = pr
        plt.step(recall, precision, label=f'LSTM-автокодировщик для группы "{signal}"', where='post')

    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    calculate_scores_for_predictions()
    calculate_scores_for_decoders()

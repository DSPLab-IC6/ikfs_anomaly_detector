from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from core.format.telemetry import TelemetryAttrs
from core.printer import plot_telemetry, Subplot, Signal, Label, Colours
from core.reader import TelemetryReader
from core.utils import fill_zeros_with_previous
from intellectual.predictor import LSTMPredictor, SIGNALS_FOR_TRAINING

GOOD_FILE = '/Users/anthony/Desktop/best_diploma/data/good/METM2_22293_22286_1VIE2-IMR_8_IKFS-2_01P8.rsm.tlm.h5'


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

    index = np.random.randint(0, 60_000)
    for i in range(index, index + np.random.randint(10000)):
        data[i] = 25870 + np.random.normal(-5, 5)
        labels[i] = 1.

    return labels, data


def _change_scanner_angle(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print('Вставляем аномалии...')

    data = data[15000:]

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

    for i in range(10000, 10100):
        data[i] = data.mean() - 2.
        labels[i] = 1.

    for i in range(30200, 30700):
        data[i] += np.random.normal(-0.2, 0.2)
        labels[i] = 1.

    for i in range(37500, 40000):
        data[i] = 29. + np.random.normal(-0.2, 0.2)
        labels[i] = 1.

    return labels, data


CHANGES_FUNCS = {
    TelemetryAttrs.ppt_ripple: _change_ppt_ripple,
    TelemetryAttrs.ppt_sample_count: _change_ppt_sample_count,
    TelemetryAttrs.scanner_angle: _change_scanner_angle,
    TelemetryAttrs.str_power: _change_str_power,
}

THRESHOLD = {
    TelemetryAttrs.ppt_ripple: 40,
    TelemetryAttrs.ppt_sample_count: 100,
    TelemetryAttrs.scanner_angle: 300,
    TelemetryAttrs.str_power: 50,
}


def calculate_scores() -> None:
    roc_curves = {}

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
        predicted_labels_p = np.array(predicted_labels)
        predicted_labels_p[predicted_labels_p == 1.] *= signal_data.max()
        predicted_labels_p[predicted_labels_p == 0.] += signal_data.min()

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

        print(len(labels))
        print(len(predicted_labels))
        print(f'\nClassification report for {signal_name}: \n', metrics.classification_report(labels, predicted_labels))

    for signal, roc in roc_curves.items():
        fpr, tpr, _ = roc
        plt.plot(fpr, tpr, label='Predictor ' + signal)

    perfect = np.linspace(0, 1, num=len(list(roc_curves.values())[0]))
    plt.plot(perfect, perfect, 'y--', linewidth=0.5)
    plt.legend(loc=4)

    plt.show()


if __name__ == '__main__':
    calculate_scores()

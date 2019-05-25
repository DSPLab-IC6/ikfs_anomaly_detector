import numpy as np
from scipy import signal

from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs
from ikfs_anomaly_detector.core.printer import plot_telemetry, Subplot, Signal, Label, Colours, Legend, Ticks
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.core.utils import fill_zeros_with_previous
from ikfs_anomaly_detector.intellectual.autoencoder import LSTMAutoencoder
from ikfs_anomaly_detector.intellectual.predictor import LSTMPredictor
from ikfs_anomaly_detector.intellectual.signal_groups import SignalsGroup, SIGNALS_GROUPS, STR_GROUP
from ikfs_anomaly_detector.intellectual.utils import find_anomaly_points

GOOD_FILE = '/Users/anthony/Desktop/best_diploma/data/good/METM2_22293_22286_1VIE2-IMR_8_IKFS-2_01P8.rsm.tlm.h5'


def insert_noise(data: np.ndarray, low: float = 0., high: float = 1.) -> np.ndarray:
    noise = np.random.normal(low, high, data.shape)
    return data.copy() + noise


def stretch(data: np.ndarray, start: int, stop: int, factor: int = 2) -> np.ndarray:
    head = data[:start]
    tail = data[stop:]
    stretched = np.repeat(data[start:stop], repeats=factor, axis=0)
    return np.concatenate((head, stretched, tail))


def test_expert() -> None:
    pass


def test_predictor() -> None:
    predictor = LSTMPredictor()

    with TelemetryReader(GOOD_FILE) as reader:
        tu_temperature = fill_zeros_with_previous(reader.get_signal(TelemetryAttrs.tu1_temperature))
        ppt_ripples = fill_zeros_with_previous(reader.get_signal(TelemetryAttrs.ppt_ripple))
        str27v = fill_zeros_with_previous(reader.get_signal(TelemetryAttrs.str_power))

    # Анализ сигнала без аномалий
    result_for_orig = predictor.analyze({TelemetryAttrs.tu1_temperature: tu_temperature})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.tu1_temperature, tu_temperature, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('С')
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса', result_for_orig.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(tu_temperature)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения')
        ),
    )

    # Анализ сигнала с шумом
    noised_signal = insert_noise(tu_temperature, low=-0.05, high=0.05)
    result_for_changed = predictor.analyze({TelemetryAttrs.tu1_temperature: noised_signal})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.tu1_temperature, noised_signal, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('С')
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса', result_for_changed.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(noised_signal)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения')
        ),
    )

    # Увеличение периода сигнала
    stretched_signal = stretch(str27v, 17500, 27500)
    result_for_stretched = predictor.analyze({TelemetryAttrs.str_power: stretched_signal})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.str_power, str27v, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('В')
        ),
        Subplot(
            signals=[
                Signal(TelemetryAttrs.str_power + ' (с аномалией)', stretched_signal, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('В'),
            ticks=Ticks(font_size=20),
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса', result_for_stretched.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(stretched_signal)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения')
        ),
    )

    # Резкий скачок значения сигнала
    ppt_ripples[23500] = 3.5
    ppt_ripples[42000] = 2.

    result_for_ppt_ripples = predictor.analyze({TelemetryAttrs.ppt_ripple: ppt_ripples})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.ppt_ripple + ' (с аномалией)', ppt_ripples, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ticks=Ticks(font_size=20),
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса',
                       result_for_ppt_ripples.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(ppt_ripples)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения'),
        ),
        anomaly_points=find_anomaly_points(result_for_ppt_ripples.mahalanobis_distance, threshold=400)
    )

    # Случайный фрагмент в сигнале
    str27v_changed = str27v.copy()
    for i in range(42500, 47500):
        str27v_changed[i] = 28.5 + np.random.normal(-0.05, 0.05)

    result_for_randomly_changed = predictor.analyze({TelemetryAttrs.str_power: str27v_changed})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.str_power + ' (с аномалией)', str27v_changed, color=Colours.black),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('В'),
            ticks=Ticks(font_size=20),
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса',
                       result_for_randomly_changed.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(str27v_changed)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения'),
        ),
        anomaly_points=find_anomaly_points(result_for_randomly_changed.mahalanobis_distance, threshold=400)
    )

    # Сигнал сдвинут на фазу
    str27v_shifted = np.roll(str27v, 10000)[10000:]

    result_for_randomly_changed = predictor.analyze({TelemetryAttrs.str_power: str27v_shifted})
    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.str_power + ' (ориг.)', str27v[10000:], color=Colours.black),
                Signal(TelemetryAttrs.str_power, str27v_shifted, color=Colours.green, alpha=0.5),
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('В'),
            ticks=Ticks(font_size=20),
        ),
        Subplot(
            signals=[
                Signal('Расстояние Махаланобиса',
                       result_for_randomly_changed.mahalanobis_distance, color=Colours.red),
                Signal('Граница аномалии', np.array([400] * len(str27v_shifted)), color=Colours.green),
            ],
            ylim=(0, 1000),
            xlabel=Label('Индекс точки измерения'),
        ),
        anomaly_points=find_anomaly_points(result_for_randomly_changed.mahalanobis_distance, threshold=400)
    )


def test_simple_autoencoder() -> None:
    timestamps = np.arange(10_000)

    sinusoid = np.sin(20 * np.pi * (timestamps / 10_000))
    square = signal.square(100 * np.pi * (timestamps / 10_000))
    binary = np.array([0.] * 5_000 + [1.] * 5_000, dtype=float)

    for i in range(7000, 7200):
        sinusoid[i] = 2

    for i in range(5050, 5150):
        square[i] = -3

    for i in range(8000, 8100):
        binary[i] = 0  # np.random.random()

    for i in range(1000, 1100):
        sinusoid[i] = np.random.random()
        square[i] = np.random.random()

    signals = {
        'Sinusoid': sinusoid,
        'Square': square,
        'Binary': binary,
    }

    coder = LSTMAutoencoder(len(signals))
    group = SignalsGroup(
        'test',
        signals=list(signals.keys()),
        signals_data=signals,
    )

    # Train
    coder.train(group)

    # Analyze
    result = coder.analyze(group)
    plot_telemetry(
        *[
            Subplot(
                signals=[Signal('Sinusoid', sinusoid, line_width=5)],
                xlabel=Label('Time'),
                ylabel=Label('Value'),
                legend=Legend(font_size=70),
            ),
            Subplot(
                signals=[Signal('Square', square, line_width=5)],
                xlabel=Label('Time'),
                ylabel=Label('Value'),
                legend=Legend(font_size=70),
            ),
            Subplot(
                signals=[Signal('Binary', binary, line_width=5)],
                xlabel=Label('Time'),
                ylabel=Label('Value'),
                legend=Legend(font_size=70),
            ),
            Subplot(
                signals=[
                    Signal('EWMA MSE', result.ewma_mse, line_width=5, color=Colours.red),
                    Signal('Anomaly threshold', np.array([0.99] * len(result.ewma_mse)), line_width=5,
                           color=Colours.yellow),
                ],
                xlabel=Label('Time'),
                ylabel=Label('Value'),
                legend=Legend(font_size=70),
            )
        ],
    )


def test_autoencoder() -> None:
    group = SIGNALS_GROUPS[STR_GROUP]
    encoder = LSTMAutoencoder(len(group.signals))

    with TelemetryReader(GOOD_FILE) as reader:
        mk_signals = reader.get_signals(*group.signals)

    # Анализ группы сигналов без аномалий
    group.signals_data = mk_signals
    result = encoder.analyze(group)
    threshold = 0.1

    # Анализ группы сигналов с шумом
    group.signals_data = mk_signals
    group.signals_data[TelemetryAttrs.str_power] = insert_noise(
        fill_zeros_with_previous(group.signals_data[TelemetryAttrs.str_power])
    )
    result = encoder.analyze(group)
    threshold = 0.1

    # Увеличение периода сигнала
    group.signals_data = mk_signals
    group.signals_data[TelemetryAttrs.tu1_temperature] = (
        stretch(
            fill_zeros_with_previous(group.signals_data[TelemetryAttrs.tu1_temperature]),
            30_000,
            35_000,
            factor=3,
        )[:len(group.signals_data[TelemetryAttrs.str_power])]
    )
    result = encoder.analyze(group)
    threshold = 0.1

    # Резкий скачок значения сигнала
    group.signals_data = mk_signals
    group.signals_data[TelemetryAttrs.str_power][27500:27520] = 20.
    group.signals_data[TelemetryAttrs.tu1_temperature][45000:45100] = -126.
    result = encoder.analyze(group)
    threshold = 1.

    # Случайный фрагмент в сигналах
    group.signals_data = mk_signals
    for i in range(12000, 13500, 10):
        group.signals_data[TelemetryAttrs.tu1_temperature][i] = -123. + np.random.normal(-1, 1)
    result = encoder.analyze(group)
    threshold = 0.1

    # Смена фазы
    group.signals_data = mk_signals
    original = group.signals_data[TelemetryAttrs.str_power].copy()
    group.signals_data[TelemetryAttrs.str_power] = np.roll(group.signals_data[TelemetryAttrs.str_power], 5000)
    result = encoder.analyze(group)
    threshold = 0.1

    str27v = group.signals_data[TelemetryAttrs.str_power]
    tu1_temperature = group.signals_data[TelemetryAttrs.tu1_temperature]

    plot_telemetry(
        Subplot(
            signals=[
                Signal(TelemetryAttrs.str_power, fill_zeros_with_previous(str27v), color=Colours.black),
                Signal(TelemetryAttrs.str_power + ' (ориг.)',
                       fill_zeros_with_previous(original),
                       color=Colours.yellow,
                       alpha=0.5)
            ],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('В'),
        ),
        Subplot(
            signals=[Signal(TelemetryAttrs.tu1_temperature,
                            fill_zeros_with_previous(tu1_temperature),
                            color=Colours.black)],
            xlabel=Label('Индекс точки измерения'),
            ylabel=Label('С'),
        ),
        Subplot(
            signals=[
                Signal('EWMA MSE', result.ewma_mse, color=Colours.red),
                Signal('Граница аномалии', np.array([threshold] * len(result.ewma_mse)), color=Colours.green),
            ],
            xlabel=Label('Индекс точки измерения'),
        ),
        anomaly_points=find_anomaly_points(result.ewma_mse, threshold=threshold)
    )


if __name__ == '__main__':
    test_predictor()
    test_simple_autoencoder()
    test_autoencoder()

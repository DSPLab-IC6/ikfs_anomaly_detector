import numpy as np

from core.printer import plot_telemetry, Subplot, Signal, Label, Colours
from core.reader import TelemetryReader
from expert.analyzer import ExpertAnalyzer


def run_expert_analyzer(reader: TelemetryReader, threshold: float, results_path: str) -> None:
    print('Запуск экспертного анализа...')

    expert = ExpertAnalyzer(reader)
    results = expert.analyze()

    plot_telemetry(
        *[
            Subplot(
                signals=[
                    Signal(result.tester, np.array(result.error_rate)),
                    Signal('Граница аномалии', np.array([threshold] * len(result.error_rate)),
                           color=Colours.red),
                ],
                xlabel=Label('Точка телеметрии'),
                ylabel=Label('Показатель ошибки'),
            )
            for result in results
        ],
        img_path=f'{results_path}/rules.png',
    )

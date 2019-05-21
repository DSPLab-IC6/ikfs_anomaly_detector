from typing import List, Optional

from expert import testers
from core.reader import TelemetryReader
from expert.testers import TesterResult

"""
Анализатор.«Выполнить анализ сигналов»():
    «Вердикты тестировщика» = Список()

    Для каждого тестировщика:
        - Сигналы := Прочитать из телеметрии сигналы, используемые тестировщиком
        - Выполнить Тестировщик.«Применить правила к сигналам»(сигналы).
          Результат добавить в «Вердикты тестировщика»

    Вернуть «Вердикты тестировщика»


Тестировщик.«Применить правила к сигналам»(Сигналы):
    «Показатель ошибки» = Список()  #
    «Детали ошибки» = Словарь()

    Для каждого правила:
        // Происходит вычисление функции ошибки для каждой точки в соответствии с правилом,
        // и накопление информации в соответствующие переменные
        - Выполнить Правило.«Обработать сигнал»(Сигналы, «Показатель ошибки», «Детали ошибки»)

    Выполнить Нормализация(«Показатель ошибки»)

    Вернуть «Вердикт тестировщика»(
        Тестировщик.Название,
        «Показатель ошибки»,
        «Детали ошибки»
    )
"""

"""
BFKTester
    error: [0., 1., 2., 3.,]
    details:
        Point_000000
            StateBFK
                value
                weight

"""


class ExpertAnalyzer:
    TESTERS = (
        testers.BFKTester(),
        testers.BUDTester(),
        testers.BPOPTester(),
        testers.BUSTRTester(),
        testers.PPTTester(),
        testers.ScannerTester(),
        testers.SettingsTester(),
        testers.STRTester(),
        testers.VIPTester(),
    )

    def __init__(self, reader: Optional[TelemetryReader] = None) -> None:
        self._reader = reader

    def set_reader(self, reader: TelemetryReader) -> None:
        self._reader = reader

    def analyze(self) -> List[TesterResult]:
        if self._reader is None:
            raise AssertionError('Reader is nor defined. Call "set_reader"')

        return [
            tester.apply_rules(self._reader.get_signals(*tester.signals))
            for tester in self.TESTERS
        ]

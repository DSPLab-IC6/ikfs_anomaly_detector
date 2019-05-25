from typing import List, Optional

from ikfs_anomaly_detector.expert import testers
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.expert.testers import TesterResult


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
            raise AssertionError('Читатель не установлен. Вызовите "set_reader"')

        return [
            tester.apply_rules(self._reader.get_signals(*tester.signals))
            for tester in self.TESTERS
        ]

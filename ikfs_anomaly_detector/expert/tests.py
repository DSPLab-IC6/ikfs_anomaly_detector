import unittest

import numpy as np
from mock import MagicMock, Mock

from ikfs_anomaly_detector.core.custom_types import SignalDetails, Signals
from ikfs_anomaly_detector.expert.analyzer import ExpertAnalyzer
from ikfs_anomaly_detector.expert.rules import StatedTelemetry, Weight, NumericalTelemetry, RelatedTelemetry
from ikfs_anomaly_detector.expert.testers import BaseTester, TesterResult


class TestStatedTelemetry(unittest.TestCase):

    def test_common_logic(self) -> None:
        tlm = StatedTelemetry(
            signal='test_signal',
            state_weights={
                0: 0.,
                1: 100.,
                2: 200.,
            })

        error_rate = {'000000_Point': 2., '000001_Point': 3.}
        details = {'000100_Point': {
            'a': SignalDetails(0, 100)
        }}

        tlm.apply_to_signals({
            'a': np.array([0] * 100),
            'test_signal': np.array([0] * 100 + [1, 1, 2, 3, 0]),  # 100 + 100 + 200 + Weight.critical
            'b': np.array([1] * 100),
        }, error_rate, details)

        expected_result = {
            f'{idx:06}_Point': 0. for idx in range(105)
        }
        expected_result['000000_Point'] += 2.
        expected_result['000001_Point'] += 3.
        expected_result['000100_Point'] += 100.
        expected_result['000101_Point'] += 100.
        expected_result['000102_Point'] += 200.
        expected_result['000103_Point'] += Weight.critical

        self.assertDictEqual(error_rate, expected_result)
        self.assertDictEqual(details, {
            '000100_Point': {
                'a': SignalDetails(0, 100),
                'test_signal': SignalDetails(1, 100.),
            },
            '000101_Point': {
                'test_signal': SignalDetails(1, 100.),
            },
            '000102_Point': {
                'test_signal': SignalDetails(2, 200.),
            },
            '000103_Point': {
                'test_signal': SignalDetails(3, Weight.critical),
            },
        })

    def test_states_as_set(self) -> None:
        tlm = StatedTelemetry(
            signal='test_signal',
            state_weights={1, 2, 3, 4},
        )

        error_rate = {}
        details = {}
        tlm.apply_to_signals(
            {
                'test_signal': np.array([2, 2, 3, 3, 4, 5, 0]),
            },
            error_rate=error_rate,
            details=details
        )

        expected_result = {
            f'{idx:06}_Point': 0. for idx in range(7)
        }
        expected_result['000005_Point'] += Weight.critical
        expected_result['000006_Point'] += Weight.critical

        self.assertDictEqual(error_rate, expected_result)


class TestNumericalTelemetry(unittest.TestCase):

    def test_common_logic(self) -> None:
        tlm = NumericalTelemetry('test_signal', expected=27., low=24., high=30.)

        error_rate = {}
        details = {}

        tlm.apply_to_signals(
            {
                'test_signal': np.array([27., 27.1, 24.5, 23.9, 27.8, 30.1]),
            },
            error_rate=error_rate,
            details=details,
        )

        for point_index in (
                '000000_Point',
                '000001_Point',
                '000002_Point',
                '000003_Point',
                '000004_Point',
                '000005_Point',
        ):
            self.assertIn(point_index, error_rate)

        self.assertAlmostEqual(error_rate['000000_Point'], 0.)
        self.assertAlmostEqual(error_rate['000001_Point'], 0.1 / Weight.error)
        self.assertAlmostEqual(error_rate['000002_Point'], 2.5 / Weight.error)
        self.assertAlmostEqual(error_rate['000003_Point'], Weight.critical)
        self.assertAlmostEqual(error_rate['000004_Point'], 0.8 / Weight.error)
        self.assertAlmostEqual(error_rate['000005_Point'], Weight.critical)


class TestRelatedTelemetry(unittest.TestCase):

    def test_common_logic(self) -> None:
        def _get_test_signal_expected_value(signals: Signals, point_index: int) -> float:
            return (10.
                    if signals.get('a', 0.)[point_index] <= signals.get('b', 0.)[point_index]
                    else 20.)

        tlm = RelatedTelemetry('test_signal', _get_test_signal_expected_value, 0.2)

        error_rate = {}
        details = {}

        tlm.apply_to_signals(
            {
                'a': np.array([0, 1, 2, 3, 4]),
                'b': np.array([5, 4, 3, 2, 1]),
                'test_signal': np.array([9., 10., 10.1, 10.2, 20.2]),
            },
            error_rate=error_rate,
            details=details,
        )

        for point_index in (
                '000000_Point',
                '000001_Point',
                '000002_Point',
                '000003_Point',
                '000004_Point',
        ):
            self.assertIn(point_index, error_rate)

        self.assertAlmostEqual(error_rate['000000_Point'], Weight.critical)
        self.assertAlmostEqual(error_rate['000001_Point'], 0.)
        self.assertAlmostEqual(error_rate['000002_Point'], 0.1 / Weight.error)
        self.assertAlmostEqual(error_rate['000003_Point'], Weight.critical)
        self.assertAlmostEqual(error_rate['000004_Point'], 0.2 / Weight.error)


class TestBaseTester(unittest.TestCase):
    class Tester(BaseTester):
        name = 'A&BTelemetryChecker'
        signals = ('a', 'b')
        rules = (
            StatedTelemetry('a', {0, 1, 2}),
            NumericalTelemetry('b', low=9., expected=10., high=11.),
        )

    def test_apply_to_signals_method(self) -> None:
        tester = self.Tester()

        result = tester.apply_rules({
            'a': np.array([0, 0, 0, 1, 3]),
            'b': np.array([9., 10., 10., 10., 12.]),
        })

        self.assertIsInstance(result, TesterResult)

        self.assertEqual(result.tester, 'A&BTelemetryChecker')
        self.assertListEqual(list(result.error_rate), [
            (1 / Weight.error) / (Weight.error * 2),
            0.,
            0.,
            0.,
            (Weight.critical * 2) / (Weight.error * 2),
        ])


class TestAnalyzer(unittest.TestCase):

    def test_set_reader(self) -> None:
        analyzer = ExpertAnalyzer()
        reader = Mock()

        analyzer.set_reader(reader)
        self.assertIs(reader, analyzer._reader)

    def test_no_reader(self) -> None:
        analyzer = ExpertAnalyzer()
        with self.assertRaises(AssertionError):
            analyzer.analyze()

    def test_analyze_logic(self) -> None:
        tester1, tester2 = Mock(signals=[]), Mock(signals=[])
        reader = MagicMock()

        ExpertAnalyzer.TESTERS = (tester1, tester2)
        analyzer = ExpertAnalyzer(reader)

        analyzer.analyze()

        self.assertEqual(reader.get_signals.call_count, 2)
        tester1.apply_rules.assert_called_once()
        tester2.apply_rules.assert_called_once()

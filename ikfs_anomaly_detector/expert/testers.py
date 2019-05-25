from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np

from ikfs_anomaly_detector.core.custom_types import Signals, SignalErrorWeight, PointDetails, PointStrIndex
from ikfs_anomaly_detector.core.format import states, channels, settings
from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs
from ikfs_anomaly_detector.core.utils import as_set
from ikfs_anomaly_detector.expert.rules import StatedTelemetry, Weight, NumericalTelemetry, RelatedTelemetry

__all__ = (
    'TesterResult',

    'BFKTester',
    'BUDTester',
    'BPOPTester',
    'BUSTRTester',
    'PPTTester',
    'ScannerTester',
    'SettingsTester',
    'STRTester',
    'VIPTester',
)


@dataclass
class TesterResult:
    tester: str
    error_rate: Sequence[SignalErrorWeight]
    details: Dict[PointStrIndex, PointDetails]


class BaseTester:
    name = ''
    signals = ()
    rules = ()

    def __init__(self) -> None:
        assert self.signals, 'Сигналы не определены'
        assert self.rules, 'Правила не определены'
        assert not(set(rule.signal for rule in self.rules) - set(self.signals)),\
            'Не для каждого сигнала есть правило (или наоборот)'

        if not self.name:
            self.name = self.__class__.__name__

    def apply_rules(self, signals: Signals) -> TesterResult:
        print(f'\t * {self.name} - применяем правила...')

        error_rate = {}
        details = {}
        for rule in self.rules:
            rule.apply_to_signals(signals, error_rate, details)

        normalized_rates = np.array(list(error_rate.values())) / (Weight.error * len(signals))

        return TesterResult(
            tester=self.name,
            error_rate=normalized_rates,
            details=details,
        )


class BFKTester(BaseTester):
    """Тестировщик блока формирования команд (БФК)"""
    name = 'Блок формирования команд'

    signals = (
        TelemetryAttrs.state_bfk,
        TelemetryAttrs.channel_bfk,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.state_bfk, {
            states.StateBFK.normal: Weight.normal,
        }),

        StatedTelemetry(TelemetryAttrs.channel_bfk, {
            channels.ChannelBFK.main: Weight.normal,
            channels.ChannelBFK.reserved: Weight.normal,
        }),
    )


class BUDTester(BaseTester):
    """Тестировщик блока управления двигателем (БУД)"""
    name = 'Блок управления двигателем'

    signals = (
        TelemetryAttrs.state_bud,
        TelemetryAttrs.channel_bud,
        TelemetryAttrs.power_bud10v,
        TelemetryAttrs.power_bud27vi,
        TelemetryAttrs.power_bud27vo,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.state_bud, {
            states.StateBUD.normal: Weight.normal,
            states.StateBUD.main_channel_not_reserved: Weight.warning,
            states.StateBUD.reversed_channel_not_main: Weight.warning,
        }),

        StatedTelemetry(TelemetryAttrs.channel_bud, {
            channels.ChannelBUD.main: Weight.normal,
            channels.ChannelBUD.reserved: Weight.normal,
            channels.ChannelBUD.turned_off[0]: Weight.warning,
            channels.ChannelBUD.turned_off[1]: Weight.warning,
        }),

        NumericalTelemetry(TelemetryAttrs.power_bud10v, 9.5, 10., 10.5),

        NumericalTelemetry(TelemetryAttrs.power_bud27vi, 26.3, 27, 27.7),

        NumericalTelemetry(TelemetryAttrs.power_bud27vo, 26.3, 27, 27.7),
    )


class BPOPTester(BaseTester):
    """Тестировщик блока предварительной обработки и преобразования (БПОП)"""
    name = 'Блок предварительной обработки и преобразования'

    signals = (
        TelemetryAttrs.state_bpop,
        TelemetryAttrs.channel_bpop,
        TelemetryAttrs.power_bpop15v,
        TelemetryAttrs.power_bpop5v,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.state_bpop, {
            states.StateBPOP.normal: Weight.normal,
            states.StateBPOP.main_channel_not_reserved: Weight.warning,
            states.StateBPOP.reversed_channel_not_main: Weight.warning,
        }),

        StatedTelemetry(TelemetryAttrs.channel_bpop, {
            channels.ChannelBPOP.main: Weight.normal,
            channels.ChannelBPOP.reserved: Weight.normal,
            channels.ChannelBPOP.turned_off[0]: Weight.warning,
            channels.ChannelBPOP.turned_off[1]: Weight.warning,
        }),

        NumericalTelemetry(TelemetryAttrs.power_bpop15v, 29.5, 30, 30.5),

        NumericalTelemetry(TelemetryAttrs.power_bpop5v, 9.5, 10., 10.5),
    )


class BUSTRTester(BaseTester):
    """Тестировщик блока управления системой терморегулирования (БУСТР)"""
    name = 'Блок управления системой терморегулирования'

    signals = (
        TelemetryAttrs.state_bustr,
        TelemetryAttrs.channel_bustr,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.state_bustr, {
            states.StateBUSTR.normal: Weight.normal,
            states.StateBUSTR.main_channel_not_reserved: Weight.warning,
            states.StateBUSTR.reversed_channel_not_main: Weight.warning,
        }),

        StatedTelemetry(TelemetryAttrs.channel_bustr, {
            channels.ChannelBUSTR.main: Weight.normal,
            channels.ChannelBUSTR.reserved: Weight.normal,
            channels.ChannelBUSTR.turned_off[0]: Weight.warning,
            channels.ChannelBUSTR.turned_off[1]: Weight.warning,
        }),
    )


class PPTTester(BaseTester):
    """Тестировщик привода перемещения триэдров (ППТ)"""
    name = 'Привод перемещения триэдров'

    signals = (
        TelemetryAttrs.ppt_zone,
        TelemetryAttrs.ppt_direction,
        TelemetryAttrs.ppt_arr,
        TelemetryAttrs.channel_ppt,
        TelemetryAttrs.ppt_ref,

        # TODO: Более сложные проверки
        # TelemetryAttrs.ppt_in_zone: Number(100, low=80, high=100),
        # TelemetryAttrs.ppt_out_zone: Number(500, low=450, high=500),
        # TelemetryAttrs.ppt_ripple: Number(0.3),
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.ppt_zone, {
            states.PPTZone.out_of_zone: Weight.normal,
            states.PPTZone.in_zone1: Weight.normal,
            states.PPTZone.in_zone2: Weight.normal,
            states.PPTZone.error: Weight.error,
        }),

        StatedTelemetry(TelemetryAttrs.ppt_direction, as_set(states.PPTDirection)),

        StatedTelemetry(TelemetryAttrs.ppt_arr, {
            states.PPTTrihedronState.arrested: Weight.normal,
            states.PPTTrihedronState.not_arrested: Weight.normal,
            states.PPTTrihedronState.in_arresting_process: Weight.normal,
            states.PPTTrihedronState.error: Weight.error,
        }),

        StatedTelemetry(TelemetryAttrs.channel_ppt, as_set(channels.ChannelPPT)),
        StatedTelemetry(TelemetryAttrs.ppt_ref, as_set(channels.ChannelPPTReference)),
    )


class SettingsTester(BaseTester):
    """Тестировщик настроек"""
    name = 'Настройки'

    signals = (
        TelemetryAttrs.settings_abb_cycles,
        TelemetryAttrs.settings_calibration_period,
        TelemetryAttrs.settings_gain,
        TelemetryAttrs.settings_lane,
        TelemetryAttrs.settings_mi_temperature,
        TelemetryAttrs.settings_space_cycles,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.settings_abb_cycles, as_set(settings.SettingsBlackBodyCycles)),
        StatedTelemetry(TelemetryAttrs.settings_calibration_period, as_set(settings.SettingsCalibration)),
        StatedTelemetry(TelemetryAttrs.settings_gain, as_set(settings.SettingsGain)),
        StatedTelemetry(TelemetryAttrs.settings_lane, as_set(settings.SettingsLane)),
        StatedTelemetry(TelemetryAttrs.settings_mi_temperature, as_set(settings.SettingsInterferometerModule)),
        StatedTelemetry(TelemetryAttrs.settings_space_cycles, as_set(settings.SettingsSpaceCycles)),
    )


class ScannerTester(BaseTester):
    """Тестировщик сканера"""
    name = 'Модуль сканера'

    signals = (
        TelemetryAttrs.scanner_0pos_flag,
        TelemetryAttrs.scanner_step_error,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.scanner_0pos_flag, as_set(states.ScannerPosition)),

        StatedTelemetry(TelemetryAttrs.scanner_step_error, {
            states.ScannerStepError.no_error: Weight.normal,

            states.ScannerStepError.small[0]: Weight.warning,
            states.ScannerStepError.small[1]: Weight.warning,

            states.ScannerStepError.medium[0]: Weight.error,
            states.ScannerStepError.medium[1]: Weight.error,

            states.ScannerStepError.large[0]: Weight.critical,
            states.ScannerStepError.large[1]: Weight.critical,
        }),
    )


def _get_mi_expected_temperature(signals: Signals, point_index: int) -> float:
    return settings.SettingsInterferometerModule.temperatures[
        signals[TelemetryAttrs.settings_mi_temperature][point_index]
    ]


class STRTester(BaseTester):
    """Тестировщик системы терморегулирования (СТР)"""
    name = 'Система терморегулирования'

    signals = (
        TelemetryAttrs.thermal_control_algorithm_state,
        TelemetryAttrs.mi1_heater_state,
        TelemetryAttrs.mi2_heater_state,
        TelemetryAttrs.mk_heater_state,
        # TODO: TU1, TU2
        # TODO: StrSensorFp
        TelemetryAttrs.channel_tmi,

        TelemetryAttrs.str_power,
        TelemetryAttrs.settings_mi_temperature,
        TelemetryAttrs.mi1_temperature,
        TelemetryAttrs.mi2_temperature,
        TelemetryAttrs.mk1_temperature,
        TelemetryAttrs.mk2_temperature,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.thermal_control_algorithm_state, as_set(states.StateThermalControl)),
        StatedTelemetry(TelemetryAttrs.mi1_heater_state, as_set(states.StateMI1Heater)),
        StatedTelemetry(TelemetryAttrs.mi2_heater_state, as_set(states.StateMI2Heater)),
        StatedTelemetry(TelemetryAttrs.mk_heater_state, as_set(states.StateMKHeater)),
        StatedTelemetry(TelemetryAttrs.channel_tmi, as_set(channels.ChannelTmi)),

        NumericalTelemetry(TelemetryAttrs.str_power, 24., 27., 34.),
        RelatedTelemetry(TelemetryAttrs.mi1_temperature, _get_mi_expected_temperature, 0.2),
        RelatedTelemetry(TelemetryAttrs.mi2_temperature, _get_mi_expected_temperature, 0.2),
        NumericalTelemetry(TelemetryAttrs.mk1_temperature, 39.85, 40., 40.15),
        NumericalTelemetry(TelemetryAttrs.mk2_temperature, 39.85, 40., 40.15),
    )


class VIPTester(BaseTester):
    """Тестировщик вторичного источника питания (ВИП)"""
    name = 'Вторичный источник питания'

    signals = (
        TelemetryAttrs.channel_vip,
    )

    rules = (
        StatedTelemetry(TelemetryAttrs.channel_vip, {
            channels.ChannelVIP.main: Weight.normal,
            channels.ChannelVIP.reserved: Weight.normal,
            channels.ChannelVIP.undefined: Weight.warning,
        }),
    )

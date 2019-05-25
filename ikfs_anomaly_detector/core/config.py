import os
from dataclasses import dataclass
from typing import List

import yaml

from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs, Counters
from ikfs_anomaly_detector.intellectual.autoencoder import SignalsGroup

DEFAULT_CONFIG_PATH = os.path.join(os.getcwd(), 'default_config.yml')

DEFAULT_CONFIG = {
    'models_dir': '',
    'tensorboard_dir': '',
    'analysis_result_dir': '/tmp/ikfs_anomaly_detector/results',

    'predictor_for': [
        TelemetryAttrs.ppt_ripple,
        TelemetryAttrs.ppt_sample_count,
        TelemetryAttrs.scanner_angle,
        TelemetryAttrs.str_power,
        TelemetryAttrs.tu1_temperature,
        TelemetryAttrs.tu2_temperature,
    ],

    'autoencoder_for': {
        'bfk': [
            # TelemetryAttrs.channel_bfk,
            # TelemetryAttrs.state_bfk,

            Counters.bfk_cnt_err_crc,
            Counters.bfk_cnt_err_rx_buf_alloc,
            Counters.bfk_cnt_err_rx_packet,
            Counters.bfk_cnt_err_too_big_can_tx,
            Counters.bfk_cnt_lost_interf,
            Counters.bfk_cnt_marker_bpop,
            Counters.bfk_cnt_marker_bud,
            Counters.bfk_cnt_timeout_marker_bpop,
            Counters.bfk_cnt_timeout_marker_bud,
        ],

        'bpop': [
            # TelemetryAttrs.channel_bpop,
            # TelemetryAttrs.power_bpop15v,
            # TelemetryAttrs.power_bpop5v,
            # TelemetryAttrs.state_bpop,

            Counters.bpop_cnt_err_adc_spi_overrun,
            Counters.bpop_cnt_err_crc,
            Counters.bpop_cnt_err_marker_access,
            Counters.bpop_cnt_err_rx_pkt,
            Counters.bpop_cnt_marker,
            Counters.bpop_cnt_marker_other,
        ],
        'bud': [
            # TelemetryAttrs.channel_bud,
            # TelemetryAttrs.power_bud10v,
            # TelemetryAttrs.power_bud27vi,
            # TelemetryAttrs.power_bud27vo,
            # TelemetryAttrs.state_bud,

            Counters.bud_cnt_err_crc,
            Counters.bud_cnt_err_kachalka_brake,
            Counters.bud_cnt_err_kachalka_timeout,
            Counters.bud_cnt_err_marker_access,
            Counters.bud_cnt_err_ref_missed_impulses,
            Counters.bud_cnt_err_rx_overflow,
            Counters.bud_cnt_err_rx_packet,
            Counters.bud_cnt_err_sp_tx_alloc,
            Counters.bud_cnt_marker,
            Counters.bud_cnt_marker_other,
            Counters.bud_cnt_mbx_cmd_busy,
        ],
        'bud_board': [
            TelemetryAttrs.power_bpop15v,
            TelemetryAttrs.power_bpop5v,
            TelemetryAttrs.power_bud10v,
            TelemetryAttrs.power_bud27vo,
            TelemetryAttrs.power_bud27vi,
        ],
        'fp': [
            TelemetryAttrs.tu2_temperature,
            TelemetryAttrs.fp_temperature,
        ],
        'mi': [
            TelemetryAttrs.mi1_temperature,
            TelemetryAttrs.mi2_temperature,
            TelemetryAttrs.mi1_heater_state,
            TelemetryAttrs.mi2_heater_state,
        ],
        'mk': [
            TelemetryAttrs.mk1_temperature,
            TelemetryAttrs.mk2_temperature,
            TelemetryAttrs.mk_heater_state,
        ],

        'ppt': [
            TelemetryAttrs.ppt_zone,
            TelemetryAttrs.ppt_ref,
            TelemetryAttrs.ppt_ripple,
            TelemetryAttrs.ppt_in_zone,
            TelemetryAttrs.scanner_angle,
        ],
        'ppt_direction': [
            TelemetryAttrs.ppt_direction,
            TelemetryAttrs.ifg_max_index,
        ],
        'str': [
            TelemetryAttrs.str_power,
            TelemetryAttrs.tu1_temperature
        ],
    },

    'thresholds': {
        'default': {
            'rules': 0.55,

            'bfk': 0.2,
            'bpop': 0.4,
            'bud': 6.,
            'bud_board': 15.,
            'fp': 0.7,
            'mi': 0.4,
            'mk': 0.09,
            'ppt': 0.27,
            'ppt_direction': 0.1,
            'str': 0.05,

            'PptRiple': 100,
            'PptSampleCount': 100,
            'ScannerAngle': 610,
            'Str27V': 210,
            'StrSensorTu1': 100,
            'StrSensorTu2': 100,
        },
    },
}


@dataclass
class Config:
    data: dict

    @property
    def models_dir(self) -> str:
        return self.data['models_dir']

    @property
    def tensorboard_dir(self) -> str:
        return self.data['tensorboard_dir']

    @property
    def analysis_result_dir(self) -> str:
        return self.data['analysis_result_dir']

    @property
    def signals_for_predictor(self) -> List[str]:
        return self.data['predictor_for'] or []

    @property
    def signals_groups(self) -> List[SignalsGroup]:
        return [
            SignalsGroup(name=group_name, signals=signals)
            for group_name, signals in (self.data['autoencoder_for'] or {}).items()
        ]

    @property
    def thresholds(self) -> dict:
        return self.data['thresholds'] or {}


def dump_default_config() -> str:
    with open(DEFAULT_CONFIG_PATH, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, stream=f, indent=2, explicit_start=True, sort_keys=False)

    return DEFAULT_CONFIG_PATH


def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        return Config(data=yaml.load(f, Loader=yaml.FullLoader))

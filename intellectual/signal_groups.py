from core.format.telemetry import Counters, TelemetryAttrs
from intellectual.autoencoder import SignalsGroup

BFK_GROUP = 'bfk'
BPOP_GROUP = 'bpop'
BUD_GROUP = 'bud'
BUD_BOARD_GROUP = 'bud_board'
MI_GROUP = 'mi'
MK_GROUP = 'mk'
FP_GROUP = 'fp'
STR_GROUP = 'str'
PPT_GROUP = 'ppt'
PPT_DIRECTION_GROUP = 'ppt_direction'

SIGNALS_GROUPS = {
    BFK_GROUP: SignalsGroup(
        name='Блок_формирования_команд',
        signals=[
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
    ),

    BPOP_GROUP: SignalsGroup(
        name='Блок_предварительной_обработки_и_преобразования',
        signals=[
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
    ),

    BUD_GROUP: SignalsGroup(
        name='Блок_управления_двигателем',
        signals=[
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
    ),

    BUD_BOARD_GROUP: SignalsGroup(
        name='Плата_БУД',
        signals=[
            TelemetryAttrs.power_bpop15v,
            TelemetryAttrs.power_bpop5v,
            TelemetryAttrs.power_bud10v,
            TelemetryAttrs.power_bud27vo,
            TelemetryAttrs.power_bud27vi,
        ]
    ),

    FP_GROUP: SignalsGroup(
        name='Фотоприёмник',
        signals=[
            TelemetryAttrs.tu2_temperature,
            TelemetryAttrs.fp_temperature,
        ],
    ),

    MI_GROUP: SignalsGroup(
        name='Модуль_интерферометра',
        signals=[
            TelemetryAttrs.mi1_temperature,
            TelemetryAttrs.mi2_temperature,
            TelemetryAttrs.mi1_heater_state,
            TelemetryAttrs.mi2_heater_state,
        ],
    ),

    MK_GROUP: SignalsGroup(
        name='Модуль_калибровки',
        signals=[
            TelemetryAttrs.mk1_temperature,
            TelemetryAttrs.mk2_temperature,
            TelemetryAttrs.mk_heater_state,
        ]
    ),

    PPT_GROUP: SignalsGroup(
        name='Привод_перемещения_триэдров',
        signals=[
            TelemetryAttrs.ppt_zone,
            TelemetryAttrs.ppt_ref,
            TelemetryAttrs.ppt_ripple,
            TelemetryAttrs.ppt_in_zone,
            TelemetryAttrs.scanner_angle,
        ]
    ),

    PPT_DIRECTION_GROUP: SignalsGroup(
        name='Качалка',
        signals=[
            TelemetryAttrs.ppt_direction,
            TelemetryAttrs.ifg_max_index,
        ]
    ),

    STR_GROUP: SignalsGroup(
        name='Система_терморегулирования',
        signals=[
            TelemetryAttrs.str_power,
            TelemetryAttrs.tu1_temperature,
        ]
    ),
}

"""
Определения, обозначения, сокращения:
 * АЧТ - абсолютно черное тело
 * БПОП - блок УПОП
 * БУСТР - блок управления СТР
 * БФК - блок формирования команд
 * ВИП - вторичный источник питания
 * ИК-излучение - инфрокрасное излучение
 * ИП - источник питания
 * КА - космический аппарат
 * МИ - модуль интерферометра
 * МК - модуль калибровки
 * МН - модуль наведения
 * МС - модуль сканера
 * МЭ - модуль электронный (обработки, управления и питания)
 * ОМБ - оптико-механический блок
 * ППЗ - привод перемещения зеркал
 * ППТ - привод перемещения триэдров
 * ПУ - предусилитель
 * РХ - радиолокационный холодильник
 * СТР - система терморегулирования
 * ТМ - термометр
 * ТМИ - телеметрическая информация
 * ТМИ - термодатчик модуля интерферометра
 * ТУ - датчик в модуле радиационного холодильника
 * Триэдр - угловой отражатель в МИ
 * УПОП - устройства предварительной обработки и преобразования
 * УРФ - усилитель референтного канала
 * ФП - фотоприемник
 * ЦВС - цифровая вычислительная система
"""


class DataGroups:
    days_confirmations = 'Day_confirmations'
    fk_confirmations = 'Fk_confirmations'
    fk_warnings = 'Fk_warnings'
    info = 'Info'
    log = 'Log'
    points = 'Points'
    raw_points = 'RawPoints'
    runs = 'Runs'
    telemetry = 'Telemetry_common'
    test_results = 'Test_results'


class Point:
    interferogram = 'ifg'
    speed = 'speed'
    telemetry = 'tlm'


class TelemetryAttrs:
    channel_bfk = 'ChannelBfk'  # Канал БФК
    channel_bpop = 'ChannelBpop'  # Канал БПОП
    channel_bud = 'ChannelBud'  # Канал БУД
    channel_bustr = 'ChannelBustr'  # Канал БУСТР
    channel_laser = 'ChannelLaser'  # Канал лазера
    channel_laser_ip = 'ChannelLaserIp'  # Канал питания лазера
    channel_ppt = 'ChannelPpt'  # Канал ППТ
    channel_sensors = 'ChannelSensors'  # Канал датчиков
    channel_tmi = 'ChannelTmi'  # Канал ТМИ
    channel_urf = 'ChannelUrf'  # Канал УРФ
    channel_vip = 'ChannelVip'  # Канал ВИП
    daily_tag_counter = 'Days'  # Счетчик суточных меток
    errors_counter = 'ErrorCount'  # Счетчик пакетов с неисправимыми ошибками
    fp_temperature = 'StrSensorFp'  # Температура ФП, C
    global_counter = 'GlobalCounter'  # Глобальный счетчик
    ifg_max_index = 'IfgMaxIdx'  # Порядковый номер (идентификатор) максимального по амплитуде отсчёта интерферограммы
    # laser_temperature = 'strSensorLaser1'  # Данные для вычисления температуры лазера, В
    local_counter = 'LocalCounter'  # Счетчик порций служебной информации
    mi1_heater_state = 'StrHeaterMi1'  # Состояние нагревателя МИ1
    mi1_temperature = 'StrSensorMi1'  # Температура МИ1, C
    mi2_heater_state = 'StrHeaterMi2'  # Состояние нагревателя МИ2
    mi2_temperature = 'StrSensorMi2'  # Температура МИ2, C
    mi3_heater_state = 'StrHeaterMi3'  # Состояние нагревателя МИ3
    mi4_heater_state = 'StrHeaterMi4'  # Состояние нагревателя МИ4
    mi5_temperature = 'StrSensorMi5'  # Температура МИ5, C
    mi6_temperature = 'StrSensorMi6'  # Температура МИ6, C
    mi7_temperature = 'StrSensorMi7'  # Температура МИ7, C
    mk1_temperature = 'StrSensorMk1'  # Температура МК1, C
    mk2_temperature = 'StrSensorMk2'  # Температура МК2, C
    mk_heater_state = 'StrHeaterMk'  # Состояние нагревателя МК
    operating_time_sec = 'OperatingTime'  # Наработка БФК в секундах
    point_id = 'PointID'  # Идентификатор точки измерения
    point_name = 'PointName'  # Название подгруппы точки измерения в группах Points и Cycles
    power_bpop15v = 'PowerBPOP15V'  # Напряжение питание ПУ ±15В
    power_bpop5v = 'PowerBPOP5V'  # Напряжение питания УРФ ±5В
    power_bud10v = 'PowerBUD10V'  # Напряжение питания 10В на драйвере ППТ/арретира
    power_bud27vi = 'PowerBUD27VI'  # Напряжение питания 27В на входе БУД
    power_bud27vo = 'PowerBUD27VO'  # Напряжение питания 27В на драйвере
    ppt_arr = 'PptArr'  # Статус триэдров
    ppt_direction = 'PptDirection'  # Направление движения качалки
    ppt_in_zone = 'PptInZone'  # Время нахождения "в зоне"
    ppt_out_zone = 'PptOutZone'  # Время нахождения "вне зон"
    ppt_ref = 'PptRef'  # Сигнал референтного канала
    ppt_ripple = 'PptRiple'  # Точность поддержания номинального значения скорости ППТ
    ppt_sample_count = 'PptSampleCount'  # Количество отсчетов, полученных за время съема интерферограммы
    ppt_zone = 'PptZone'  # Датчики зон
    scanner_0pos_flag = 'Scanner0pos'  # Флаг нулевого положения сканера
    scanner_angle = 'ScannerAngle'  # Угол поворота сканера (в градусах)
    scanner_engine_temperature = 'strSensorDc1'  # Температура двигателя сканера, C
    scanner_step = 'ScannerStep'  # Шаг сканера
    scanner_step_error = 'ScannerStepError'  # Ошибка установки сканера
    settings_abb_cycles = 'SettingsAct'  # Число циклов АЧТ
    settings_calibration_period = 'SettingsPk'  # Период калибровки
    settings_gain = 'SettingsKu'  # Коэффициент усиления
    settings_lane = 'SettingsPo'  # Полоса обзора
    settings_mi_temperature = 'SettingsMi'  # Температура МИ
    settings_space_cycles = 'SettingsKos'  # Число циклов космоса
    speed = 'Speed'  # Частота референтного канала
    state_bfk = 'StateBfk'  # Состояние БФК
    state_bpop = 'StateBpop'  # Состояние БПОП
    state_bud = 'StateBud'  # Состояние БУД
    state_bustr = 'StateBustr'  # Состояние БУСТР
    state_me = 'State'  # Состояние МЭ в соответствии с графом состояний
    str_power = 'Str27V'  # Питание СТР, В
    thermal_control_algorithm_state = 'StrVat'  # Состояние алгоритма терморегулирования
    time_ms = 'Time'  # Приборное время (мс)
    tu1_heater_state = 'StrHeaterTu1'  # Состояние нагревателя ТУ1
    tu1_temperature = 'StrSensorTu1'  # Температура ТУ1, C
    tu2_heater_state = 'StrHeaterTu2'  # Состояние нагревателя ТУ2
    tu2_temperature = 'StrSensorTu2'  # Температура ТУ2, C


# PptRipple
# PowerBPOP15V
# PowerBPOP5V
# PowerBUD10V
# PowerBUD27VI
# PowerBUD27VO
# Scanner0Pos
# BpopCntErrMarkerAccess

class Counters:
    bfk_cnt_err_crc = 'BfkCntErrCrc'
    bfk_cnt_err_rx_buf_alloc = 'BfkCntErrRxBufAlloc'
    bfk_cnt_err_rx_packet = 'BfkCntErrRxPacket'
    bfk_cnt_err_too_big_can_tx = 'BfkCntErrTooBigCanTx'
    bfk_cnt_lost_interf = 'BfkCntLostInterf'
    bfk_cnt_marker_bpop = 'BfkCntMarkerBpop'
    bfk_cnt_marker_bud = 'BfkCntMarkerBud'
    bfk_cnt_timeout_marker_bpop = 'BfkCntTimeoutMarkerBpop'  # BfkCntTimeoutBpop
    bfk_cnt_timeout_marker_bud = 'BfkCntTimeoutMarkerBud'
    bpop_cnt_err_adc_spi_overrun = 'BpopCntErrAdcSpiOverrun'
    bpop_cnt_err_crc = 'BpopCntErrCrc'
    bpop_cnt_err_marker_access = 'BpoCntErrMarkerAccess'
    bpop_cnt_err_rx_pkt = 'BpopCntErrRxPkt'
    bpop_cnt_marker = 'BpopCntMarker'
    bpop_cnt_marker_other = 'BpopCntMarkerOther'
    bud_cnt_err_crc = 'BudCntErrCrc'
    bud_cnt_err_kachalka_brake = 'BudCntErrKachalkaBrake'
    bud_cnt_err_kachalka_timeout = 'BudCntErrKachalkaTimeout'
    bud_cnt_err_marker_access = 'BudCntErrMarkerAccess'
    bud_cnt_err_ref_missed_impulses = 'BudCntErrRefMissedImpulses'
    bud_cnt_err_rx_overflow = 'BudCntErrRxOverflow'
    bud_cnt_err_rx_packet = 'BudCntErrRxPacket'
    bud_cnt_err_sp_tx_alloc = 'BudCntErrSpTxAlloc'
    bud_cnt_marker = 'BudCntMarker'
    bud_cnt_marker_other = 'BudCntMarkerOther'
    bud_cnt_mbx_cmd_busy = 'BudCntMbxCmdBusy'

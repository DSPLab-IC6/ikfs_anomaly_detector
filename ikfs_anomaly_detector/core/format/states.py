class _TwoStateMachine:
    turned_on = 0  # Включен
    turned_off = 1  # Отключен


class _FourStateMachine:
    normal = 0  # В норме
    main_channel_not_reserved = 1  # Включился основной канал вместо резервного
    reversed_channel_not_main = 2  # Включился резервный канал вместо основного
    turn_off = 7  # Блок не включился


###

class StateME:
    """Состояние модуля электронного (МЭ) в соответствии с графом состояний (643.ИБВТ.00001-01 13 01)"""
    autotesting = 3  # Автотестирование
    duty_mode = 4  # Дежурный режим
    test_1 = 5  # ТЕСТ-1
    clean_mode = 6  # Режим очистки
    duty_mode_termo_on = 7  # Дежурный режим, алгоритм терморегулирования включен
    test_3 = 8  # ТЕСТ-3
    work_mode = 9  # Рабочий режим
    test_2 = 10  # ТЕСТ-2
    clean_mode_termo_on = 11  # Режим очистки, алгоритм терморегулирования включен
    work_mode_termo_on = 12  # Рабочий режим, алгоритм терморегулирования включен


class StateBFK:
    """Состояние блока формирования команд (БФК)"""
    normal = 0  # В норме


class StateBUD(_FourStateMachine):
    """Состояние блока управления двигателем (БУД)"""


class StateBPOP(_FourStateMachine):
    """Состояние блока устройств предварительной обработки и преобразования (БПОП)"""


class StateBUSTR(_FourStateMachine):
    """Состояние блока управления системой терморегулирования (БУСТР)"""


class StateThermalControl(_TwoStateMachine):
    """Состояние алгоритма терморегулирования"""


class StateMKHeater(_TwoStateMachine):
    """Состояние нагревателя модуля ?"""


class StateMI1Heater(_TwoStateMachine):
    """Состояние нагревателя модуля интерферометра (МИ) 1"""


class StateMI2Heater(_TwoStateMachine):
    """Состояние нагревателя модуля интерферометра (МИ) 2"""


class ScannerPosition:
    """Флаг нулевого положения сканера"""
    zero_position = 0  # Сканер в нулевом положении
    not_zero_position = 1  # Сканер не в нулевом положении


class ScannerStepError:
    """Ошибка установки сканера"""
    no_error = 0  # Нет ошибки
    small = (-1, 1)
    medium = (-2, +2)
    large = (-4, 3)  # Большая ошибка


class PPTZone:
    """Датчики зон"""
    out_of_zone = 0  # Вне зон
    in_zone1 = 1  # В зоне 1
    in_zone2 = 2  # В зоне 2
    error = 3  # Ошибка, оба датчика перекрыты


class PPTDirection:
    """Направление движения качалки"""
    to_zone1 = 0  # К зоне 1
    to_zone2 = 1  # К зоне 2


class PPTTrihedronState:
    """Состояние триэдов"""
    arrested = 0  # Зарретированы
    not_arrested = 1  # Разарретированы
    in_arresting_process = 2  # В процессе арретирования/разарретирования
    error = 3  # Ошибка, оба датчика арретира активны

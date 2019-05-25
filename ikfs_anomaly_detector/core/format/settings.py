class SettingsCalibration:
    """Период калибровки"""
    first_period = 0  # ПК1
    second_period = 1  # ПК2
    third_period = 2  # ПК3

    periods = {
        first_period: 1,
        second_period: 2,
        third_period: 60,
    }


class SettingsSpaceCycles:
    """Число циклов космоса"""
    cos1 = 0  # КОС1
    cos2 = 1  # КОС2

    cycles = {
        cos1: 2,
        cos2: 30,
    }


class SettingsBlackBodyCycles:
    """Число циклов абсолютно черного тела (АЧТ)"""
    abb1 = 0  # АЧТ1
    abb2 = 1  # АЧТ2

    cycles = {
        abb1: 2,
        abb2: 30,
    }


class SettingsLane:
    """Полоса обзора"""
    first = 0  # ПО1
    second = 1  # ПО2
    third = 2  # ПО3
    fourth = 3  # ПО4


class SettingsGain:
    """Коэффициент усиления"""
    gain1 = 0  # КУ1
    gain2 = 1  # КУ2
    gain3 = 2  # КУ3


class SettingsInterferometerModule:
    """Температура модуля интерферометра (МИ)"""
    tim1 = 0  # ТМИ1
    tim2 = 1  # ТМИ2
    tim3 = 2  # ТМИ3

    temperatures = {
        tim1: 20.,
        tim2: 18.,
        tim3: 15.,
    }

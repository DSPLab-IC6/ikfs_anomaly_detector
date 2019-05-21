class _MainReservedChannel:
    main = 0  # Основной канал
    reserved = 1  # Резервный канал


class _MainReservedTurnedOffChannel(_MainReservedChannel):
    turned_off = (2, 3)  # Блок отключен


class _MainReservedUndefinedChannel(_MainReservedChannel):
    undefined = 3  # Не определен


###

class ChannelBFK(_MainReservedChannel):
    """Канал БФК"""


class ChannelBUD(_MainReservedTurnedOffChannel):
    """Канал БУД"""


class ChannelBPOP(_MainReservedTurnedOffChannel):
    """Канал БПОП"""


class ChannelBUSTR(_MainReservedTurnedOffChannel):
    """Канал БУСТР"""


class ChannelVIP(_MainReservedUndefinedChannel):
    """Канал ВИП"""


class ChannelSensors(_MainReservedChannel):
    """Канал комплекта датчиков нулевого положения сканера, арретира и зон"""


class ChannelURF(_MainReservedChannel):
    """Канал УРФ"""


class ChannelLaser(_MainReservedChannel):
    """Канал лазера"""


class ChannelLaserIP(_MainReservedChannel):
    """Канал источника питания лазера"""


class ChannelPPT(_MainReservedChannel):
    """Канал ППТ"""


class ChannelTmi:
    """Датчик ТМИ"""
    tmi1 = 0
    tmi2 = 1
    # undefined else


class ChannelPPTReference:
    """Сигнал референтного канала"""
    there_is = 0  # В наличии
    there_is_not = 1  # Отсутствует

## Программная система обнаружения аномалий в телеметрии бортового фурье-спектрометра ИКФС-2

#### Включает в себя:
* Модуль экспертного анализа
* Модуль интеллектуального анализа:
    * LSTM-предиктор
    * LSTM-автокодировщик

#### Сценарий работы:

1 . Сформировать конфигурационный файл
```bash
$ ikfs-anomaly-detector dump-config
$ mv default_config.yml /tmp/myconfig.yml
```

2 . Указать в нём соответствующие одиночные сигналы и группы сигналов,
предполагаемые для анализа
```yml
# Директория для хранения обученных моделей
models_dir: '/tmp/ikfs_anomaly_detector/models'

# Директория с логами tensorboard [опционально]
tensorboard_dir: ''  

# Директория с результатами
analysis_result_dir: /tmp/ikfs_anomaly_detection/results  

# Одиночные сигналы для LSTM-предиктора
predictor_for:
- PptRiple
...

# Группы сигналов для LSTM-автокодировщика
autoencoder_for:
  bfk:
  - BfkCntErrCrc
  - BfkCntErrRxBufAlloc
  - BfkCntErrRxPacket
  ...
  
  mygroup:
  - StrSensorMi1
  - BpopCntErrCrc

thresholds:
  # Границы аномалий по умолчанию
  default:
    rules: 0.55  # Граница аномалий для экспертного анализатора
    
    # Границы аномалий для групп и одиночных сигналов
    bfk: 0.2
    bpop: 0.4
    bud: 6.0
    bud_board: 15.0
    fp: 0.7
    mi: 0.4
    mk: 0.09
    ppt: 0.27
    ppt_direction: 0.1
    str: 0.05
    PptRiple: 100
    PptSampleCount: 100
    ScannerAngle: 610
    Str27V: 210
    StrSensorTu1: 100
    StrSensorTu2: 100
  
  # Границы аномалий для конкретного файла
  METM2_9181_9174_1VIE2-IMR_8_IKFS-2_01P8:
    mi: 0.9
    mk: 0.19
    fp: 1.
    ppt_direction: 0.015
```

3 . Произвести обучение системы
```bash
$ ikfs-anomaly-detector train --telemetry-dir /Downloads/good_data -c /tmp/myconfig.yml
```

4 . Выполнить поверхностный анализ телеметрии 
```bash
$ ikfs-anomaly-detector analyze --telemetry-file /Downloads/bad_data/METM2_9181_9174_1VIE2-IMR_8_IKFS-2_01P8.tlm.h5 -c /tmp/myconfig.yml
```

5 . Посмотреть результаты (фукнции ошибок по каждому из сигналов) и при необходимости подправить границы

6 . Выполнить полный анализ телеметрии
```bash
$ ikfs-anomaly-detector analyze --full-report --telemetry-file /Downloads/bad_data/METM2_9181_9174_1VIE2-IMR_8_IKFS-2_01P8.tlm.h5 -c /tmp/myconfig.yml
```

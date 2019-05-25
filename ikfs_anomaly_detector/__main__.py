import gettext
from ikfs_anomaly_detector.core.translation import translate

gettext.gettext = translate

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from ikfs_anomaly_detector.core.config import dump_default_config, load_config
from ikfs_anomaly_detector.core.reader import TelemetryReader
from ikfs_anomaly_detector.core.utils import write_rolluped_points, roll_up_points
from ikfs_anomaly_detector.expert.analysis import run_expert_analyzer
from ikfs_anomaly_detector.intellectual.analysis import run_predictor, run_autoencoder
from ikfs_anomaly_detector.intellectual.training import train_autoencoder, train_predictor


def dump_config(args: argparse.Namespace) -> None:
    path = dump_default_config()
    print(f'Конфигурационный файл сохранен в "{path}"')
    exit(0)


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not config.models_dir:
        raise SystemExit('Директория для сохранения моделей "models_dir" не определена')

    train_predictor(
        signals=config.signals_for_predictor,
        telemetry_dir=args.telemetry_dir,
        models_dir=config.models_dir,
        tensorboard_dir=config.tensorboard_dir,
    )

    print('\n' + '*' * 100 + '\n')

    train_autoencoder(
        signal_groups=config.signals_groups,
        telemetry_dir=args.telemetry_dir,
        models_dir=config.models_dir,
        tensorboard_dir=config.tensorboard_dir,
    )
    exit(0)


def analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not config.models_dir:
        print('Используются системные (обученные заранее) модели')

    if not config.analysis_result_dir:
        raise SystemExit('Директория для вывода результатов "analysis_result_dir" не определена')

    fname = os.path.basename(args.telemetry_file).split('.')[0]
    thresholds = {}
    for k, v in config.thresholds['default'].items():
        thresholds[k] = config.thresholds.get(fname, {}).get(k, v)

    results_path = os.path.join(config.analysis_result_dir, fname)
    os.makedirs(results_path, exist_ok=True)

    anomalies_points = {}

    with TelemetryReader(args.telemetry_file) as reader:
        print('\n' + '*' * 100 + '\n')

        run_expert_analyzer(
            reader=reader,
            threshold=thresholds['rules'],
            results_path=results_path,
        )
        print('\n' + '*' * 100 + '\n')

        run_predictor(
            reader=reader,
            models_dir=config.models_dir,
            signals=config.signals_for_predictor,
            thresholds=thresholds,
            results_path=results_path,
            anomalies_points_storage=anomalies_points,
            full_report=args.full_report,
        )
        print('\n' + '*' * 100 + '\n')

        run_autoencoder(
            reader=reader,
            models_dir=config.models_dir,
            signal_groups=config.signals_groups,
            thresholds=thresholds,
            results_path=results_path,
            anomalies_points_storage=anomalies_points,
            full_report=True,
        )
        print('\n' + '*' * 100 + '\n')

    print('Печать выходного файла...')
    anomalies_counter = [0] * 100_000

    with open(f'{results_path}/anomalies.txt', 'w') as f:
        f.write('Аномальные участки:\n')

        for signal, points in anomalies_points.items():
            if not points:
                continue

            for i in points:
                anomalies_counter[i] += 1

            f.write(f'- {signal}\n')
            write_rolluped_points(roll_up_points(points), f)

        for min_cnt in range(2, 5 + 1):
            anomaly_points = roll_up_points([i for i, cnt in enumerate(anomalies_counter) if cnt > min_cnt])

            f.write(f'\nАномальные участки, встречающиеся среди анализаторов более {min_cnt} раз:\n')
            if anomaly_points:
                write_rolluped_points(anomaly_points, f)
            else:
                f.write('\tотсутствуют.\n')
    exit(0)


class Formatter(argparse.RawTextHelpFormatter):

    def __init__(self, **kwargs) -> None:
        kwargs['width'] = 50_000
        super().__init__(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Программная система обнаружения аномалий в телеметрии бортового фурье-спектрометра ИКФС-2',
        formatter_class=Formatter,
    )
    subparsers = parser.add_subparsers()

    dump_parser = subparsers.add_parser(
        'dump-config', help='Создать конфигурационный файл с параметрами по умолчанию', formatter_class=Formatter
    )
    dump_parser.set_defaults(func=dump_config)  # config --dump, config --help

    train_parser = subparsers.add_parser(
        'train', help='Выполнить обучение моделей', formatter_class=Formatter
    )
    train_parser.add_argument(
        '--telemetry-dir', '-d', type=str, required=True, help='Директория с .h5 телеметрией, не содержащей аномалий',
    )
    train_parser.add_argument(
        '--config', '-c', type=str, required=True, help='Путь к конфигурационному .yml файлу',
    )
    train_parser.set_defaults(func=train)

    analyze_parser = subparsers.add_parser(
        'analyze', help='Выполнить анализ телеметрии', formatter_class=Formatter
    )
    analyze_parser.add_argument(
        '--telemetry-file', '-f', type=str, required=True, help='Файл с телеметрией для анализа (.h5)'
    )
    analyze_parser.add_argument(
        '--config', '-c', type=str, required=True, help='Путь к конфигурационному .yml файлу',
    )
    analyze_parser.add_argument(
        '--full-report',
        action='store_true', default=False, help='Анализ с полным отчётом (графики сигналов, поиск аномали и пр.)'
    )
    analyze_parser.set_defaults(func=analyze)

    args = parser.parse_args()
    if getattr(args, 'func', None):
        args.func(args)

    parser.print_help()


if __name__ == '__main__':
    main()

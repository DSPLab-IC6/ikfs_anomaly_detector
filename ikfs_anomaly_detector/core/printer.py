import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ikfs_anomaly_detector.core.custom_types import AnomalyPoints
from ikfs_anomaly_detector.core.utils import as_set

matplotlib.rcParams['agg.path.chunksize'] = 10_000


class Colours:
    black = 'k'
    blue = 'b'
    cyan = 'c'
    green = 'g'
    magenta = 'm'
    red = 'r'
    yellow = 'y'


COLOURS = as_set(Colours)


class FontWeights:
    light = 'light'
    normal = 'normal'
    regular = 'regular'
    bold = 'bold'


@dataclass
class Label:
    text: str = ''
    font_size: int = 24
    font_weight: str = FontWeights.bold


@dataclass
class Signal:
    name: str
    data: np.ndarray
    line_width: int = 2
    color: str = ''
    alpha: float = 1.


@dataclass
class Legend:
    font_size: int = 44
    location: str = 'upper right'


@dataclass
class Ticks:
    start: int = 0
    period: int = 2500
    font_size: int = 24


@dataclass
class Subplot:
    signals: List[Signal]
    xlabel: Label = Label()
    ylabel: Label = Label()
    xlim: Tuple = ()
    ylim: Tuple = ()
    legend: Legend = Legend()
    ticks: Ticks = Ticks()


def plot_telemetry(
        *subplots: Subplot,
        width: int = 40,  # inches
        height: int = 10,  # inches
        img_path: str = '',
        anomaly_points: Optional[AnomalyPoints] = None,
        anomaly_selection_width: int = 2000,
) -> None:
    anomaly_points = anomaly_points or []

    plt.style.use('ggplot')

    fig, axes = plt.subplots(nrows=len(subplots), figsize=(width, height * len(subplots)))
    if len(subplots) == 1:
        _plot_subplot(axes, subplots[0])
        _plot_anomalies(axes, anomaly_points, selection_width=anomaly_selection_width)
    else:
        for i, plot in enumerate(subplots):
            _plot_subplot(axes[i], plot)
            _plot_anomalies(axes[i], anomaly_points, selection_width=anomaly_selection_width)

    if img_path:
        plt.savefig(img_path, bbox_inches='tight')
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()


def _plot_subplot(axes: Axes, subplot: Subplot) -> None:
    assert subplot.signals, 'Nothing to plot'

    if subplot.xlim:
        axes.set_xlim(*subplot.xlim)

    if subplot.ylim:
        axes.set_ylim(*subplot.ylim)

    axes.set_xlabel(subplot.xlabel.text,
                    fontweight=subplot.xlabel.font_weight,
                    fontsize=subplot.xlabel.font_size)

    axes.set_ylabel(subplot.ylabel.text,
                    fontweight=subplot.ylabel.font_weight,
                    fontsize=subplot.ylabel.font_size,
                    rotation=0)

    ticks = [subplot.ticks.period * i
             for i in range(len(subplot.signals[0].data) // subplot.ticks.period)]
    axes.set_xticks(ticks)

    if subplot.ticks.start != 0:
        ticks_labels = [str(subplot.ticks.period * i + subplot.ticks.start)
                        for i in range(len(subplot.signals[0].data) // subplot.ticks.period)]
        axes.set_xticklabels(ticks_labels)

    for tick_label in axes.get_xticklabels() + axes.get_yticklabels():
        tick_label.set_fontsize(subplot.ticks.font_size)

    lines = []
    names = []
    for signal in subplot.signals:
        color = signal.color or random.choice(list(COLOURS - {Colours.red}))
        lines.append(
            axes.plot(signal.data,
                      color=color,
                      linewidth=signal.line_width,
                      alpha=signal.alpha,
                      )[0]
        )
        names.append(signal.name)

    axes.legend(lines, names, loc=subplot.legend.location, fontsize=subplot.legend.font_size)


def _plot_anomalies(axes: Axes, anomaly_points: AnomalyPoints, selection_width: int = 2000) -> None:
    ymin, ymax = axes.get_ylim()
    selection_half = selection_width // 2

    for anomaly in (
            np.arange(i - selection_half, i + selection_half - 1)
            for i in anomaly_points
    ):
        y1 = [ymin] * len(anomaly)
        y2 = [ymax] * len(anomaly)
        axes.fill_between(anomaly, y1, y2, facecolor='g', alpha=.05)

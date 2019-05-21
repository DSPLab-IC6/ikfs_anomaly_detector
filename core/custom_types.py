from dataclasses import dataclass
from typing import Dict, Union, List

import numpy
from h5py import Dataset

PointT = Union[Dataset, dict]

SignalName = str
SignalValue = Union[int, float]
SignalValues = numpy.ndarray
Signals = Dict[SignalName, SignalValues]

SignalErrorWeight = float


@dataclass
class SignalDetails:
    value: SignalValue
    weight: SignalErrorWeight


PointStrIndex = str
PointIndex = int
PointDetails = Dict[SignalName, SignalDetails]

AnomalyPoints = List[PointIndex]

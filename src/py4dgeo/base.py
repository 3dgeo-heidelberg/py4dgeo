from py4dgeo.epoch import Epoch
from py4dgeo.util import (
    Py4DGeoError,
    as_double_precision,
    make_contiguous,
)

import typing
import numpy as np


class PairwiseEpochCorepointsBase:
    """Shared handling of paired epochs and corepoints."""

    def __init__(
        self,
        epochs: typing.Optional[typing.Tuple[Epoch, ...]] = None,
        corepoints: typing.Optional[np.ndarray] = None,
    ):
        self.epochs = epochs
        self.corepoints = corepoints

    @property
    def corepoints(self):
        return self._corepoints

    @corepoints.setter
    def corepoints(self, _corepoints):
        if _corepoints is None:
            self._corepoints = None
            return

        if len(_corepoints.shape) != 2 or _corepoints.shape[1] != 3:
            raise Py4DGeoError("Corepoints need to be given as an array of shape nx3")

        self._corepoints = as_double_precision(make_contiguous(_corepoints))

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, _epochs):
        if _epochs is not None and len(_epochs) != 2:
            raise Py4DGeoError("Exactly two epochs need to be given!")
        self._epochs = _epochs

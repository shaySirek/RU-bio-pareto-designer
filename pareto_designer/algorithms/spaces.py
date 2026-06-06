from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Type

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

ScoreType = Union[float, NDArray[np.float64]]
WeightType = Union[int, float, NDArray[np.float64]]


class ScoreSpace(ABC):
    Identity: float

    @staticmethod
    def _ret(out: ScoreType) -> ScoreType:
        return out.item() if isinstance(out, np.ndarray) and out.ndim == 0 else out

    @classmethod
    @abstractmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        pass

    @classmethod
    @abstractmethod
    def _subtract(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        pass

    @classmethod
    @abstractmethod
    def _distance(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        pass

    @classmethod
    def _weighted_distance(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        return (w_x * w_y) * cls._distance(x, y) / (w_x + w_y)

    @classmethod
    @abstractmethod
    def _weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        pass

    @classmethod
    def add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return cls._ret(cls._add(x, y))

    @classmethod
    def subtract(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return cls._ret(cls._subtract(x, y))

    @classmethod
    def distance(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return cls._ret(cls._distance(x, y))

    @classmethod
    def weighted_distance(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        return cls._ret(cls._weighted_distance(x, w_x, y, w_y))

    @classmethod
    def weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        return cls._ret(cls._weighted_mean(x, w_x, y, w_y))


class LinearSpace(ScoreSpace):
    Identity: float = 0.0

    @classmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return x + y

    @classmethod
    def _subtract(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return x - y

    @classmethod
    def _distance(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return (x - y) ** 2

    @classmethod
    def _weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        return (w_x * x + w_y * y) / (w_x + w_y)


class ExpSpace(ScoreSpace):
    Identity: float = -float("inf")

    @classmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return np.logaddexp(x, y)

    @classmethod
    def _subtract(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        xa, ya = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )

        if np.any(xa < ya):
            raise ValueError(f"Log-space subtraction underflow: x={x}, y={y}")

        out = np.empty_like(xa)
        equal = xa == ya
        y_inf = ya == cls.Identity
        valid = ~(equal | y_inf)

        out[equal] = cls.Identity
        out[y_inf] = xa[y_inf]

        if np.any(valid):
            out[valid] = ya[valid] + np.log(np.expm1(xa[valid] - ya[valid]))

        return out

    @classmethod
    def _distance(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        xa, ya = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        return (np.exp(xa) - np.exp(ya)) ** 2

    @classmethod
    def _weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)

        stacked_scores = np.stack([xa, ya], axis=0)
        stacked_weights = np.stack([w_x, w_y], axis=0)

        return logsumexp(stacked_scores, axis=0, b=stacked_weights) - np.log(w_x + w_y)


class ScoreSpaceOption(Enum):
    Linear = "linear"
    LogExp = "logexp"

    def get_space(self) -> Type[ScoreSpace]:
        return {
            ScoreSpaceOption.Linear: LinearSpace,
            ScoreSpaceOption.LogExp: ExpSpace,
        }[self]

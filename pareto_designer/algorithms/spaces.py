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

    @staticmethod
    @abstractmethod
    def proj(x: ScoreType) -> ScoreType:
        pass

    @classmethod
    @abstractmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        pass

    @classmethod
    @abstractmethod
    def _subtract(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        pass

    @classmethod
    def _distance(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return np.square(cls.proj(x) - cls.proj(y))

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

    @staticmethod
    def proj(x: ScoreType) -> ScoreType:
        return x

    @classmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return x + y

    @classmethod
    def _weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        return (w_x * x + w_y * y) / (w_x + w_y)


class ExpSpace(ScoreSpace):
    Identity: float = -float("inf")

    @staticmethod
    def proj(x: ScoreType) -> ScoreType:
        return np.exp(x)

    @classmethod
    def _add(cls, x: ScoreType, y: ScoreType) -> ScoreType:
        return np.logaddexp(x, y)

    @classmethod
    def _weighted_mean(
        cls, x: ScoreType, w_x: WeightType, y: ScoreType, w_y: WeightType
    ) -> ScoreType:
        stacked_scores = np.stack(
            [np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=0
        )
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import math

import numpy as np


@runtime_checkable
class SamplingMethod(Protocol):
    k: int

    @property
    def params(self) -> str: ...

    def __call__(
        self, scores: np.ndarray, indices: np.ndarray, position: int
    ) -> np.ndarray: ...


@dataclass
class PowerLawSUS:
    """
    Stochastic Universal Sampling (SUS) with fitness values calculated by
    applying the inverse Power-Law distribution on the average functional cost per bp.

    Args:
        k: The number of solutions to sample.
        alpha: Exponent for the inverse Power-Law weighting.
        use_dynamic_log_position_exponent: Whether to multiply the exponent by log(position + 1).
    """

    k: int
    alpha: float
    use_dynamic_log_position_exponent: bool

    @property
    def params(self) -> str:
        params = f"k_{self.k}__alpha_{self.alpha}"
        if self.use_dynamic_log_position_exponent:
            params += "_log_pos"
        return params

    def __call__(
        self, scores: np.ndarray, indices: np.ndarray, position: int
    ) -> np.ndarray:
        return self._sample(scores[indices], position)

    def _sample(self, scores: np.ndarray, i: int) -> np.ndarray:
        n = len(scores)
        if self.k == 0 or n <= self.k:
            return scores

        avg_costs = -scores[:, 0] / i
        exp = -self.alpha
        if self.use_dynamic_log_position_exponent:
            exp *= math.log(i + 1)
        weights = np.power(avg_costs + 1, exp)

        cum_weights = np.cumsum(weights)
        total_w = cum_weights[-1]
        step = total_w / self.k
        pointers = np.random.uniform(0, step) + np.arange(self.k) * step
        sampled_indices = np.searchsorted(cum_weights, pointers)
        unique_indices = np.unique(sampled_indices)

        return scores[unique_indices]


NO_SAMPLING: SamplingMethod = PowerLawSUS(0, 1.0, False)

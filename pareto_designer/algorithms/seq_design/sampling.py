from typing import Protocol
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SamplingMethod(Protocol):
    k: int

    @property
    def params(self) -> str:
        return f"k_{self.k}"

    def __call__(
        self, scores: np.ndarray, indices: np.ndarray, position: int
    ) -> np.ndarray: ...


@dataclass
class PowerLawSUS(SamplingMethod):
    """
    Stochastic Universal Sampling (SUS) with fitness values calculated by
    applying the inverse Power-Law distribution on the average functional cost per bp.

    Args:
        k: The number of solutions to sample.
        alpha: Exponent for the inverse Power-Law weighting.
        use_dynamic_log_position_exponent: Whether to multiply the exponent by log(position + 1).
    """

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
        # Unbounded or less than bound (k)
        if self.k == 0 or n <= self.k:
            return scores

        # Weighting function
        avg_costs = -scores[:, 0] / i
        exp = -self.alpha
        if self.use_dynamic_log_position_exponent:
            exp *= math.log(i + 1)
        weights = np.power(avg_costs + 1, exp)

        # Stochastic Universal Sampling (SUS)
        cum_weights = np.cumsum(weights)
        total_w = cum_weights[-1]
        step = total_w / self.k
        pointers = np.random.uniform(0, step) + np.arange(self.k) * step
        sampled_indices = np.searchsorted(cum_weights, pointers)
        unique_indices = np.unique(sampled_indices)

        return scores[unique_indices]


NO_SAMPLING: SamplingMethod = PowerLawSUS(0, 1.0, False)


@dataclass
class RankedPowerLawSampling(SamplingMethod):
    """
    Rank-based sampling on an index array using a composite
    power-law distribution defined by multiple alphas and budget ratios.

    Args:
        k: The number of solutions to sample.
        alphas: Exponents determining the greedy bias for each budget segment.
        ratios: Proportional allocation of the total budget $k$ for each alpha.
    """

    alphas: tuple[float, ...]
    ratios: tuple[float, ...]

    @property
    def params(self) -> str:
        return f"k_{self.k}__alphas_{self.alphas}__ratios_{self.ratios}"

    def __call__(
        self, scores: np.ndarray, indices: np.ndarray, position: int
    ) -> np.ndarray:
        sampled_indices = self._sample(indices)
        return scores[sampled_indices]

    def _sample(self, indices: np.ndarray) -> np.ndarray:
        n = len(indices)
        # Unbounded or less than bound (k)
        if self.k == 0 or n <= self.k:
            return indices

        final_idx_set = set()
        for alpha, ratio in zip(self.alphas, self.ratios):
            k_part = max(1, int(round(self.k * ratio)))
            j = np.linspace(0, 1, k_part)
            rank_indices = np.round(np.power(j, alpha) * (n - 1)).astype(int)
            final_idx_set.update(indices[np.unique(rank_indices)])

        sampled_indices = np.sort(np.array(list(final_idx_set)))[: self.k]
        return sampled_indices

    @classmethod
    def get_sampler(
        cls, i: int, orf: tuple[int, int] | None
    ) -> "RankedPowerLawSampling":
        if not orf:
            return cls(alphas=(1.5, 1.0), ratios=(0.5, 0.5))

        start, end = orf
        # Phase I: Upstream
        if i < start:
            return cls(alphas=(1.8, 1.1), ratios=(0.5, 0.5))

        # Phase II: CDS
        if i <= end:
            pos_in_codon = (i - start) % 3

            # Positions 1 & 2: Diversity Phase
            if pos_in_codon < 2:
                return cls(alphas=(1.05, 1.0), ratios=(0.5, 0.5))

            # Position 3: Selection Phase, handles high costs
            return cls(alphas=(4.5, 1.2), ratios=(0.7, 0.3))

        # Phase III: Downstream
        return cls(alphas=(2.5, 1.5), ratios=(0.8, 0.2))

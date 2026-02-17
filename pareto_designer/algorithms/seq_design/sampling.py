from typing import Protocol
from dataclasses import dataclass

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
class SUS(SamplingMethod):
    """
    Stochastic Universal Sampling (SUS) with fitness values
    proportional to the average functional score per bp.

    Args:
        k: The number of solutions to sample.
    """

    def __call__(
        self, scores: np.ndarray, indices: np.ndarray, position: int
    ) -> np.ndarray:
        return self._sus(scores[indices], position)

    def _sus(self, scores: np.ndarray, position: int):
        n = len(scores)
        # Unbounded or less than bound (k)
        if self.k == 0 or n <= self.k:
            return scores

        # Weights proportional to average functional score per bp
        weights = np.exp(scores[:, 0] / position)

        # Stochastic Universal Sampling (SUS)
        cum_weights = np.cumsum(weights)
        total_w = cum_weights[-1]

        # Create equidistant pointers to ensure proportional representation
        step = total_w / self.k
        pointers = np.random.uniform(0, step) + np.arange(self.k) * step

        # Vectorized binary search for pointer placement
        sampled_indices = np.searchsorted(cum_weights, pointers)
        unique_indices = np.unique(sampled_indices)

        return scores[unique_indices]


NO_SAMPLING: SamplingMethod = SUS(0)


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
        sampled_indices = self._rank_based_sampler(indices)
        return scores[sampled_indices]

    def _rank_based_sampler(self, indices: np.ndarray) -> np.ndarray:
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

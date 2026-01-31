import re

import numpy as np

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.amino_acid_utils import AminoAcidConfig


def evaluate_non_coding_substitution(
    target_sequence: str,
    i: int,
    sigma: str,
    alpha: float,
    beta: float,
) -> float:
    # No substitution
    if target_sequence[i] == sigma:
        return 0.0

    # Transition substitution
    if AminoAcidConfig.is_transition(target_sequence[i], sigma):
        return alpha

    # Transversion substitution
    return beta


def evaluate_coding_substitution(
    proposed_codon: str,
    target_codon: str,
    is_start_codon: int,
    codon_usage_costs: dict[str, float],
    w: float,
) -> float:
    # No substitution
    if proposed_codon == target_codon:
        return 0.0

    # Synonymous substitution with a logarithmic penalty based on codon usage
    if AminoAcidConfig.encodes_same_amino_acid(proposed_codon, target_codon):
        return codon_usage_costs[proposed_codon]

    # Penalize stop codon formation
    if is_start_codon or AminoAcidConfig.either_is_stop_codon(
        target_codon, proposed_codon
    ):
        return float("inf")

    # Non-synonymous substitution
    return w + AminoAcidConfig.edit_dist(target_codon, proposed_codon)


def calculate_cost(
    target_sequence: str,
    coding_positions: list[int],
    codon_usage_costs: dict[str, float],
    i: int,
    v: str,
    sigma: str,
    alpha: float,
    beta: float,
    w: float,
) -> float:
    codon_pos = coding_positions[i]

    # Non-coding region logic
    if codon_pos == 0:
        return evaluate_non_coding_substitution(target_sequence, i, sigma, alpha, beta)

    # Coding region positions 1 and 2
    if codon_pos in {1, 2}:
        return 0.0

    # Coding region, position 3
    if codon_pos in {-3, 3}:
        proposed_codon = f"{v[-2:]}{sigma}"
        target_codon = target_sequence[i - 2 : i + 1]
        is_start_codon = AminoAcidConfig.is_start_codon(codon_pos)
        return evaluate_coding_substitution(
            proposed_codon, target_codon, is_start_codon, codon_usage_costs, w
        )

    # Fallback (should not be reached under correct conditions)
    raise ValueError(f"Unexpected codon position value: {codon_pos}")


class BioCostFunction(ScoreFunction):
    def __init__(
        self,
        target_sequence: str,
        coding_positions: list[int],
        codon_usage: dict[str, float],
        alpha: float,
        beta: float,
        w: float,
    ):
        self._target_sequence = target_sequence
        self._coding_positions = coding_positions
        self._codon_usage_costs = self._get_codon_usage_costs(codon_usage)
        self._alpha = alpha
        self._beta = beta
        self._w = w

    @staticmethod
    def _get_codon_usage_costs(codon_usage: dict[str, float]) -> dict[str, float]:
        usage_values = np.array(list(codon_usage.values()))
        min_usage = usage_values.min()
        costs = np.log(usage_values) / np.log(min_usage)
        return dict(zip(codon_usage.keys(), costs))

    @property
    def target_sequence(self):
        return self._target_sequence

    def cost(self, i: int, v: str, sigma: str) -> float:
        return calculate_cost(
            self._target_sequence,
            self._coding_positions,
            self._codon_usage_costs,
            i,
            v,
            sigma,
            self._alpha,
            self._beta,
            self._w,
        )

    @property
    def orfs(self) -> list[tuple[int, int]]:
        return [
            (m.start() + 1, m.end() - 1)
            for m in re.finditer(
                r"12-3[123-]*3(?=0|$)", "".join(map(str, self._coding_positions))
            )
        ]

    @property
    def params(self) -> dict:
        return {
            "Transition": self._alpha,
            "Transversion": self._beta,
            "Non-synonymous codon": self._w,
            "Synonymous codon": "by codon usage",
        }

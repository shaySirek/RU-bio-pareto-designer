import re

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.cost_utils import CostUtils


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

    # non-coding region
    if codon_pos == 0:
        if target_sequence[i] == sigma:
            return 0.0
        if CostUtils.is_transition(target_sequence[i], sigma):
            return alpha
        return beta

    # coding region
    if codon_pos in {1, 2}:
        return 0.0

    proposed_codon = f"{v[-2:]}{sigma}"
    target_codon = target_sequence[i - 2 : i + 1]
    if CostUtils.encodes_same_amino_acid(proposed_codon, target_codon):
        return codon_usage_costs[proposed_codon]

    if CostUtils.is_orf_start(codon_pos) or CostUtils.either_is_stop_codon(
        target_codon, proposed_codon
    ):
        return float("inf")

    return w + CostUtils.hamming_dist(target_codon, proposed_codon)


class BioCostFunction(ScoreFunction):
    def __init__(
        self,
        target_sequence: str,
        coding_positions: list[int],
        codon_usage_costs: dict[str, float],
        alpha: float,
        beta: float,
        w: float,
    ):
        self._target_sequence = target_sequence
        self._coding_positions = coding_positions
        self._codon_usage_costs = codon_usage_costs
        self._alpha = alpha
        self._beta = beta
        self._w = w

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

    @property
    def maximum(self) -> float:
        costs = (self._alpha, self._beta, self._w + 3)
        return max(filter(lambda c: c != float("inf"), costs))

import re

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.cost_utils import CostUtils


class BioCostFunction(ScoreFunction):
    def __init__(
        self,
        target_sequence: str,
        cost_utils: CostUtils,
        codon_usage: dict[str, float],
        alpha: float,
        beta: float,
        w: float,
    ):
        self._codon_usage_costs = cost_utils.calculate_codon_costs(codon_usage)
        self._target_sequence, self._coding_positions = cost_utils.get_coding_positions(
            target_sequence
        )
        self._cost_utils = cost_utils
        self._alpha = alpha
        self._beta = beta
        self._w = w

    @property
    def target_sequence(self) -> str:
        return self._target_sequence

    @property
    def codon_usage_costs(self) -> dict[str, float]:
        return self._codon_usage_costs

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def w(self) -> float:
        return self._w

    def cost(self, i: int, v: str, sigma: str) -> float:
        codon_pos = self._coding_positions[i]

        # non-coding region
        if codon_pos == 0:
            if self._target_sequence[i] == sigma:
                return 0.0
            if self._cost_utils.is_transition(self._target_sequence[i], sigma):
                return self._alpha
            return self._beta

        # coding region
        if codon_pos in {1, 2}:
            return 0.0

        proposed_codon = f"{v[-2:]}{sigma}"
        target_codon = self._target_sequence[i - 2 : i + 1]
        if self._cost_utils.encodes_same_amino_acid(proposed_codon, target_codon):
            return self._codon_usage_costs[proposed_codon]

        if self._cost_utils.is_orf_start(
            codon_pos
        ) or self._cost_utils.either_is_stop_codon(target_codon, proposed_codon):
            return float("inf")

        return self._w + self._cost_utils.hamming_dist(target_codon, proposed_codon)

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

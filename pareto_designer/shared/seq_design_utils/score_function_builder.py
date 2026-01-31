from pathlib import Path

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.exact_match_function import ExactMatchCostFunction
from pareto_designer.shared.func_cost.bio_function import BioCostFunction
from pareto_designer.shared.parsing import read_sequence, read_codon_usage
from pareto_designer.shared.cds_util import get_coding_positions


class ScoreFunctionBuilder:
    def __init__(self):
        self._target_sequence: str = None
        self._is_exact_match_cost: bool = False
        self._codon_usage: dict[str, float] = None
        self._alpha: float = None
        self._beta: float = None
        self._w: float = None

    def with_target_sequence(self, seq_file: Path) -> "ScoreFunctionBuilder":
        self._target_sequence = read_sequence(seq_file)
        return self

    def with_is_exact_match_cost(
        self, is_exact_match_cost: bool
    ) -> "ScoreFunctionBuilder":
        self._is_exact_match_cost = is_exact_match_cost
        return self

    def with_codon_usage(self, codon_usage_file: Path) -> "ScoreFunctionBuilder":
        if not self._is_exact_match_cost:
            self._codon_usage = read_codon_usage(codon_usage_file)
        return self

    def with_params(self, **kwargs) -> "ScoreFunctionBuilder":
        if not self._is_exact_match_cost:
            self._alpha = kwargs.pop("alpha")
            self._beta = kwargs.pop("beta")
            self._w = kwargs.pop("w")
        return self

    def build(self) -> ScoreFunction:
        if self._is_exact_match_cost:
            return ExactMatchCostFunction(self._target_sequence)

        target_sequence, coding_positions = get_coding_positions(self._target_sequence)
        return BioCostFunction(
            target_sequence,
            coding_positions,
            self._codon_usage,
            self._alpha,
            self._beta,
            self._w,
        )

from loguru import logger
from pathlib import Path

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.exact_match_function import ExactMatchCostFunction
from pareto_designer.shared.func_cost.bio_function import BioCostFunction
from pareto_designer.shared.parsing import read_sequence, read_codon_usage
from pareto_designer.shared.cds_util import get_coding_positions, START_CODON_MARKER


class ScoreFunctionBuilder:
    def __init__(self):
        self._target_sequence = None
        self._is_exact_match_cost = False
        self._coding_positions = None
        self._codon_usage = None
        self._alpha = None
        self._beta = None
        self._w = None

    def with_target_sequence(self, seq_file: Path) -> "ScoreFunctionBuilder":
        self._target_sequence = read_sequence(seq_file)
        length = len(self._target_sequence.replace(START_CODON_MARKER, ""))
        logger.info(f"Read sequence of length {length} ({str(seq_file)})")
        return self

    def with_slice(self, start: int, end: int) -> "ScoreFunctionBuilder":
        if self._target_sequence:
            self._target_sequence = self._target_sequence[start:end]
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
        if not self._is_exact_match_cost:
            self._target_sequence, self._coding_positions = get_coding_positions(
                self._target_sequence
            )

        score_function_cls = (
            ExactMatchCostFunction if self._is_exact_match_cost else BioCostFunction
        )
        missing = [
            f for f in score_function_cls.required_fields if getattr(self, f) is None
        ]
        if missing:
            fields_str = ", ".join(missing)
            raise ValueError(f"Cannot create score function, missing: {fields_str}.")

        return score_function_cls(
            *[getattr(self, f) for f in score_function_cls.required_fields]
        )

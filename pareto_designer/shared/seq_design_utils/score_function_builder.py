from pathlib import Path
import csv

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.func_cost.cost_utils import CostUtils
from pareto_designer.shared.func_cost.bio_function import BioCostFunction
from pareto_designer.shared.parsing import read_sequence, read_codon_usage


class ScoreFunctionBuilder:
    def __init__(self):
        self._target_sequence: str = None
        self._codon_usage_file: Path = None
        self._alpha: float = None
        self._beta: float = None
        self._w: float = None

    def with_target_sequence(self, seq_file: Path) -> "ScoreFunctionBuilder":
        self._target_sequence = read_sequence(seq_file)
        return self

    def with_codon_usage(self, codon_usage_file: Path) -> "ScoreFunctionBuilder":
        self._codon_usage_file = codon_usage_file
        return self

    def with_params(self, **kwargs) -> "ScoreFunctionBuilder":
        self._alpha = kwargs.pop("alpha")
        self._beta = kwargs.pop("beta")
        self._w = kwargs.pop("w")
        return self

    def build(self) -> ScoreFunction:
        cost_utils = CostUtils()
        codon_usage = read_codon_usage(self._codon_usage_file)
        func = BioCostFunction(
            self._target_sequence,
            cost_utils,
            codon_usage,
            self._alpha,
            self._beta,
            self._w,
        )
        with self._codon_usage_file.with_suffix(".costs.csv").open(
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Codon", "Cost"])
            writer.writerows(
                [
                    (codon, round(cost, 3))
                    for codon, cost in func.codon_usage_costs.items()
                ]
            )

        return func

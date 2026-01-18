from pareto_designer.shared.func_cost.base_function import ScoreFunction


class ExactMatchCostFunction(ScoreFunction):
    required_fields = [
        "_target_sequence",
    ]

    def __init__(self, target_sequence: str):
        self._target_sequence = target_sequence

    @property
    def target_sequence(self):
        return self._target_sequence

    def cost(self, i: int, v: str, sigma: str) -> float:  # noqa: ARG001
        return 0.0 if self.target_sequence[i] == sigma else 1.0

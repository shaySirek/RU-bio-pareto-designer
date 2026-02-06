from abc import ABC, abstractmethod

from pareto_designer.shared.consts import STATE_MINIMUM_COMMON_SUFFIX_LENGTH


class ScoreFunction(ABC):
    @property
    @abstractmethod
    def target_sequence(self) -> str:
        pass

    @abstractmethod
    def cost(self, i: int, v: str, sigma: str) -> float:
        pass

    def get_costs(self, sequence: str) -> list[float]:
        costs = []
        for i, sigma in enumerate(sequence):
            v = (
                sequence[i - STATE_MINIMUM_COMMON_SUFFIX_LENGTH : i]
                if i >= STATE_MINIMUM_COMMON_SUFFIX_LENGTH
                else STATE_MINIMUM_COMMON_SUFFIX_LENGTH * "#"
            )
            costs.append(self.cost(i, v, sigma))

        return costs

    def __call__(self, *args, **kwds):
        return -self.cost(*args)

    @property
    def orfs(self) -> list[tuple[int, int]]:
        return []

    @property
    def params(self) -> dict:
        return {}

    @property
    def maximum(self) -> float:
        return 1.0

from typing import Generic
from itertools import combinations
from dataclasses import dataclass

from pareto_designer.algorithms.fsm import T_STATE
from pareto_designer.algorithms.fsm_reduction.util import hash_pair


class Multisets(Generic[T_STATE]):
    def __init__(self, binding_score_map: dict[T_STATE, float]):
        self._multiset_descriptors: dict[T_STATE, SetDescriptor] = {
            v: SetDescriptor(size=1, mean=score)
            for v, score in binding_score_map.items()
        }

    def get_distance(self, v1: T_STATE, v2: T_STATE) -> float:
        v1_desc = self._multiset_descriptors[v1]
        v2_desc = self._multiset_descriptors[v2]
        mean_diff = v1_desc.mean - v2_desc.mean
        numerator = v1_desc.size * v2_desc.size * mean_diff * mean_diff
        # use SSE rather than MSE to better precision
        denominator = v1_desc.size + v2_desc.size
        return numerator / denominator

    def merge(self, v1: T_STATE, v2: T_STATE):
        v1_desc = self._multiset_descriptors.pop(v1)
        v2_desc = self._multiset_descriptors.pop(v2)
        merged_size = v1_desc.size + v2_desc.size
        merged_mean = (v1_desc.sum + v2_desc.sum) / merged_size
        self._multiset_descriptors[v1] = SetDescriptor(
            size=merged_size, mean=merged_mean
        )
        return self

    def get_descriptor(self, v: T_STATE) -> "SetDescriptor":
        return self._multiset_descriptors[v]

    def get_binding_score_map(self) -> dict[T_STATE, float]:
        return {v: v_desc.mean for v, v_desc in self._multiset_descriptors.items()}

    def get_mergeable_pairs_distances(
        self, mergeable_sets: list[list[T_STATE]]
    ) -> list[tuple[float, tuple[T_STATE, T_STATE]]]:
        mergeable_pairs_distances = [
            (self.get_distance(*pair), hash_pair(*pair))
            for mergeable_set in mergeable_sets
            for pair in combinations(mergeable_set, 2)
        ]
        return mergeable_pairs_distances


@dataclass
class SetDescriptor:
    size: int
    mean: float

    @property
    def sum(self) -> float:
        return self.size * self.mean

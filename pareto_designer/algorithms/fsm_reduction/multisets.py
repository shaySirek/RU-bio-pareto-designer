from typing import Generic
from itertools import combinations

import numpy as np

from pareto_designer.algorithms.fsm import T_STATE
from pareto_designer.algorithms.fsm_reduction.util import hash_pair
from pareto_designer.algorithms.fsm_reduction.hierarchical_clustering import (
    ClusterState,
    HierarchicalClustering,
)
from pareto_designer.algorithms.spaces import ScoreType


class Multisets(Generic[T_STATE]):
    def __init__(
        self,
        binding_score_map: dict[T_STATE, float],
        hc: HierarchicalClustering,
    ):
        self._multiset_descriptors: dict[T_STATE, ClusterState] = {
            v: ClusterState(size=1, mean=score)
            for v, score in binding_score_map.items()
        }
        self._hc = hc

    def get_distance(self, v1: T_STATE, v2: T_STATE) -> float:
        c1 = self._multiset_descriptors[v1]
        c2 = self._multiset_descriptors[v2]
        return self._hc.distance(c1, c2)

    def merge(self, v1: T_STATE, v2: T_STATE):
        c1 = self._multiset_descriptors.pop(v1)
        c2 = self._multiset_descriptors.pop(v2)
        self._multiset_descriptors[v1] = self._hc.merge(c1, c2)
        return self

    def get_descriptor(self, v: T_STATE) -> ClusterState:
        return self._multiset_descriptors[v]

    def get_binding_score_map(self) -> dict[T_STATE, float]:
        return {v: v_desc.mean for v, v_desc in self._multiset_descriptors.items()}

    def get_mergeable_pairs_distances(
        self, mergeable_sets: list[list[T_STATE]]
    ) -> list[tuple[ScoreType, tuple[T_STATE, T_STATE]]]:
        pairs = []
        x_list, wx_list = [], []
        y_list, wy_list = [], []

        for mergeable_set in mergeable_sets:
            for v1, v2 in combinations(mergeable_set, 2):
                c1 = self._multiset_descriptors[v1]
                c2 = self._multiset_descriptors[v2]
                pairs.append(hash_pair(v1, v2))
                x_list.append(c1.mean)
                wx_list.append(c1.size)
                y_list.append(c2.mean)
                wy_list.append(c2.size)

        if not pairs:
            return []

        x = np.array(x_list, dtype=float)
        w_x = np.array(wx_list, dtype=float)
        y = np.array(y_list, dtype=float)
        w_y = np.array(wy_list, dtype=float)
        distances = self._hc.space.weighted_distance(x, w_x, y, w_y).tolist()

        return list(zip(distances, pairs))

from typing import NamedTuple, Type

from pareto_designer.algorithms.spaces import ScoreSpace


class ClusterState(NamedTuple):
    size: int
    mean: float


class HierarchicalClustering:

    def __init__(self, space: Type[ScoreSpace]) -> None:
        self.space = space

    def distance(self, c1: ClusterState, c2: ClusterState) -> float:
        return float(self.space.weighted_distance(c1.mean, c1.size, c2.mean, c2.size))

    def merge(self, c1: ClusterState, c2: ClusterState) -> ClusterState:
        n1, n2 = c1.size, c2.size
        n_new = n1 + n2
        mean_new = self.space.weighted_mean(c1.mean, n1, c2.mean, n2)
        return ClusterState(size=n_new, mean=mean_new)

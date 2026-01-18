from typing import Generic, Iterator
from itertools import product, combinations
from operator import itemgetter
import sys

from loguru import logger
import numpy as np
from sklearn.metrics import mean_squared_error

from pareto_designer.algorithms.fsm import FSM, T_STATE, T_CHAR
from pareto_designer.algorithms.fsm_reduction.union_find import UnionFind
from pareto_designer.algorithms.fsm_reduction.multisets import Multisets
from pareto_designer.algorithms.fsm_reduction.min_heap import MinHeap
from pareto_designer.algorithms.fsm_reduction.fsm_builder import FSM_Merger
from pareto_designer.algorithms.fsm_reduction.util import get_scores, hash_pair


class DB_FSM_Reducer(Generic[T_STATE, T_CHAR]):
    """This class implements the algorithm for finding a series of reduced versions of a DB FSM."""

    def __init__(
        self,
        fsm: FSM[T_STATE, T_CHAR],
        binding_score_map: dict[T_STATE, float],
        run_id: str,
        validate: bool = False,
        validate_every: int = 100,
    ):
        self.origin_fsm = fsm
        self.binding_score_map = binding_score_map
        self.run_id = run_id
        self._validate = validate
        self._validate_every = validate_every

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(
            f"logs/colorless_{self.run_id}.log",
            format="{time} {level} {message}",
            level="DEBUG",
        )

    def find_reduced_fsms(
        self,
    ) -> Iterator[
        tuple[
            FSM[T_STATE, T_CHAR],
            float,
            tuple[
                dict[T_STATE, float],
                dict[T_STATE, T_STATE],
                dict[T_STATE, list[T_STATE]],
            ],
        ]
    ]:
        self._ds: UnionFind[T_STATE] = UnionFind[T_STATE]()
        self._multisets: Multisets[T_STATE] = Multisets(self.binding_score_map)
        self._merge_list: MinHeap[tuple[T_STATE, T_STATE], float] = MinHeap()
        self._sse: float = 0.0
        reduced_fsm_idx: int = 0
        self._n_origin = len(self.origin_fsm.V)

        self._init_ds_and_sorted_list()
        fsm_builder: FSM_Merger = FSM_Merger[T_STATE, T_CHAR](
            self._ds, self.origin_fsm, self._add_mergeable_set
        )
        self._reduced_fsm, n_mergeable_sets, self._f, self._f_inverse = (
            fsm_builder.get_current()
        )

        logger.info("Starting process of state reduction ...")
        while not self._merge_list.is_empty():
            logger.debug(f"Starting iteration {reduced_fsm_idx}")
            run_validation = (
                self._validate and reduced_fsm_idx % self._validate_every == 0
            )
            if run_validation:
                self._validate_ds_eq_fsm()
                self._validate_mergeable_pairs_in_mergeList()

            (v1, v2), d = self._merge_list.extract_min()
            _, mergeable_set = self._ds.get(v1)
            mergeable_set.difference_update({v1, v2})
            if run_validation:
                self._validate_extracted_pair_with_minimal_distance(v1, v2, d)

            consider_for_merge = fsm_builder.merge_and_get_consider_for_merge([v1, v2])
            self._reduced_fsm_binding_score_map = self._multisets.merge(
                v1, v2
            ).get_binding_score_map()
            for u in mergeable_set:
                self._update_pair(v1, u)
                self._delete_pair(v2, u)
            if run_validation:
                self._validate_unchanged_pairwise_distances_in_set(
                    v1, v2, mergeable_set
                )

            self._reduced_fsm, n_mergeable_sets, self._f, self._f_inverse = (
                consider_for_merge().get_current()
            )
            self._sse += d
            if run_validation:
                self._validate_fsm_eq_origin()
                self._validate_sse()
                logger.info(f"validation passed at iteration {reduced_fsm_idx}")

            logger.debug(
                f"iteration {reduced_fsm_idx}:"
                f" d{(v1, v2)}={d:.6f},"
                f" SSE={self._sse:.6f},"
                f" |V'|={len(self._reduced_fsm.V)}"
                f" |mergeList|={len(self._merge_list)}"
                f"\n\t{n_mergeable_sets} mergeable sets were detected during the process"
            )

            reduced_fsm_idx += 1
            yield self._reduced_fsm, self.mse, (
                self._reduced_fsm_binding_score_map,
                self._f,
                self._f_inverse,
            )

        logger.info(
            f"Finished process of state reduction after {reduced_fsm_idx} iterations"
        )
        logger.info(f"{n_mergeable_sets} mergeable sets were detected during the loop")

    @property
    def mse(self) -> float:
        return self._sse / self._n_origin

    def _init_ds_and_sorted_list(self):
        logger.info("Initializing DS and mergeList from the DB FSM ...")
        m = len(list(self.origin_fsm.V)[0])
        for v in self.origin_fsm.V:
            self._ds.add(v)
        for beta in product(self.origin_fsm.Sigma, repeat=m - 1):
            mergeable_set = set(
                ["".join([sigma, *beta]) for sigma in self.origin_fsm.Sigma]
            )
            self._add_mergeable_set(mergeable_set)
        logger.debug(f"|mergeList|={len(self._merge_list)}")
        logger.info("Finished initialization")

    def _add_mergeable_set(self, mergeable_set: set[T_STATE]):
        logger.debug(f"add_mergeable_set({mergeable_set})")
        v = mergeable_set.pop()
        for u in mergeable_set:
            self._ds.union(v, u)

        _, union_mergeable_set = self._ds.get(v)
        logger.debug(f"add_mergeable_set: union set = {union_mergeable_set}")
        for u1, u2 in combinations(union_mergeable_set, 2):
            self._update_pair(u1, u2)

    def _update_pair(self, u1: T_STATE, u2: T_STATE):
        dist = self._multisets.get_distance(u1, u2)
        pair = hash_pair(u1, u2)
        logger.debug(f"update_pair({pair}, {dist})")
        self._merge_list.update(pair, dist)

    def _delete_pair(self, u1: T_STATE, u2: T_STATE):
        pair = hash_pair(u1, u2)
        logger.debug(f"delete_pair({pair})")
        self._merge_list.delete(pair)

    def _validate_ds_eq_fsm(self):
        """items in `DS` == states in FSM"""
        states_in_ds = self._ds.get_items()
        states_in_fsm = self._reduced_fsm.V
        assert sorted(states_in_ds) == sorted(states_in_fsm), (
            "DS is not compatiable with FSM"
            f"\n\t{len(states_in_ds)} states in DS"
            f"\n\t{len(states_in_fsm)} states in FSM"
        )

    def _validate_mergeable_pairs_in_mergeList(self):
        """mergeable pairs from `DS` == mergeable pairs in `mergeList`"""
        mergeable_sets = self._ds.get_partitioning()
        self._mergeable_pairs_distances = self._multisets.get_mergeable_pairs_distances(
            mergeable_sets
        )
        assert self._merge_list == self._mergeable_pairs_distances, (
            f"DS is not compatiable with mergeList"
            f"\n\t{len(self._mergeable_pairs_distances)} mergeable pairs from DS ({len(mergeable_sets)} mergeable sets)"
            f"\n\t{len(self._merge_list)} mergeable pairs in mergeList"
        )

    def _validate_extracted_pair_with_minimal_distance(
        self, v1: T_STATE, v2: T_STATE, d: float
    ):
        """the extracted pair `(v1,v2)` is a mergeable pair with minimal distance `d`"""
        min_dist = min(map(itemgetter(0), self._mergeable_pairs_distances))
        pairs_with_min_dist = list(
            map(
                itemgetter(1),
                filter(lambda it: it[0] == min_dist, self._mergeable_pairs_distances),
            )
        )
        assert (v1, v2) in pairs_with_min_dist and d == min_dist, (
            f"{(v1, v2)} should be in the list of mergeable pairs with minimal pairwise distance {pairs_with_min_dist}"
            f"\nd{(v1, v2)}: distance from mergeList {d:.6f} should be equal to calculated distance {min_dist:.6f}"
            f"\n\t{v1}: {self._multisets.get_descriptor(v1)}"
            f"\n\t{v2}: {self._multisets.get_descriptor(v2)}"
        )
        del self._mergeable_pairs_distances

    def _validate_unchanged_pairwise_distances_in_set(
        self, v1: T_STATE, v2: T_STATE, mergeable_set: set[T_STATE]
    ):
        """pairwise distances of pairs in `mergeable_set` kept unchanged due to the merge of `v1` and `v2`

        Note that the algorithm updated `(v1, u)` and deleted `(v2, u)`, for each `u` in `mergeable_set`.
        """
        for u1, u2 in combinations(mergeable_set, 2):
            pair = hash_pair(u1, u2)
            pair_dist = self._multisets.get_distance(*pair)
            dist_in_merge_list = self._merge_list.get_key(pair)
            if pair_dist != dist_in_merge_list:
                logger.error(
                    f"pairwise distance in the mergeable set of {v1} has been changed in the merge of {(v1, v2)}:"
                    f"\n\td{pair}: distance from mergeList {dist_in_merge_list:.6f}"
                    f" should be equal to calculated distance {pair_dist:.6f}"
                    f"\n\t{u1}      ---> {self._multisets.get_descriptor(u1)}"
                    f"\n\t{u2}      ---> {self._multisets.get_descriptor(u2)}"
                    f"\n\t{v1}+{v2} ---> {self._multisets.get_descriptor(v1)}"
                )

    def _validate_fsm_eq_origin(self):
        """reduced FSM is equivalent to the origin FSM (transitions)"""
        for v in self.origin_fsm.V:
            v_in_reduced = self._f[v]
            for sigma in self.origin_fsm.Sigma:
                out_in_origin = self.origin_fsm.t(v, sigma)
                out_in_reduced = self._reduced_fsm.t(v_in_reduced, sigma)
                assert self._f[out_in_origin] == out_in_reduced

    def _validate_sse(self):
        """SSE == sum of distances (`d_SSE`)"""
        origin_scores, current_scores = get_scores(
            self.binding_score_map, self._reduced_fsm_binding_score_map, self._f
        )
        calculated_sse = (
            mean_squared_error(origin_scores, current_scores) * self._n_origin
        )
        assert np.isclose(
            calculated_sse, self._sse
        ), f"SSE: {calculated_sse:.6f} != {self._sse:.6f}"

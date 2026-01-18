from typing import Generic
from queue import Queue
from itertools import product
from operator import itemgetter
from collections import defaultdict
import statistics
import sys

from loguru import logger
import numpy as np
from sklearn.metrics import mean_squared_error

from pareto_designer.algorithms.fsm import ColoredFSM, T_STATE, T_CHAR, T_COLOR
from pareto_designer.algorithms.fsm_reduction.union_find import UnionFind
from pareto_designer.algorithms.fsm_reduction.fsm_builder import ColoredFSM_Merger
from pareto_designer.algorithms.fsm_reduction.util import get_scores


class Colored_DB_FSM_Reducer(Generic[T_STATE, T_CHAR, T_COLOR]):
    """This class implements the algorithm for finding the irreducible version of a colored DB FSM."""

    def __init__(self, fsm: ColoredFSM[T_STATE, T_CHAR, T_COLOR], run_id: str):
        self.origin_fsm = fsm
        self.run_id = run_id

        # f: V -> V'    f^{-1}: V' -> V*
        self._f: dict[T_STATE, T_STATE] = {}
        self._f_inverse: dict[T_STATE, list[T_STATE]] = {}
        self._finished = False

        # B: V -> R
        self.binding_score_map: dict[T_STATE, float] = {}
        self._approx_binding_score_map = None

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(
            f"logs/colored_{self.run_id}.log",
            format="{time} {level} {message}",
            level="DEBUG",
        )

    def find_irreducible_fsm(self):
        logger.info("Staring find_irreducible_fsm ...")
        self._q = Queue()
        self._ds: UnionFind = UnionFind[T_STATE]()

        self._init_ds_and_q()
        fsm_builder: ColoredFSM_Merger = ColoredFSM_Merger[T_STATE, T_CHAR, T_COLOR](
            self._ds, self.origin_fsm, self._add_mergeable_set
        )

        logger.info("Starting main loop of reduction ...")
        while not self._q.empty():
            x = self._q.get(block=False)
            v, mergeable_set = self._ds.get(x)
            logger.debug(f"Dequeueing {x}, v={v}")
            if len(mergeable_set) > 1:
                fsm_builder.merge_and_consider_for_merge(
                    [v, *list(mergeable_set - {v})]
                )
            else:
                logger.debug(f"Skipping {v}")
        self._reduced_fsm, n_mergeable_sets_after_init, _, _ = fsm_builder.get_current()
        logger.info("Finished main loop of reduction")
        logger.info(
            f"{n_mergeable_sets_after_init} mergeable sets were detected during the loop"
        )
        logger.info(f"The irreducible FSM has {len(self._reduced_fsm.V)} states")
        logger.debug(f"{{{', '.join(self._reduced_fsm.V)}}}")

        self._build_mapping()
        self._finished = True

        return self._reduced_fsm

    def _init_ds_and_q(self):
        logger.info("Initializing DS and Q from the DB FSM ...")
        candidates: dict[T_COLOR, set[T_STATE]] = {}
        m = len(list(self.origin_fsm.V)[0])

        for v in self.origin_fsm.V:
            self._ds.add(v)
        for beta in product(self.origin_fsm.Sigma, repeat=m - 1):
            for col in self.origin_fsm.C:
                candidates[col] = set()
            for sigma in self.origin_fsm.Sigma:
                gamma = "".join([sigma, *beta])
                candidates[self.origin_fsm.c(gamma)].add(gamma)
            for col in self.origin_fsm.C:
                candidate_set = candidates[col]
                if len(candidate_set) > 1:
                    self._add_mergeable_set(candidate_set)
                del candidates[col]
        logger.info(f"The queue contains {self._q.qsize()} mergeable sets")
        logger.info("Finished initialization")

    def _add_mergeable_set(self, mergeable_set: set[T_STATE]):
        logger.debug(f"Adding set for merge {{{', '.join(mergeable_set)}}}")
        v = mergeable_set.pop()
        for u in mergeable_set:
            self._ds.union(v, u)
        logger.debug(f"Enqueueing representative {v}")
        self._q.put(v)

    def _build_mapping(self):
        logger.debug("Building mapping function f: V -> V' and its inverse function...")
        n_states_irreducible_fsm = 0
        singletons_iter = self._ds.get_singleton_sets_iterator()
        for v, v_set in singletons_iter:
            assert v in self._reduced_fsm.V
            n_states_irreducible_fsm += 1
            self._f.update({w: v for w in v_set})
            self._f_inverse[v] = sorted(list(v_set))
        logger.debug("Finished building mapping function (f)")

        logger.debug(
            "Validating representation of all states in the origin FSM by states in the reduced FSM ..."
        )
        assert len(self._reduced_fsm.V) == n_states_irreducible_fsm
        assert len(self.origin_fsm.V) == len(self._f)
        logger.debug("Representation:\tOK")

    def _ensure_finished(self):
        if not self._finished:
            raise RuntimeError("Run find_irreducible_fsm first")

    @property
    def states_mapping(self) -> dict[T_STATE, T_STATE]:
        """f: V -> V'"""
        self._ensure_finished()
        return self._f

    @property
    def inverse_states_mapping(self) -> dict[T_STATE, list[T_STATE]]:
        """f^{-1}: V' -> V*"""
        self._ensure_finished()
        return self._f_inverse

    def validate(self):
        self._ensure_finished()
        logger.debug("Validation of the reduced FSM")
        logger.debug(90 * "=")

        # validate equivalence by comparing pairs of state paths of random sequences
        # the comparison is done by the mapping function (self._f)
        logger.debug("Validating equivalence of the origin FSM and the reduced FSM ...")
        for v in self.origin_fsm.V:
            v_in_reduced = self._f[v]
            assert self.origin_fsm.c(v) == self._reduced_fsm.c(v_in_reduced)
            for sigma in self.origin_fsm.Sigma:
                out_in_origin = self.origin_fsm.t(v, sigma)
                out_in_reduced = self._reduced_fsm.t(v_in_reduced, sigma)
                assert self._f[out_in_origin] == out_in_reduced
        logger.debug("Equivalence\t:\tOK")

        logger.debug("Validating irreducibility of the reduced FSM ...")
        eq_classes_per_color: dict[T_COLOR, set[tuple[T_STATE, ...]]] = {
            col: set() for col in self._reduced_fsm.C
        }
        for v in self._reduced_fsm.V:
            col = self._reduced_fsm.c(v)
            out = self._reduced_fsm.get_outgoing_transitions(v)
            assert out not in eq_classes_per_color[col]
            eq_classes_per_color[col].add(out)
        logger.debug("Irreducibility:\tOK")
        logger.debug(90 * "=")

    def with_binding_score_map(
        self, binding_score_map: dict[T_STATE, float]
    ) -> "Colored_DB_FSM_Reducer":
        self.binding_score_map = binding_score_map
        return self

    def _ensure_binding_map(self):
        if len(self.binding_score_map) == 0:
            raise RuntimeError("Set binding_score_map")

    def get_approx_binding_score_map(self) -> dict[T_STATE, float]:
        self._ensure_finished()
        self._ensure_binding_map()

        if self._approx_binding_score_map is None:
            self._approx_binding_score_map: dict[T_STATE, float] = {
                v: statistics.mean(
                    self.binding_score_map[w] for w in self._f_inverse[v]
                )
                for v in self._reduced_fsm.V
            }

        return self._approx_binding_score_map

    def get_scores(self) -> tuple[np.ndarray, np.ndarray]:
        approx_binding_score_map: dict[T_STATE, float] = (
            self.get_approx_binding_score_map()
        )
        origin_scores, approx_scores = get_scores(
            self.binding_score_map, approx_binding_score_map, self._f
        )

        return origin_scores, approx_scores

    def get_binding_score_mse(self) -> float:
        origin_scores, approx_scores = self.get_scores()
        mse = round(mean_squared_error(origin_scores, approx_scores), 6)

        return mse

    def get_initial_potential_merging_sets(
        self, eps: float = 1.0
    ) -> list[list[T_STATE]]:
        self._ensure_binding_map()

        potential_merging_sets: list[list[T_STATE]] = []
        states_by_trans: dict[tuple, list[T_STATE]] = defaultdict(list)
        for v in self.origin_fsm.V:
            v_trans = self.origin_fsm.get_outgoing_transitions(v)
            states_by_trans[v_trans].append(v)

        for grouped_states in states_by_trans.values():
            sorted_scores = sorted(
                [(v, self.binding_score_map[v]) for v in grouped_states],
                key=itemgetter(1),
            )
            for i, (v, score) in enumerate(sorted_scores[:-1]):
                next_v, next_score = sorted_scores[i + 1]
                if next_score - score <= eps and self.origin_fsm.c(
                    v
                ) != self.origin_fsm.c(next_v):
                    potential_merging_sets.append([v, next_v])

        return potential_merging_sets

    def get_potential_merging_sets(self, eps: float = 1.0) -> list[list[T_STATE]]:
        self._ensure_finished()
        self._ensure_binding_map()

        potential_merging_sets: list[list[T_STATE]] = []
        states_by_trans: dict[tuple, list[T_STATE]] = defaultdict(list)
        for v in self._reduced_fsm.V:
            v_trans = self._reduced_fsm.get_outgoing_transitions(v)
            states_by_trans[v_trans].append(v)

        for grouped_states in states_by_trans.values():
            sorted_scores = sorted(
                [
                    (w, self.binding_score_map[w])
                    for v in grouped_states
                    for w in self.inverse_states_mapping[v]
                ],
                key=itemgetter(1),
            )
            pms: list[tuple[T_STATE, float]] = [sorted_scores[0]]
            vs: set[T_STATE] = {self.states_mapping[pms[-1][0]]}
            for w, score in sorted_scores[1:]:
                prev_score = pms[-1][1]
                v = self.states_mapping[w]
                if score - prev_score <= eps and v not in vs:
                    pms.append((w, score))
                    vs.add(v)
                else:
                    if len(pms) > 1:
                        potential_merging_sets.append(list(map(itemgetter(0), pms)))
                    pms = [(w, score)]
                    vs = {self.states_mapping[pms[-1][0]]}
        if len(pms) > 1:
            potential_merging_sets.append(list(map(itemgetter(0), pms)))

        return potential_merging_sets

    def __str__(self) -> str:
        approx_binding_score_map: dict[T_STATE, float] = (
            self.get_approx_binding_score_map()
        )
        sorted_scores = dict(
            sorted(
                [
                    (v, round(float(score), 3))
                    for v, score in approx_binding_score_map.items()
                ],
                key=itemgetter(1),
            )
        )

        def state_str(v) -> str:
            clr = int(self._reduced_fsm.c(v))
            return f"{v} [{clr:2d} | {len(self._f_inverse[v]):3d}]"

        states_list = []
        for v, score in sorted_scores.items():
            v_trans = "    ".join(
                map(state_str, self._reduced_fsm.get_outgoing_transitions(v))
            )
            states_list.append(f"{state_str(v)} {score:>10.3f}    {v_trans}")

        fsm_display = "\n".join(states_list)

        return fsm_display

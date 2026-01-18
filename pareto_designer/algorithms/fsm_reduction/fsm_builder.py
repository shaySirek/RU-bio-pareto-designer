from typing import Generic, Callable
from bisect import bisect_left
from operator import itemgetter
from copy import deepcopy

from loguru import logger

from pareto_designer.algorithms.fsm_reduction.union_find import UnionFind
from pareto_designer.algorithms.fsm import FSM, ColoredFSM, T_STATE, T_CHAR, T_COLOR
from pareto_designer.shared.consts import STATE_MINIMUM_COMMON_SUFFIX_LENGTH


class FSM_Merger(Generic[T_STATE, T_CHAR]):
    def __init__(
        self,
        ds: UnionFind[T_STATE],
        current_fsm: FSM[T_STATE, T_CHAR],
        add_mergeable_set_fn: Callable[[set[T_STATE]], None],
    ):
        self._ds = ds
        self._reduced_fsm = deepcopy(current_fsm)
        self._f: dict[T_STATE, T_STATE] = {v: v for v in current_fsm.V}
        self._f_inverse: dict[T_STATE, list[T_STATE]] = {v: [v] for v in current_fsm.V}
        self._add_mergeable_set_fn = add_mergeable_set_fn
        self._n_mergeable_sets: int = 0

    def merge_and_get_consider_for_merge(self, mergeable_set: list[T_STATE]):
        candidate_set: set[T_STATE] = set()
        v = mergeable_set.pop(0)
        logger.debug(f"Merging {mergeable_set} into {v}")

        for u, _ in self._reduced_fsm.pred(v):
            candidate_set.add(u)
        for w in mergeable_set:
            for u, sigma in self._reduced_fsm.pred(w):
                candidate_set.add(u)
                self._reduced_fsm.set_transition(u, sigma, v)
            self._ds.remove(w)
            self._update_mapping(v, w)

        # delete merged states from FSM, including the functions t and pred
        self._reduced_fsm.clean_merged_states(mergeable_set, v)
        # no need to consider deleted (merged) states as candidates for future merges
        candidate_set.difference_update(mergeable_set)

        return lambda: self._consider_sets_after_merge([candidate_set])

    def _update_mapping(self, v: T_STATE, w: T_STATE) -> None:
        self._f[w] = v
        self._f_inverse[v].append(w)
        prev_to_w = self._f_inverse.pop(w)
        for x in prev_to_w:
            self._f[x] = v
        self._f_inverse[v].extend(prev_to_w)

    def _consider_sets_after_merge(self, candidate_sets: list[set[T_STATE]]):
        for candidate_set in candidate_sets:
            self._consider_for_merge(candidate_set)
        return self

    def _get_state_equivalence_class(self, u: T_STATE):
        out_t = self._reduced_fsm.get_outgoing_transitions(u)
        if STATE_MINIMUM_COMMON_SUFFIX_LENGTH > 0:
            common_suffix = u[-STATE_MINIMUM_COMMON_SUFFIX_LENGTH:]
            return (out_t, common_suffix)
        return out_t

    def _consider_for_merge(self, candidate_set: set[T_STATE]) -> None:
        sorted_candidates: list[tuple[T_STATE, ...]] = []
        for u in candidate_set:
            eq_class = self._get_state_equivalence_class(u)
            v = self._ds.find(u)
            insert_sorted_unique(sorted_candidates, v, eq_class)

        set_for_merge: set[T_STATE] = set()
        prev_eq_class = None
        for v, eq_class in sorted_candidates:
            if prev_eq_class == eq_class:
                set_for_merge.add(v)
            else:
                self._cond_add_set(set_for_merge)
                set_for_merge = {v}
                prev_eq_class = eq_class
        self._cond_add_set(set_for_merge)

    def _cond_add_set(self, s: set[T_STATE]):
        if len(s) > 1:
            self._add_mergeable_set_fn(s)
            self._n_mergeable_sets += 1

    def get_current(self):
        # set v_init to an arbitrary state, as done when initializing FSM
        self._reduced_fsm.v_init = list(self._reduced_fsm.V)[0]
        return self._reduced_fsm, self._n_mergeable_sets, self._f, self._f_inverse


class ColoredFSM_Merger(FSM_Merger, Generic[T_STATE, T_CHAR, T_COLOR]):
    def __init__(
        self,
        ds: UnionFind[T_STATE],
        current_fsm: ColoredFSM[T_STATE, T_CHAR, T_COLOR],
        add_mergeable_set_fn: Callable[[set[T_STATE]], None],
    ):
        self._ds = ds
        self._reduced_fsm = deepcopy(current_fsm)
        self._f: dict[T_STATE, T_STATE] = {v: v for v in current_fsm.V}
        self._f_inverse: dict[T_STATE, list[T_STATE]] = {v: [v] for v in current_fsm.V}
        self._add_mergeable_set_fn = add_mergeable_set_fn
        self._n_mergeable_sets: int = 0

    def merge_and_consider_for_merge(self, mergeable_set: list[T_STATE]):
        candidate_sets: dict[T_COLOR, set[T_STATE]] = {}
        v = mergeable_set.pop(0)
        logger.debug(f"Merging {mergeable_set} into {v}")

        for col in self._reduced_fsm.C:
            candidate_sets[col] = set()
        for u, _ in self._reduced_fsm.pred(v):
            candidate_sets[self._reduced_fsm.c(u)].add(u)
        for w in mergeable_set:
            for u, sigma in self._reduced_fsm.pred(w):
                candidate_sets[self._reduced_fsm.c(u)].add(u)
                self._reduced_fsm.set_transition(u, sigma, v)
            self._ds.remove(w)
            self._update_mapping(v, w)

        # delete merged states from FSM, including the functions t and pred
        self._reduced_fsm.clean_merged_states(mergeable_set, v)
        # no need to consider deleted (merged) states as candidates for future merges
        candidate_sets[self._reduced_fsm.c(v)].difference_update(mergeable_set)

        return self._consider_sets_after_merge(list(candidate_sets.values()))


def insert_sorted_unique(
    sorted_list: list[tuple[T_STATE, ...]],
    x: T_STATE,
    key,
):
    item = (x, key)
    index = bisect_left(sorted_list, key, key=itemgetter(1))
    # Ensure uniqueness
    if index < len(sorted_list) and sorted_list[index] == item:
        return
    sorted_list.insert(index, item)

from typing import Generic, Generator
import sys

import numpy as np
from loguru import logger

from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.algorithms.fsm import FSM, T_STATE, T_CHAR
from pareto_designer.algorithms.seq_design.dp_matrix import DP_Matrix
from pareto_designer.algorithms.seq_design.types import (
    T_SOLUTION,
    T_LAZY_SOL_ITER_FACTORY,
)
from pareto_designer.algorithms.seq_design.util import find_po_from_sorted_iters


class ParetoOptimalDesign(Generic[T_STATE, T_CHAR]):
    """This class implements the algorithm to find Pareto-optimal sequences."""

    def __init__(
        self,
        sequence_length: int,
        score_function: ScoreFunction,
        fsm: FSM[T_STATE, T_CHAR],
        binding_score_map: dict[T_STATE, float],
        motif_length: int,
        verbose: bool = True,
    ):
        self.score_function = score_function
        self.fsm = fsm
        self.binding_score_map = binding_score_map
        self.n = sequence_length
        self.m = motif_length
        self.verbose = verbose

        logger.remove()
        logger.add(sys.stdout, level="INFO")

    def find_pareto_optimal(self) -> set[tuple[str, T_SOLUTION]]:
        with DP_Matrix(self.fsm, self.n) as dp:
            self._dp_matrix = dp
            logger.info("Calculating DP matrix...")
            self._update_step_phase_1()
            self._update_step_phase_2()
            logger.info("Reconstructing Pareto-optimal solutions...")
            self._po_set = self._dp_matrix.reconstruct_po_set(find_po_from_sorted_iters)

        return self._po_set

    def _update_step_phase_1(self):
        for i in range(1, self.m):
            self._dp_matrix.start_row()
            for v in self.fsm.V:
                max_f = -float("inf")
                back_ptr = None
                for u, sigma in self.fsm.pred(v):
                    cell = self._dp_matrix.get(u)
                    if len(cell) == 0:
                        continue

                    f_score_current = cell["f"] + self.score_function(i - 1, u, sigma)
                    local_max_idx = np.argmax(f_score_current)
                    local_max_f = f_score_current[local_max_idx]
                    if local_max_f > max_f:
                        max_f = local_max_f
                        back_ptr = ((u, sigma), int(local_max_idx))

                if back_ptr is not None:
                    self._dp_matrix.update(v, [(max_f, 0.0)], [[back_ptr]])

            self._dp_matrix.end_row(i)

    def _update_step_phase_2(self):
        for i in range(self.m, self.n + 1):
            self._dp_matrix.start_row()
            for v in self.fsm.V:
                sorted_po_scores, sorted_back_ptrs = find_po_from_sorted_iters(
                    self._get_sorted_scores_with_back_ptrs(i, v)
                )
                self._dp_matrix.update(v, sorted_po_scores, sorted_back_ptrs)

            self._dp_matrix.end_row(i)

    def _get_sorted_scores_with_back_ptrs(
        self, i: int, v: T_STATE
    ) -> Generator[T_LAZY_SOL_ITER_FACTORY, None, None]:
        b_0 = self.binding_score_map[v]
        for u, sigma in self.fsm.pred(v):
            f_0 = self.score_function(i - 1, u, sigma)
            dp_cell = self._dp_matrix.get(u)
            if len(dp_cell) == 0:
                continue

            def sorted_scores_factory(
                scores_arr=dp_cell, f_off=f_0, b_off=b_0, u_state=u, base=sigma
            ):
                # Secondary sort: binding (b) ASC, Primary sort: functional (f) DESC
                indices = np.lexsort(
                    (scores_arr["b"] + b_off, -(scores_arr["f"] + f_off))
                )
                for idx in indices:
                    score = scores_arr[idx]
                    yield (
                        (score["f"] + f_off, score["b"] + b_off),
                        ((u_state, base), int(idx)),
                    )

            yield sorted_scores_factory

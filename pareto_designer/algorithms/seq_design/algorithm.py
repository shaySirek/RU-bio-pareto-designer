from typing import Generator
import sys
import threading
from functools import partial

import numpy as np
from loguru import logger

from pareto_designer.algorithms.fsm import T_STATE
from pareto_designer.algorithms.seq_design.dp_matrix import DP_Matrix, get_flush_every
from pareto_designer.algorithms.seq_design.types import (
    T_SOLUTION,
    T_LAZY_SOL_ITER_FACTORY,
)
from pareto_designer.algorithms.seq_design.util import find_po_from_sorted_iters
from pareto_designer.models.context import DesignContext
from pareto_designer.views.pareto_set import ParetoSet


class ParetoOptimalDesign:
    """This class implements the algorithm to find Pareto-optimal sequences."""

    def __init__(self, ctx: DesignContext):
        self._score_function = ctx.score_function
        self._fsm = ctx.fsm_ctx.fsm
        self._binding_score_map = ctx.fsm_ctx.binding_score_map
        self._n = ctx.sequence_length
        self._m = ctx.fsm_ctx.motif_length
        self._l = ctx.run_ctx.solutions_limit
        self._output_path = ctx.run_ctx.output_path
        self._find_po_func = partial(find_po_from_sorted_iters, limit=self._l)
        self._flush_every = get_flush_every(self._fsm, self._l)
        self._n_pruned = None
        self._pareto_set_reporter = (
            ParetoSet(ctx.run_ctx.output_path) if self._l == 0 else None
        )

        logger.remove()
        logger.add(sys.stdout, level="INFO")

    def find_pareto_optimal(self) -> set[tuple[str, T_SOLUTION]]:
        with DP_Matrix(
            self._fsm,
            self._n,
            self._flush_every,
        ) as dp:
            self._dp_matrix = dp
            logger.info("Calculating DP matrix...")
            self._update_step_phase_1()
            self._update_step_phase_2()
            if self._pareto_set_reporter:
                threading.Thread(target=self._pareto_set_reporter.plot).start()

            logger.info("Reconstructing Pareto-optimal solutions...")
            self._po_set = self._dp_matrix.reconstruct_po_set(self._find_po_func)

        return self._po_set

    def _update_step_phase_1(self):
        for i in range(1, self._m):
            self._start_row()
            for v in self._fsm.V:
                max_f = -float("inf")
                back_ptr = None
                for u, sigma in self._fsm.pred(v):
                    cell = self._dp_matrix.get(u)
                    if len(cell) == 0:
                        continue

                    f_score_current = cell["f"] + self._score_function(i - 1, u, sigma)
                    local_max_idx = np.argmax(f_score_current)
                    local_max_f = f_score_current[local_max_idx]
                    if local_max_f > max_f:
                        max_f = local_max_f
                        back_ptr = ((u, sigma), int(local_max_idx))

                if back_ptr is not None:
                    self._dp_matrix.update(v, [(max_f, 0.0)], [[back_ptr]])

            self._end_row(i)

    def _update_step_phase_2(self):
        for i in range(self._m, self._n + 1):
            self._start_row()
            for v in self._fsm.V:
                sorted_po_scores, sorted_back_ptrs = self._find_po_func(
                    self._get_sorted_scores_with_back_ptrs(i, v)
                )
                self._dp_matrix.update(v, sorted_po_scores, sorted_back_ptrs)

            self._end_row(i)

    def _start_row(self):
        self._dp_matrix.start_row()
        self._n_pruned = 0

    def _end_row(self, i: int):
        sizes = self._dp_matrix.end_row(i)
        if self._pareto_set_reporter:
            self._pareto_set_reporter.report_row_size(i, sizes, self._n_pruned)

    def _get_sorted_scores_with_back_ptrs(
        self, i: int, v: T_STATE
    ) -> Generator[T_LAZY_SOL_ITER_FACTORY, None, None]:
        b_0 = self._binding_score_map[v]
        for u, sigma in self._fsm.pred(v):
            dp_cell = self._dp_matrix.get(u)
            if len(dp_cell) == 0:
                continue

            f_0 = self._score_function(i - 1, u, sigma)
            if f_0 == -float("inf"):
                self._n_pruned += len(dp_cell)
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

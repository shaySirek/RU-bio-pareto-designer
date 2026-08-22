import sys
import threading

import numpy as np
from loguru import logger

from pareto_designer.algorithms.seq_design.dp_matrix import DP_Matrix, get_flush_every
from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.algorithms.seq_design.util import (
    PTR_DTYPE,
    SCORE_DTYPE,
    find_po_from_arrays,
)
from pareto_designer.models.context import DesignContext
from pareto_designer.views.pareto_set import ParetoSet


class ParetoOptimalDesign:
    """This class implements the algorithm to find Pareto-optimal sequences."""

    def __init__(self, ctx: DesignContext):
        self._score_function = ctx.score_function
        self._fsm = ctx.fsm_ctx.fsm
        self._binding_score_map = ctx.fsm_ctx.binding_score_map
        self._binding_score_space = ctx.fsm_ctx.binding_score_space
        self._n = ctx.sequence_length
        self._m = ctx.fsm_ctx.motif_length
        self._sampler = ctx.run_ctx.sampler
        self._output_path = ctx.run_ctx.output_path

        self._state_list = list(self._fsm.V)
        self._alphabet = list(self._fsm.Sigma)
        self._num_states = len(self._state_list)
        self._b_off = np.array(
            [self._binding_score_map[v] for v in self._state_list], dtype=np.float64
        )
        state_to_idx = {s: i for i, s in enumerate(self._state_list)}
        sigma_to_idx = {s: i for i, s in enumerate(self._alphabet)}
        pred_pairs: list[tuple[int, int]] = []
        pred_off = [0]
        for v in self._state_list:
            for u, sigma in self._fsm.pred(v) or ():
                pred_pairs.append((state_to_idx[u], sigma_to_idx[sigma]))
            pred_off.append(len(pred_pairs))
        self._pred_us = (
            np.asarray(pred_pairs, dtype=np.int32).reshape(-1, 2)
            if pred_pairs
            else np.empty((0, 2), dtype=np.int32)
        )
        self._pred_off = np.asarray(pred_off, dtype=np.int32)

        self._flush_every = get_flush_every(self._fsm, self._sampler.k)
        self._pruned_sizes = None
        self._pareto_set_reporter = (
            ParetoSet(ctx.run_ctx.output_path) if self._sampler.k == 0 else None
        )

        logger.remove()
        logger.add(sys.stdout, level="INFO")

    def find_pareto_optimal(self) -> set[tuple[str, T_SOLUTION]]:
        with DP_Matrix(
            self._fsm,
            self._n,
            self._flush_every,
            state_order=self._state_list,
            alphabet=self._alphabet,
        ) as dp:
            self._dp_matrix = dp
            self._dp_matrix.start_row()
            self._dp_matrix.update(
                self._fsm.v_init, [(0.0, self._binding_score_space.Identity)], None
            )
            self._dp_matrix.end_row(0)
            logger.info("Calculating DP matrix...")
            self._update_step_phase_1()
            self._update_step_phase_2()
            if self._pareto_set_reporter:
                threading.Thread(target=self._pareto_set_reporter.plot).start()

            logger.info("Reconstructing Pareto-optimal solutions...")
            self._po_set = self._dp_matrix.reconstruct_po_set(self._sampler, self._n)

        return self._po_set

    def _pred_cells(self, v_idx: int, i: int):
        s, e = self._pred_off[v_idx], self._pred_off[v_idx + 1]
        preds = self._pred_us[s:e]
        for p in range(e - s):
            u_idx = int(preds[p, 0])
            sigma_idx = int(preds[p, 1])
            cell = self._dp_matrix.get_idx(u_idx)
            if len(cell) == 0:
                continue
            f_off = self._score_function(
                i - 1, self._state_list[u_idx], self._alphabet[sigma_idx]
            )
            yield u_idx, sigma_idx, cell, f_off

    def _update_step_phase_1(self):
        identity = self._binding_score_space.Identity
        for i in range(1, self._m):
            self._start_row()
            for v_idx in range(self._num_states):
                max_f = -float("inf")
                back_ptr = None
                for u_idx, sigma_idx, cell, f_off in self._pred_cells(v_idx, i):
                    f_score_current = cell["f"] + f_off
                    local_max_idx = int(np.argmax(f_score_current))
                    local_max_f = float(f_score_current[local_max_idx])
                    if local_max_f > max_f:
                        max_f = local_max_f
                        back_ptr = (u_idx, sigma_idx, local_max_idx)

                if back_ptr is not None:
                    scores = np.empty(1, dtype=SCORE_DTYPE)
                    scores["f"] = max_f
                    scores["b"] = identity
                    ptrs = np.empty(1, dtype=PTR_DTYPE)
                    ptrs["u_idx"] = back_ptr[0]
                    ptrs["sigma_idx"] = back_ptr[1]
                    ptrs["j"] = back_ptr[2]
                    self._dp_matrix.update_from_arrays(v_idx, scores, [ptrs])

            self._end_row(i)

    def _update_step_phase_2(self):
        add = self._binding_score_space._add
        empty_scores = np.empty(0, dtype=SCORE_DTYPE)
        for i in range(self._m, self._n + 1):
            self._start_row()
            n_pruned_row: list[int] = []
            for v_idx in range(self._num_states):
                n_pruned = 0
                b_off = self._b_off[v_idx]
                f_parts: list[np.ndarray] = []
                b_parts: list[np.ndarray] = []
                u_parts: list[np.ndarray] = []
                sig_parts: list[np.ndarray] = []
                j_parts: list[np.ndarray] = []
                for u_idx, sigma_idx, cell, f_off in self._pred_cells(v_idx, i):
                    if f_off == -float("inf"):
                        n_pruned += len(cell)
                        continue
                    n_cell = len(cell)
                    f_parts.append(cell["f"] + f_off)
                    b_parts.append(np.asarray(add(cell["b"], b_off), dtype=np.float64))
                    u_parts.append(np.full(n_cell, u_idx, dtype=np.int32))
                    sig_parts.append(np.full(n_cell, sigma_idx, dtype=np.int32))
                    j_parts.append(np.arange(n_cell, dtype=np.int32))

                n_pruned_row.append(n_pruned)
                if not f_parts:
                    self._dp_matrix.update_from_arrays(v_idx, empty_scores, None)
                    continue

                scores, ptr_groups = find_po_from_arrays(
                    np.concatenate(f_parts),
                    np.concatenate(b_parts),
                    np.concatenate(u_parts),
                    np.concatenate(sig_parts),
                    np.concatenate(j_parts),
                    sampler=self._sampler,
                    position=i,
                )
                self._dp_matrix.update_from_arrays(v_idx, scores, ptr_groups)

            self._pruned_sizes = n_pruned_row
            self._end_row(i)

    def _start_row(self):
        self._dp_matrix.start_row()
        self._pruned_sizes = []

    def _end_row(self, i: int):
        sizes = self._dp_matrix.end_row(i)
        if self._pareto_set_reporter:
            self._pareto_set_reporter.report_row_size(i, sizes, self._pruned_sizes)

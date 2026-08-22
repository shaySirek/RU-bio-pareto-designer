from copy import deepcopy
from math import ceil
from typing import Iterator, Sequence

from loguru import logger

from pareto_designer.algorithms.spaces import ScoreSpaceOption
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.models.motif import BindingMotif, StrandForBindingScore
from pareto_designer.models.context import FSMContext
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm


class FSMBuilder:
    def __init__(self):
        self._matrix_id: str | None = None
        self._strand: StrandForBindingScore = StrandForBindingScore.Double
        self._binding_score_space: ScoreSpaceOption | None = None
        self._max_total_error: float | None = None
        self._reduction_ratios: list[float] = []
        self._hit_pvalue: float = 2e-3
        self._motif: BindingMotif | None = None
        self._fsm: FSM | None = None
        self._origin_binding_score_map: dict[str, float] | None = None

    def with_matrix_id(self, matrix_id: str) -> "FSMBuilder":
        self._matrix_id = matrix_id
        return self

    def with_binding_score_space(
        self, score_space_option: ScoreSpaceOption
    ) -> "FSMBuilder":
        self._binding_score_space = score_space_option
        return self

    def with_hit_pvalue(self, hit_pvalue: float) -> "FSMBuilder":
        self._hit_pvalue = hit_pvalue
        return self

    def with_fsm_reduction(
        self,
        max_total_error: float | None,
        reduction_ratio_threshold: float | Sequence[float] | None,
    ) -> "FSMBuilder":
        self._max_total_error = max_total_error
        if reduction_ratio_threshold is None:
            self._reduction_ratios = []
        elif isinstance(reduction_ratio_threshold, (int, float)):
            self._reduction_ratios = [float(reduction_ratio_threshold)]
        else:
            self._reduction_ratios = [float(r) for r in reduction_ratio_threshold]
        return self

    def build(self, dry_run: bool = False) -> FSMContext:
        return next(self.iter_contexts(dry_run))

    def iter_contexts(self, dry_run: bool = False) -> Iterator[FSMContext]:
        if self._matrix_id is None:
            raise ValueError("Cannot build FSM: motif is not set.")

        self._motif, db_fsm, origin_map = get_binding_motif_fsm(
            self._matrix_id, self._strand
        )
        self._fsm = db_fsm
        self._origin_binding_score_map = origin_map
        db_size = len(db_fsm.V)
        ratios = self._reduction_ratios or (
            [0.0] if self._max_total_error is None else []
        )
        size_targets: dict[int, float] = {}
        for ratio in ratios:
            if not ratio:
                size_targets[db_size] = 0.0
            else:
                size_targets[ceil((1 - ratio) * db_size)] = ratio

        multiple_sizes = len(size_targets) > 1
        use_max_error = self._max_total_error is not None and not multiple_sizes

        if dry_run:
            if not size_targets and use_max_error:
                yield self._make_context(
                    db_fsm, origin_map, 0.0, "db_fsm", 0.0, db_size, db_size
                )
                return
            for n_states, ratio in sorted(size_targets.items(), reverse=True):
                local_id = "db_fsm" if ratio == 0.0 else f"reduced_fsm_{n_states}"
                yield self._make_context(
                    db_fsm, origin_map, 0.0, local_id, ratio, db_size, n_states
                )
            return

        remaining: dict[int, float] = dict(size_targets)
        if db_size in remaining:
            yield self._make_context(
                db_fsm, origin_map, 0.0, "db_fsm", remaining[db_size], db_size, db_size
            )
            del remaining[db_size]

        if not remaining and not use_max_error:
            return

        fsm_reducer = DB_FSM_Reducer[str, str](
            db_fsm,
            origin_map,
            self._binding_score_space.get_space(),
            self._matrix_id,
        )
        for (
            reduced_fsm,
            reduced_fsm_err,
            (reduced_fsm_binding_score_map, _, _),
        ) in fsm_reducer.find_reduced_fsms():
            n_states = len(reduced_fsm.V)
            hit_error = use_max_error and reduced_fsm_err >= self._max_total_error
            hit_size = n_states in remaining
            if not (hit_error or hit_size):
                if remaining and n_states < min(remaining):
                    break
                continue

            ratio = remaining.pop(n_states, next(iter(size_targets.values()), 0.0))
            ctx = self._make_context(
                deepcopy(reduced_fsm),
                reduced_fsm_binding_score_map.copy(),
                reduced_fsm_err,
                f"reduced_fsm_{n_states}",
                ratio,
                db_size,
                n_states,
            )
            logger.info(
                f"Reduced DB FSM to FSM with {n_states} states and err={reduced_fsm_err:.3f}"
            )
            yield ctx
            if hit_error or (not remaining and not use_max_error):
                break

    def _make_context(
        self,
        fsm: FSM,
        binding_score_map: dict[str, float],
        err: float,
        local_id: str,
        reduce_fsm_by: float,
        db_fsm_size: int,
        reported_size: int,
    ) -> FSMContext:
        return FSMContext(
            self._motif,
            self._origin_binding_score_map,
            binding_score_map,
            self._binding_score_space.get_space(),
            err,
            fsm,
            f"{self._binding_score_space.value}_{local_id}",
            reduce_fsm_by,
            db_fsm_size,
            self._hit_pvalue,
            reported_size,
        )

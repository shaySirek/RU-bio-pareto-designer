from copy import deepcopy
from math import ceil
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
        self._matrix_id: str = None
        self._strand: StrandForBindingScore = StrandForBindingScore.Double
        self._binding_score_space: ScoreSpaceOption = None
        self._max_total_error: float = None
        self._reduction_ratio_threshold: float = None
        self._motif: BindingMotif = None
        self._fsm: FSM = None
        self._binding_score_map: dict[str, float] = None
        self._fsm_id: str = None

    @property
    def _fsm_size(self) -> int:
        return len(self._fsm.V)

    def with_matrix_id(self, matrix_id: str) -> "FSMBuilder":
        self._matrix_id = matrix_id
        return self

    def with_binding_score_space(
        self, score_space_option: ScoreSpaceOption
    ) -> "FSMBuilder":
        self._binding_score_space = score_space_option
        return self

    def with_fsm_reduction(
        self,
        max_total_error: float,
        reduction_ratio_threshold: float,
    ) -> "FSMBuilder":
        self._max_total_error = max_total_error
        self._reduction_ratio_threshold = reduction_ratio_threshold
        return self

    def build(self, dry_run: bool = False) -> FSMContext:
        if self._matrix_id is None:
            raise ValueError("Cannot build FSM: motif is not set.")

        self._motif, self._fsm, self._binding_score_map = get_binding_motif_fsm(
            self._matrix_id, self._strand
        )
        self._fsm_id = "db_fsm"
        self._reduce_fsm(dry_run)
        self._fsm_id = f"{self._binding_score_space.value}_{self._fsm_id}"
        return FSMContext(
            self._motif,
            self._binding_score_map,
            self._binding_score_space.get_space(),
            self._fsm,
            self._fsm_id,
        )

    def _reduce_fsm(self, dry_run: bool = False):
        if not (self._max_total_error or self._reduction_ratio_threshold):
            return

        err = 0.0
        min_fsm_size = 0
        if self._reduction_ratio_threshold is not None:
            min_fsm_size = ceil((1 - self._reduction_ratio_threshold) * self._fsm_size)
            if dry_run:
                self._fsm_id = f"reduced_fsm_{min_fsm_size}"
                return

        fsm_reducer = DB_FSM_Reducer[str, str](
            self._fsm,
            self._binding_score_map,
            self._binding_score_space.get_space(),
            self._matrix_id,
        )
        for (
            reduced_fsm,
            reduced_fsm_err,
            (reduced_fsm_binding_score_map, _, _),
        ) in fsm_reducer.find_reduced_fsms():
            fsm_size = len(reduced_fsm.V)
            if (
                self._max_total_error is not None
                and err >= self._reduction_ratio_threshold
            ) or fsm_size == min_fsm_size:
                err = reduced_fsm_err
                self._fsm = deepcopy(reduced_fsm)
                self._binding_score_map = reduced_fsm_binding_score_map.copy()
                break

        self._fsm_id = f"reduced_fsm_{self._fsm_size}"
        logger.info(
            f"Reduced DB FSM to FSM with {self._fsm_size} states and err={err:.3f}"
        )

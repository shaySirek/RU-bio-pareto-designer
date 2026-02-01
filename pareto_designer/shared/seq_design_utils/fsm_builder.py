from loguru import logger
from copy import deepcopy

from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.models.motif import BindingMotif
from pareto_designer.models.context import FSMContext
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.shared.fsm_utils.reduced_fsms_generator import get_reduced_fsms


class FSMBuilder:
    def __init__(self):
        self._matrix_id: str = None
        self._delta_mse_threshold: float = None
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

    def with_fsm_reduction(
        self,
        delta_mse_threshold: float = 0.5,
        reduction_ratio_threshold: float = 0.5,
    ) -> "FSMBuilder":
        self._delta_mse_threshold = delta_mse_threshold
        self._reduction_ratio_threshold = reduction_ratio_threshold
        return self

    def build(self) -> FSMContext:
        if self._matrix_id is None:
            raise ValueError("Cannot build FSM: motif is not set.")

        self._motif, self._fsm, self._binding_score_map = get_binding_motif_fsm(
            self._matrix_id
        )
        self._fsm_id = "db_fsm"
        self._reduce_fsm()
        return FSMContext(self._motif, self._binding_score_map, self._fsm, self._fsm_id)

    def _reduce_fsm(self):
        if not (self._delta_mse_threshold or self._reduction_ratio_threshold):
            return

        fsm_reducer = DB_FSM_Reducer[str, str](
            self._fsm, self._binding_score_map, self._matrix_id
        )
        reduced_fsms_iter = get_reduced_fsms(
            fsm_reducer,
            self._delta_mse_threshold,
            self._reduction_ratio_threshold,
        )
        fsm, binding_score_map, mse = next(reduced_fsms_iter)
        self._fsm = deepcopy(fsm)
        self._binding_score_map = binding_score_map.copy()
        mse_at_trivial_fsm = mse
        for _, _, current_mse in reduced_fsms_iter:
            mse_at_trivial_fsm = current_mse
        reduction_efficiency = (
            1 - (mse / mse_at_trivial_fsm) if mse_at_trivial_fsm > 0 else 1
        )
        self._fsm_id = f"reduced_fsm_{self._fsm_size}"
        logger.info(
            f"Reduced DB FSM to FSM with {self._fsm_size} states and MSE={mse:.3f} (reduction efficiency={reduction_efficiency:.3f})"
        )

from loguru import logger
from pathlib import Path
from copy import deepcopy

from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.algorithms.seq_design.algorithm import ParetoOptimalDesign
from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.models.motif import BindingMotif
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.shared.fsm_utils.reduced_fsms_generator import get_reduced_fsms
from pareto_designer.shared.prof import run_with_timing, format_duration
from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.models.pareto_front import RunContext
from pareto_designer.shared.seq_design_utils.export import export_solutions


class SequenceDesigner:
    def __init__(self):
        self._sequence_id: str = None
        self._score_function_builder: ScoreFunctionBuilder = None
        self._score_function: ScoreFunction = None
        self._motif: BindingMotif = None
        self._fsm: FSM = None
        self._binding_score_map: dict[str, float] = None
        self._fsm_id: str = None
        self._solutions_limit: int = 0  # 0 -> unbounded
        self._solutions: set[tuple[str, T_SOLUTION]] = None
        self._duration: float = None

    @property
    def _target_sequence(self) -> str:
        return self._score_function.target_sequence

    @property
    def _sequence_length(self) -> int:
        return len(self._target_sequence)

    @property
    def _motif_id(self) -> str:
        return self._motif.matrix_id

    @property
    def _motif_length(self) -> int:
        return self._motif.length

    @property
    def _alphabet(self) -> list[str]:
        return self._motif.alphabet

    @property
    def _fsm_size(self) -> int:
        return len(self._fsm.V)

    @property
    def _runtime(self) -> str:
        return format_duration(int(self._duration))

    def with_target_sequence(
        self, sequence_file: Path, score_function_builder: ScoreFunctionBuilder
    ) -> "SequenceDesigner":
        self._sequence_id = sequence_file.stem
        self._score_function_builder = score_function_builder.with_target_sequence(
            sequence_file
        )
        return self

    def with_binding_motif(self, matrix_id: str) -> "SequenceDesigner":
        self._motif, self._fsm, self._binding_score_map = get_binding_motif_fsm(
            matrix_id
        )
        self._fsm_id = "db_fsm"
        return self

    def with_reduced_fsm(
        self,
        delta_mse_threshold: float = 0.5,
        reduction_ratio_threshold: float = 0.5,
    ) -> "SequenceDesigner":
        if self._motif is None:
            raise ValueError("Cannot reduce FSM: motif is not set.")

        fsm_reducer = DB_FSM_Reducer[str, str](
            self._fsm, self._binding_score_map, self._motif_id
        )
        reduced_fsms_iter = get_reduced_fsms(
            fsm_reducer,
            delta_mse_threshold,
            reduction_ratio_threshold,
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
        return self

    def with_solutions_limit(self, solutions_limit: int) -> "SequenceDesigner":
        self._solutions_limit = solutions_limit
        return self

    def run(self) -> RunContext:
        self._score_function = self._score_function_builder.build()
        designer = ParetoOptimalDesign[str, str](
            self._sequence_length,
            self._score_function,
            self._fsm,
            self._binding_score_map,
            self._motif_length,
            self._solutions_limit,
        )
        logger.info(
            f"Running algorithm on target sequence {self._sequence_id}"
            f" and binding motif of {self._motif_id}"
            f" [n={self._sequence_length}, |V|={self._fsm_size}, L={self._solutions_limit}]..."
        )
        self._solutions, self._duration = run_with_timing(designer.find_pareto_optimal)
        logger.info(
            f"Found {len(self._solutions)} Pareto-optimal sequences in {self._runtime}"
        )

        ctx = RunContext(
            target_sequence_id=self._sequence_id,
            target_sequence=self._target_sequence,
            orfs=self._score_function.orfs,
            cost_params=self._score_function.params,
            motif_id=self._motif_id,
            fsm_id=self._fsm_id,
            fsm_size=self._fsm_size,
            solutions_limit=self._solutions_limit,
            n_solutions=len(self._solutions),
            runtime=self._runtime,
        )

        export_solutions(self._solutions, ctx, self._score_function, self._motif)

        return ctx

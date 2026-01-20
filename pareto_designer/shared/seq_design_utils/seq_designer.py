from loguru import logger
from pathlib import Path

from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.algorithms.seq_design.algorithm import ParetoOptimalDesign
from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.shared.fsm_utils.reduced_fsms_generator import get_reduced_fsms
from pareto_designer.shared.prof import run_with_timing, format_duration
from pareto_designer.shared.seq_design_utils.export import export_solutions


class SequenceDesigner:
    def __init__(self):
        self._sequence_length: int = None
        self._score_function: ScoreFunction = None
        self._motif_id: str = None
        self._motif: BindingMotif = None
        self._fsm: FSM = None
        self._binding_score_map: dict[str, float] = None
        self._motif_length: int = None
        self._alphabet: list[str] = None
        self._fsm_id: str = None
        self._limit_solutions: int = None

    def with_score_function(self, score_function: ScoreFunction) -> "SequenceDesigner":
        self._sequence_length = len(score_function.target_sequence)
        self._score_function = score_function
        return self

    def with_binding_motif(self, matrix_id: str) -> "SequenceDesigner":
        self._motif_id = matrix_id
        self._motif, self._fsm, self._binding_score_map = get_binding_motif_fsm(
            matrix_id
        )
        self._motif_length = self._motif.length
        self._alphabet = self._motif.alphabet
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
        self._fsm, self._binding_score_map, mse = next(
            get_reduced_fsms(
                fsm_reducer,
                delta_mse_threshold,
                reduction_ratio_threshold,
            )
        )
        n_states = len(self._fsm.V)
        self._fsm_id = f"reduced_fsm_{n_states}"
        logger.info(f"Reduced DB FSM to FSM with {n_states} states and MSE={mse:.3f}")
        return self

    def with_limit_solutions(self, limit_solutions: int) -> "SequenceDesigner":
        self._limit_solutions = limit_solutions
        return self

    def _validate(self):
        required_fields = [
            "_sequence_length",
            "_score_function",
            "_fsm",
            "_binding_score_map",
            "_motif_length",
            "_limit_solutions",
        ]
        missing = [f for f in required_fields if getattr(self, f) is None]
        if missing:
            fields_str = ", ".join(missing)
            raise ValueError(f"Cannot create sequence designer, missing: {fields_str}.")

    def run(self, seq_id: str):
        self._validate()
        output_path = (
            Path("designer_results")
            / seq_id
            / self._motif_id
            / self._fsm_id
            / f"limit_solutions_{self._limit_solutions}"
        )

        designer = ParetoOptimalDesign[str, str](
            self._sequence_length,
            self._score_function,
            self._fsm,
            self._binding_score_map,
            self._motif_length,
            self._limit_solutions,
        )
        logger.info(
            f"Running algorithm on target sequence {seq_id}"
            f" and binding motif of {self._motif_id}"
            f" [n={self._sequence_length}, |V|={len(self._fsm.V)}, L={self._limit_solutions}]..."
        )
        po_set, duration = run_with_timing(designer.find_pareto_optimal)
        logger.info(
            f"Found {len(po_set)} Pareto-optimal sequences in {format_duration(int(duration))}"
        )

        export_solutions(po_set, self._score_function, self._motif_id, output_path)

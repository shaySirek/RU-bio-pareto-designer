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
        self._motif: BindingMotif = None
        self._db_fsm: FSM = None
        self._binding_score_map: dict[str, float] = None
        self._motif_length: int = None
        self._alphabet: list[str] = None
        self._reduced: tuple[FSM, dict[str, float], float] = None

    def with_score_function(self, score_function: ScoreFunction) -> "SequenceDesigner":
        self._sequence_length = len(score_function.target_sequence)
        self._score_function = score_function
        return self

    def with_binding_motif(self, matrix_id: str) -> "SequenceDesigner":
        self._motif, self._db_fsm, self._binding_score_map = get_binding_motif_fsm(
            matrix_id
        )
        self._motif_length = self._motif.length
        self._alphabet = self._motif.alphabet
        return self

    def with_reduced_fsm(
        self,
        delta_mse_threshold: float = 0.5,
        reduction_ratio_threshold: float = 0.875,
    ) -> "SequenceDesigner":
        if self._motif is None:
            raise ValueError("Cannot reduce FSM: motif is not set.")

        fsm_reducer = DB_FSM_Reducer[str, str](
            self._db_fsm, self._binding_score_map, self._motif.matrix_id
        )
        self._reduced = next(
            get_reduced_fsms(
                fsm_reducer,
                delta_mse_threshold=delta_mse_threshold,
                reduction_ratio_threshold=reduction_ratio_threshold,
            )
        )
        return self

    def _validate(self):
        required_fields = [
            "_sequence_length",
            "_score_function",
            "_db_fsm",
            "_binding_score_map",
            "_motif_length",
        ]
        missing = [f for f in required_fields if getattr(self, f) is None]
        if missing:
            fields_str = ", ".join(missing)
            raise ValueError(f"Cannot create sequence designer, missing: {fields_str}.")

    def run(self, seq_id: str):
        self._validate()
        self._output_path = (
            Path("designer_results") / self._motif.matrix_id / seq_id / "runs"
        )

        if self._reduced is None:
            logger.info(
                f"Using the DB FSM ({len(self._db_fsm.V)} states) to find Pareto-optimal sequences"
            )
            self._run_designer(self._db_fsm, self._binding_score_map, "db_fsm")
        else:
            fsm, binding_score_map, mse = self._reduced
            logger.info(
                f"Using a reduced FSM with {len(fsm.V)} states and MSE={mse:.3f} to find Pareto-optimal sequences"
            )
            self._run_designer(fsm, binding_score_map, f"reduced_fsm_{len(fsm.V)}")

    def _run_designer(
        self,
        fsm: FSM,
        binding_score_map: dict[str, float],
        subdir: str,
    ):
        designer = ParetoOptimalDesign[str, str](
            self._sequence_length,
            self._score_function,
            fsm,
            binding_score_map,
            self._motif_length,
        )
        po_set, duration = run_with_timing(designer.find_pareto_optimal)
        logger.info(
            f"Found {len(po_set)} Pareto-optimal sequences in {format_duration(int(duration))}"
        )

        export_solutions(po_set, self._score_function, self._output_path / subdir)

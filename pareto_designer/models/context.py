from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pareto_designer.models.motif import BindingMotif
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.shared.func_cost.base_function import ScoreFunction


@dataclass
class FSMContext:
    motif: BindingMotif
    binding_score_map: dict[str, float]
    fsm: FSM
    fsm_id: str

    @property
    def motif_id(self) -> str:
        return self.motif.matrix_id

    @property
    def motif_length(self) -> int:
        return self.motif.length

    @property
    def alphabet(self) -> list[str]:
        return self.motif.alphabet

    @property
    def size(self) -> int:
        return len(self.fsm.V)


@dataclass
class RunContext:
    target_sequence_id: str
    target_sequence: str
    orfs: list[tuple[int, int]]
    cost_params: dict[str, float]
    motif_id: str
    fsm_id: str
    fsm_size: int
    solutions_limit: int
    n_solutions: int = 0
    runtime: str = ""

    @property
    def output_path(self) -> Path:
        return (
            Path("designer_results")
            / self.target_sequence_id
            / self.motif_id
            / self.fsm_id
            / f"solutions_limit_{self.solutions_limit}"
        )


@dataclass
class ParetoResult:
    cost: float
    binding_score: float
    id: str
    url: str
    txt_file: str
    fasta_file: str
    sequence: str
    costs: np.ndarray
    n_substitutions: int
    n_motif_hits: int


@dataclass
class DesignContext:
    score_function: ScoreFunction
    fsm_ctx: FSMContext
    run_ctx: RunContext

    @property
    def sequence_length(self) -> int:
        return len(self.score_function.target_sequence)

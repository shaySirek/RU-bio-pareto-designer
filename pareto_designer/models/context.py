from dataclasses import dataclass
from pathlib import Path

from pareto_designer.models.motif import BindingMotif
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.spaces import ScoreSpace
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod
from pareto_designer.shared.seq_design_utils.run_paths import (
    format_cost_params_str,
    run_output_path,
)


@dataclass
class FSMContext:
    motif: BindingMotif
    origin_binding_score_map: dict[str, float]
    binding_score_map: dict[str, float]
    binding_score_space: type[ScoreSpace]
    fsm_binding_score_err: float
    fsm: FSM
    fsm_id: str
    reduce_fsm_by: float
    db_fsm_size: int
    hit_pvalue: float = 2e-3
    reported_size: int | None = None

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
        if self.reported_size is not None:
            return self.reported_size
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
    sampler: SamplingMethod
    n_solutions: int = 0
    runtime: str = ""
    runtime_seconds: float = 0.0
    results_root: Path = Path("designer_results")

    @property
    def cost_params_str(self) -> str:
        return format_cost_params_str(self.cost_params)

    @property
    def output_path(self) -> Path:
        return run_output_path(
            self.results_root,
            self.target_sequence_id,
            self.cost_params_str,
            self.motif_id,
            self.fsm_id,
            self.sampler,
        )


@dataclass
class ParetoResult:
    cost: float
    binding_score: float
    origin_binding_score: float
    id: str
    url: str
    txt_file: str
    fasta_file: str
    positional_objectives_file: str
    max_positional_cost: float
    min_positional_binding: float
    max_positional_binding: float
    sequence: str
    n_cost_items: int
    motif_hits: list[tuple[int, int]]

    @property
    def n_motif_hits(self) -> int:
        return len(self.motif_hits)


@dataclass
class DesignContext:
    score_function: ScoreFunction
    fsm_ctx: FSMContext
    run_ctx: RunContext

    @property
    def target_sequence(self) -> str:
        return self.score_function.target_sequence

    @property
    def sequence_length(self) -> int:
        return len(self.target_sequence)

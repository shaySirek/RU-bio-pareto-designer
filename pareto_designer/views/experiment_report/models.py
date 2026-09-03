from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from openpyxl.utils import get_column_letter

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.pareto_utils import sampler_alpha_label


class ReportGranularity(StrEnum):
    SOLUTION = "solution"
    DESIGN_RUN = "design_run"


@dataclass(frozen=True)
class SamplerParams:
    k: int
    alpha: float
    log_pos: bool

    @property
    def alpha_label(self) -> str:
        return sampler_alpha_label(self.alpha, self.log_pos)


@dataclass(frozen=True)
class RunParams:
    seq_id: str
    fsm_id: str
    fsm_size: int
    reduce_fsm_by: float
    sampler: SamplerParams

    @property
    def run_key(self) -> tuple:
        return (
            self.seq_id,
            self.fsm_id,
            self.sampler.k,
            self.sampler.alpha,
            self.sampler.log_pos,
        )


@dataclass
class LoadedRun:
    params: RunParams
    metadata: dict
    solutions: list[ParetoResult]
    path: Path


@dataclass
class SolutionRecord:
    seq_id: str
    fsm_id: str
    fsm_size: int
    reduce_fsm_by: float
    k: int
    alpha: float
    log_pos: bool
    alpha_label: str
    sweeps: str
    solution_id: str
    cost: float
    binding_score: float
    origin_binding_score: float
    kmer_binding_score_mse: float
    kmer_binding_score_err_std: float
    n_motif_hits: int
    n_nonsyn: int
    n_cost_items: int
    quality_region: str


@dataclass
class DesignRunSummary:
    seq_id: str
    fsm_id: str
    fsm_size: int
    reduce_fsm_by: float
    k: int
    alpha: float
    log_pos: bool
    alpha_label: str
    sweeps: str
    n_solutions: int
    runtime_s: float
    kmer_binding_score_mse_mean: float
    kmer_binding_score_mse_solution_std: float
    db_fsm_size: int
    fsm_binding_score_err: float
    n_with_hits: int
    n_roi: int
    n_plateau: int
    n_with_nonsyn: int
    roi_cost_min: float
    roi_cost_p25: float
    roi_cost_p50: float
    roi_cost_p75: float
    roi_cost_max: float
    roi_cost_mean: float
    roi_cost_std: float
    roi_binding_min: float
    roi_binding_p25: float
    roi_binding_p50: float
    roi_binding_p75: float
    roi_binding_max: float
    roi_binding_mean: float
    roi_binding_std: float


@dataclass(frozen=True)
class ExpectedRun:
    seq_id: str
    sweep: str
    params: RunParams
    metadata_path: Path


@dataclass(frozen=True)
class SweepGrid:
    k_values: list[int]
    sampler_alpha: list[str]
    reduce_fsm_by: list[float]


@dataclass
class RangeRef:
    start_row: int
    end_row: int
    col_map: dict[str, int]
    sheet_name: str = ""

    def col_letter(self, name: str) -> str:
        return get_column_letter(self.col_map[name])

    def cell_range(self, col_name: str) -> str:
        col = self.col_letter(col_name)
        return f"{col}{self.start_row}:{col}{self.end_row}"


@dataclass
class ExperimentConfig:
    name: str
    fixed: dict
    sweeps: dict[str, dict]
    config_path: Path = field(default_factory=lambda: Path("."))

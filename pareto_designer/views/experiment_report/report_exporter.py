from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pareto_designer.models.context import FSMContext
from pareto_designer.views.experiment_report import config as config_module
from pareto_designer.views.experiment_report.config import nonsyn_w
from pareto_designer.views.experiment_report.loader import load_all
from pareto_designer.views.experiment_report.metrics import (
    build_design_run_summaries,
    solution_records,
)
from pareto_designer.views.experiment_report.models import (
    DesignRunSummary,
    ExperimentConfig,
    SolutionRecord,
)
from pareto_designer.views.experiment_report.sweeps import assign_sweep_memberships
from pareto_designer.views.experiment_report.xlsx_exporter import export_experiment_xlsx


class ExperimentReportExporter:
    def __init__(
        self,
        results_root: Path,
        config: ExperimentConfig | None = None,
        fsm_contexts: Sequence[FSMContext] | None = None,
    ):
        self.results_root = Path(results_root)
        self.config = config
        self.fsm_contexts = list(fsm_contexts) if fsm_contexts else []
        self.runs = []
        self.design_runs: list[DesignRunSummary] = []
        self.solutions: list[SolutionRecord] = []

    def load(self) -> None:
        self.runs = load_all(self.results_root, fsm_contexts=self.fsm_contexts)
        assign_sweep_memberships(self.runs, self.config)
        self.design_runs = build_design_run_summaries(self.runs, self.config)
        self.solutions = []
        for run in self.runs:
            sweeps = getattr(run, "_sweeps", [])
            self.solutions.extend(
                solution_records(run, sweeps, w=nonsyn_w(self.config))
            )

    def export(self, output_path: Path | None = None) -> Path:
        if not self.design_runs and not self.runs:
            self.load()

        out = output_path
        if out is None:
            if self.config is not None:
                out = config_module.report_output_path(self.config)
            else:
                out = self.results_root / "pareto_experiment_report.xlsx"

        export_experiment_xlsx(
            out,
            config=self.config,
            checklist=self._build_checklist(),
            design_runs=self.design_runs,
            solutions=self.solutions,
        )
        return out

    def _build_checklist(self) -> list[tuple[str, str, str, bool]]:
        if self.config is not None:
            expected = config_module.expected_runs(
                self.config, fsm_contexts=self.fsm_contexts
            )
            return [
                (
                    item.seq_id,
                    item.sweep,
                    str(item.metadata_path),
                    item.metadata_path.exists(),
                )
                for item in expected
            ]
        return [
            (
                run.params.seq_id,
                ",".join(getattr(run, "_sweeps", [])),
                str(run.path),
                True,
            )
            for run in self.runs
        ]

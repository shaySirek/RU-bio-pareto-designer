from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from pareto_designer.views.experiment_report.excel_tables import (
    write_overview_sheet,
    write_solutions_sheet,
    write_summary_sheet,
    write_sweep_sheets,
)
from pareto_designer.views.experiment_report.models import (
    DesignRunSummary,
    ExperimentConfig,
    SolutionRecord,
)


def export_experiment_xlsx(
    output_path: Path,
    *,
    config: ExperimentConfig | None,
    checklist: list[tuple[str, str, str, bool]],
    design_runs: list[DesignRunSummary],
    solutions: list[SolutionRecord],
) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    write_overview_sheet(wb, config, checklist)
    write_summary_sheet(wb, design_runs)
    write_sweep_sheets(wb, design_runs)
    write_solutions_sheet(wb, solutions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path

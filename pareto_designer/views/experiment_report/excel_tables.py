from __future__ import annotations

from typing import Any, Iterable, Sequence

from openpyxl.styles import Border, Font, Side
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.worksheet.worksheet import Worksheet

from pareto_designer.views.experiment_report.excel_charts import (
    GroupedSeries,
    write_chart_row,
)
from pareto_designer.views.experiment_report.metrics import (
    CORREL_METRIC_LABELS,
    CORREL_ROW_LABEL,
    SWEEP_CORREL_METRICS,
    filter_design_runs_by_alpha_regime,
    filter_design_runs_by_sweep,
    sort_design_runs,
    sort_solutions,
    swept_param_value_from_summary,
)
from pareto_designer.views.experiment_report.models import (
    DesignRunSummary,
    ExperimentConfig,
    RangeRef,
    SolutionRecord,
)

SUMMARY_GAP_COLS = 1
SEQ_BORDER = Border(top=Side(style="medium"))

SWEEP_SHEETS = (
    ("Sweep alpha", "alpha"),
    ("Sweep K", "k"),
    ("Sweep FSM size", "fsm_size"),
)

# (row label, data column for CORREL x-axis)
CorrelSection = tuple[str, str, bool | None]

ALPHA_CORREL_SECTIONS: tuple[CorrelSection, ...] = (
    ("alpha (const)", "alpha", False),
    ("alpha (log_pos)", "alpha", True),
)
K_CORREL_SECTIONS: tuple[CorrelSection, ...] = (("k", "k", None),)
FSM_CORREL_SECTIONS: tuple[CorrelSection, ...] = (
    ("fsm_size", "fsm_size", None),
    ("fsm_err", "fsm_binding_score_err", None),
)


def write_section_title(ws: Worksheet, row: int, title: str, *, col: int = 1) -> int:
    ws.cell(row=row, column=col, value=title).font = Font(bold=True, size=12)
    return row + 2


def _sweep_category_label(run: DesignRunSummary, sweep: str) -> str:
    if sweep == "alpha":
        return run.alpha_label
    if sweep == "k":
        return str(run.k)
    return str(run.fsm_size)


def write_data_block(
    ws: Worksheet,
    row: int,
    col: int,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    seq_border: bool = False,
) -> RangeRef:
    col_map = {h: col + idx for idx, h in enumerate(headers)}
    n_cols = len(headers)
    for idx, header in enumerate(headers):
        ws.cell(row=row, column=col + idx, value=header).font = Font(bold=True)
    data_start = row + 1
    end_row = data_start - 1
    prev_seq_id: str | None = None
    for record in rows:
        end_row += 1
        if seq_border and prev_seq_id is not None and record[0] != prev_seq_id:
            for c in range(col, col + n_cols):
                ws.cell(row=end_row, column=c).border = SEQ_BORDER
        for idx, _header in enumerate(headers):
            ws.cell(row=end_row, column=col + idx, value=record[idx])
        if seq_border:
            prev_seq_id = str(record[0])
    return RangeRef(
        start_row=data_start,
        end_row=end_row,
        col_map=col_map,
        sheet_name=ws.title,
    )


def _correl_range_formula(data: RangeRef, param_col: str, metric_col: str) -> str:
    param = data.col_letter(param_col)
    metric = data.col_letter(metric_col)
    start = data.start_row
    end = data.end_row
    return f"=CORREL(${param}${start}:${param}${end},${metric}${start}:${metric}${end})"


def _correl_if_formula(
    data: RangeRef, param_col: str, metric_col: str, log_pos: bool
) -> str:
    param = data.col_letter(param_col)
    metric = data.col_letter(metric_col)
    log_col = data.col_letter("log_pos")
    start = data.start_row
    end = data.end_row
    flag = "TRUE" if log_pos else "FALSE"
    return (
        f"=CORREL(IF(${log_col}${start}:${log_col}${end}={flag},${param}${start}:${param}${end}),"
        f"IF(${log_col}${start}:${log_col}${end}={flag},${metric}${start}:${metric}${end}))"
    )


def _set_correl_cell(
    ws: Worksheet, row: int, col: int, formula: str, *, array: bool = False
) -> None:
    cell = ws.cell(row=row, column=col)
    if array:
        cell.value = ArrayFormula(ref=cell.coordinate, text=formula)
    else:
        cell.value = formula


def write_correlations_block(
    ws: Worksheet,
    header_row: int,
    col: int,
    data: RangeRef,
    sections: Sequence[CorrelSection],
) -> RangeRef:
    ws.cell(row=header_row - 2, column=col, value="Correlations").font = Font(
        bold=True, size=12
    )
    ws.cell(row=header_row, column=col, value=CORREL_ROW_LABEL).font = Font(bold=True)
    for idx, metric in enumerate(SWEEP_CORREL_METRICS, start=1):
        label = CORREL_METRIC_LABELS[metric]
        ws.cell(row=header_row, column=col + idx, value=label).font = Font(bold=True)

    data_start = header_row + 1
    end_row = data_start + len(sections) - 1

    for section_idx, (label, param_col, log_pos) in enumerate(sections):
        row = data_start + section_idx
        ws.cell(row=row, column=col, value=label)
        for metric_idx, metric in enumerate(SWEEP_CORREL_METRICS, start=1):
            if log_pos is None:
                formula = _correl_range_formula(data, param_col, metric)
                array = False
            else:
                formula = _correl_if_formula(data, param_col, metric, log_pos)
                array = True
            _set_correl_cell(ws, row, col + metric_idx, formula, array=array)

    headers = [
        CORREL_ROW_LABEL,
        *[CORREL_METRIC_LABELS[m] for m in SWEEP_CORREL_METRICS],
    ]
    col_map = {h: col + idx for idx, h in enumerate(headers)}
    return RangeRef(
        start_row=data_start,
        end_row=end_row,
        col_map=col_map,
        sheet_name=ws.title,
    )


def _metric_value(run: DesignRunSummary, metric: str) -> float:
    value = getattr(run, metric)
    if value is None:
        return float("nan")
    return float(value)


def _run_at_sweep_value(
    runs: list[DesignRunSummary], sweep: str, seq_id: str, swept_value: float
) -> DesignRunSummary | None:
    for run in runs:
        if run.seq_id != seq_id:
            continue
        if swept_param_value_from_summary(run, sweep) == swept_value:
            return run
    return None


def _grouped_bar_chart_spec(
    runs: list[DesignRunSummary],
    sweep: str,
    metric: str,
    *,
    log_pos: bool | None = None,
) -> tuple[list[str], list[GroupedSeries]] | None:
    subset = runs
    if log_pos is not None:
        regime = "log_pos" if log_pos else "const"
        subset = filter_design_runs_by_alpha_regime(subset, regime)
    if not subset:
        return None

    swept_values = sorted({swept_param_value_from_summary(r, sweep) for r in subset})
    categories = []
    for v in swept_values:
        sample = next(
            r for r in subset if swept_param_value_from_summary(r, sweep) == v
        )
        categories.append(_sweep_category_label(sample, sweep))

    seq_ids = sorted({r.seq_id for r in subset})
    series: list[GroupedSeries] = []
    for seq_id in seq_ids:
        values = []
        for v in swept_values:
            run = _run_at_sweep_value(subset, sweep, seq_id, v)
            values.append(_metric_value(run, metric) if run else float("nan"))
        series.append((seq_id, values))
    return categories, series


def _chart_specs_for_sweep(
    runs: list[DesignRunSummary],
    sweep: str,
    charts: Sequence[tuple[str, bool | None, str]],
) -> list[tuple[str, list[str], list[GroupedSeries]]]:
    specs: list[tuple[str, list[str], list[GroupedSeries]]] = []
    for title, log_pos, metric in charts:
        grouped = _grouped_bar_chart_spec(runs, sweep, metric, log_pos=log_pos)
        if grouped is None:
            continue
        categories, series = grouped
        specs.append((title, categories, series))
    return specs


ALPHA_CHARTS = (
    ("Alpha (const): hypervolume vs alpha", False, "hypervolume"),
    ("Alpha (log_pos): hypervolume vs alpha", True, "hypervolume"),
)
K_CHARTS = (("K sweep: hypervolume vs k", None, "hypervolume"),)
FSM_CHARTS = (("FSM size: hypervolume vs fsm_size", None, "hypervolume"),)

SWEEP_CORREL_SECTIONS: dict[str, tuple[CorrelSection, ...]] = {
    "alpha": ALPHA_CORREL_SECTIONS,
    "k": K_CORREL_SECTIONS,
    "fsm_size": FSM_CORREL_SECTIONS,
}

SWEEP_CHARTS: dict[str, Sequence[tuple[str, bool | None, str]]] = {
    "alpha": ALPHA_CHARTS,
    "k": K_CHARTS,
    "fsm_size": FSM_CHARTS,
}

DESIGN_RUN_HEADERS = [
    "seq_id",
    "fsm_id",
    "fsm_size",
    "reduce_fsm_by",
    "k",
    "alpha",
    "log_pos",
    "alpha_label",
    "sweeps",
    "n_solutions",
    "runtime_s",
    "binding_score_sse",
    "binding_score_mse",
    "binding_score_rmse",
    "fsm_binding_score_err",
    "hypervolume",
]


def design_run_row(summary: DesignRunSummary) -> list[Any]:
    return [
        summary.seq_id,
        summary.fsm_id,
        summary.fsm_size,
        summary.reduce_fsm_by,
        summary.k,
        summary.alpha,
        summary.log_pos,
        summary.alpha_label,
        summary.sweeps,
        summary.n_solutions,
        summary.runtime_s,
        summary.binding_score_sse,
        summary.binding_score_mse,
        summary.binding_score_rmse,
        summary.fsm_binding_score_err,
        summary.hypervolume,
    ]


SOLUTION_HEADERS = [
    "seq_id",
    "fsm_id",
    "fsm_size",
    "reduce_fsm_by",
    "k",
    "alpha_label",
    "sweeps",
    "solution_id",
    "cost",
    "binding_score",
    "origin_binding_score",
    "n_motif_hits",
    "n_cost_items",
]


def solution_row(record: SolutionRecord) -> list[Any]:
    return [
        record.seq_id,
        record.fsm_id,
        record.fsm_size,
        record.reduce_fsm_by,
        record.k,
        record.alpha_label,
        record.sweeps,
        record.solution_id,
        record.cost,
        record.binding_score,
        record.origin_binding_score,
        record.n_motif_hits,
        record.n_cost_items,
    ]


def freeze_header_row(ws: Worksheet, header_row: int) -> None:
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)


def write_overview_sheet(
    wb,
    config: ExperimentConfig | None,
    checklist: list[tuple[str, str, str, bool]],
) -> None:
    ws = wb.create_sheet("Overview", 0)
    row = write_section_title(ws, 1, "Experiment overview")
    if config:
        ws.cell(row=row, column=1, value="Config name")
        ws.cell(row=row, column=2, value=config.name)
        row += 1
        ws.cell(row=row, column=1, value="Results root")
        ws.cell(row=row, column=2, value=config.fixed.get("results_root", ""))
        row += 2
    row = write_section_title(ws, row, "Expected runs checklist")
    headers = ["seq_id", "sweep", "metadata_path", "exists"]
    write_data_block(ws, row, 1, headers, checklist)


def write_summary_sheet(wb, design_runs: list[DesignRunSummary]) -> None:
    ws = wb.create_sheet("Summary")
    write_data_block(
        ws,
        1,
        1,
        DESIGN_RUN_HEADERS,
        (design_run_row(r) for r in sort_design_runs(design_runs)),
        seq_border=True,
    )


def write_sweep_sheet(
    wb,
    sheet_title: str,
    sweep: str,
    design_runs: list[DesignRunSummary],
) -> None:
    runs = sort_design_runs(filter_design_runs_by_sweep(design_runs, sweep))
    if not runs:
        return
    ws = wb.create_sheet(sheet_title)
    header_row = write_section_title(ws, 1, "Design runs")
    data = write_data_block(
        ws,
        header_row,
        1,
        DESIGN_RUN_HEADERS,
        (design_run_row(r) for r in runs),
        seq_border=True,
    )
    correl_col = len(DESIGN_RUN_HEADERS) + SUMMARY_GAP_COLS + 1
    sections = SWEEP_CORREL_SECTIONS[sweep]
    correl = write_correlations_block(ws, header_row, correl_col, data, sections)
    chart_row = max(data.end_row, correl.end_row) + 3
    charts = SWEEP_CHARTS[sweep]
    write_chart_row(ws, chart_row, _chart_specs_for_sweep(runs, sweep, charts))


def write_sweep_sheets(wb, design_runs: list[DesignRunSummary]) -> None:
    for sheet_title, sweep in SWEEP_SHEETS:
        write_sweep_sheet(wb, sheet_title, sweep, design_runs)


def write_solutions_sheet(wb, solutions: list[SolutionRecord]) -> None:
    ws = wb.create_sheet("Solutions")
    write_data_block(
        ws,
        1,
        1,
        SOLUTION_HEADERS,
        (solution_row(r) for r in sort_solutions(solutions)),
        seq_border=True,
    )
    freeze_header_row(ws, 1)

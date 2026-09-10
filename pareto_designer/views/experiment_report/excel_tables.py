from __future__ import annotations

from typing import Any, Iterable, Sequence

from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.worksheet.worksheet import Worksheet

from pareto_designer.views.experiment_report.metrics import (
    filter_design_runs_by_sweep,
    sort_design_runs,
    sort_solutions,
)
from pareto_designer.views.experiment_report.excel_schema import (
    OVERVIEW_CHECKLIST_TABLE,
    SOLUTION_TABLE,
    ExcelTableSpec,
    GroupedExcelTableSpec,
    design_run_table,
)
from pareto_designer.views.experiment_report.models import (
    DesignRunSummary,
    ExperimentConfig,
    RangeRef,
    SolutionRecord,
)

SEQ_BORDER = Border(top=Side(style="medium"))
_HEADER_FONT = Font(bold=True)
_HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)

SWEEP_SHEETS = (
    ("Sweep alpha", "alpha"),
    ("Sweep K", "k"),
    ("Sweep FSM size", "fsm_size"),
)


def write_data_block(
    ws: Worksheet,
    row: int,
    col: int,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    keys: Sequence[str] | None = None,
    seq_border: bool = False,
    write_headers: bool = True,
) -> RangeRef:
    col_keys = keys if keys is not None else headers
    col_map = {key: col + idx for idx, key in enumerate(col_keys)}
    n_cols = len(headers)
    if write_headers:
        for idx, header in enumerate(headers):
            cell = ws.cell(row=row, column=col + idx, value=header)
            cell.font = _HEADER_FONT
        data_start = row + 1
    else:
        data_start = row
    end_row = data_start - 1
    prev_seq_id: str | None = None
    for record in rows:
        end_row += 1
        if seq_border and prev_seq_id is not None and record[0] != prev_seq_id:
            for c in range(col, col + n_cols):
                ws.cell(row=end_row, column=c).border = SEQ_BORDER
        for idx, _header in enumerate(headers):
            ws.cell(row=end_row, column=col + idx, value=_excel_cell_value(record[idx]))
        if seq_border:
            prev_seq_id = str(record[0])
    return RangeRef(
        start_row=data_start,
        end_row=end_row,
        col_map=col_map,
        sheet_name=ws.title,
    )


def _excel_cell_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if value == value else None
    return value


def _merge_and_label(
    ws: Worksheet,
    title: str,
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
) -> None:
    if end_row > start_row or end_col > start_col:
        ws.merge_cells(
            start_row=start_row,
            start_column=start_col,
            end_row=end_row,
            end_column=end_col,
        )
    cell = ws.cell(row=start_row, column=start_col, value=title)
    cell.font = _HEADER_FONT
    cell.alignment = _HEADER_ALIGN


def _write_nested_headers(
    ws: Worksheet,
    row: int,
    col: int,
    spec: GroupedExcelTableSpec,
) -> int:
    category_row = row
    subcategory_row = row + 1
    leaf_row = row + 2
    col_idx = col
    for group in spec.groups:
        leaves = group.leaf_columns()
        n_leaves = len(leaves)
        end_col = col_idx + n_leaves - 1
        if group.subgroups:
            _merge_and_label(
                ws, group.title, category_row, col_idx, category_row, end_col
            )
            sub_col = col_idx
            for sub in group.subgroups:
                sub_n = len(sub.leaf_columns())
                _merge_and_label(
                    ws,
                    sub.title,
                    subcategory_row,
                    sub_col,
                    subcategory_row,
                    sub_col + sub_n - 1,
                )
                sub_col += sub_n
        else:
            _merge_and_label(
                ws, group.title, category_row, col_idx, subcategory_row, end_col
            )
        col_idx += n_leaves

    for idx, column in enumerate(spec.columns):
        cell = ws.cell(row=leaf_row, column=col + idx, value=column.label)
        cell.font = _HEADER_FONT
        cell.alignment = _HEADER_ALIGN
    return leaf_row


def write_table(
    ws: Worksheet,
    row: int,
    col: int,
    spec: ExcelTableSpec | GroupedExcelTableSpec,
    records: Iterable[Any],
    *,
    seq_border: bool = False,
    freeze: bool = False,
) -> RangeRef:
    if isinstance(spec, GroupedExcelTableSpec):
        leaf_row = _write_nested_headers(ws, row, col, spec)
        ref = write_data_block(
            ws,
            leaf_row + 1,
            col,
            spec.headers,
            spec.rows(records),
            keys=spec.keys,
            seq_border=seq_border,
            write_headers=False,
        )
        if freeze:
            freeze_header_row(ws, leaf_row)
        return ref

    ref = write_data_block(
        ws,
        row,
        col,
        spec.headers,
        spec.rows(records),
        keys=spec.keys,
        seq_border=seq_border,
    )
    if freeze:
        freeze_header_row(ws, row)
    return ref


DESIGN_RUN_HEADERS = design_run_table().headers


def design_run_row(
    summary: DesignRunSummary, *, dominance_sweep: str | None = None
) -> list[Any]:
    return list(design_run_table(dominance_sweep).row(summary))


SOLUTION_HEADERS = SOLUTION_TABLE.headers


def solution_row(record: SolutionRecord) -> list[Any]:
    return list(SOLUTION_TABLE.row(record))


def freeze_header_row(ws: Worksheet, header_row: int) -> None:
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)


def write_overview_sheet(
    wb,
    config: ExperimentConfig | None,
    checklist: list[tuple[str, str, str, bool]],
) -> None:
    ws = wb.create_sheet("Overview", 0)
    write_table(
        ws,
        1,
        1,
        OVERVIEW_CHECKLIST_TABLE,
        (
            {"seq_id": s, "sweep": sw, "metadata_path": p, "exists": e}
            for s, sw, p, e in checklist
        ),
        freeze=True,
    )


def write_summary_sheet(wb, design_runs: list[DesignRunSummary]) -> None:
    ws = wb.create_sheet("Summary")
    write_table(
        ws,
        1,
        1,
        design_run_table(),
        sort_design_runs(design_runs),
        seq_border=True,
        freeze=True,
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
    write_table(
        ws,
        1,
        1,
        design_run_table(sweep),
        runs,
        seq_border=True,
        freeze=True,
    )


def write_sweep_sheets(wb, design_runs: list[DesignRunSummary]) -> None:
    for sheet_title, sweep in SWEEP_SHEETS:
        write_sweep_sheet(wb, sheet_title, sweep, design_runs)


def write_solutions_sheet(wb, solutions: list[SolutionRecord]) -> None:
    ws = wb.create_sheet("Solutions")
    write_table(
        ws,
        1,
        1,
        SOLUTION_TABLE,
        sort_solutions(solutions),
        seq_border=True,
        freeze=True,
    )

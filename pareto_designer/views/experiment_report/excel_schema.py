from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExcelColumn:
    key: str
    header: str | None = None

    @property
    def label(self) -> str:
        return self.header if self.header is not None else self.key


@dataclass(frozen=True)
class ExcelColumnGroup:
    title: str
    columns: tuple[ExcelColumn, ...]


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    return getattr(record, key)


@dataclass(frozen=True)
class ExcelTableSpec:
    columns: tuple[ExcelColumn, ...]

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(col.label for col in self.columns)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(col.key for col in self.columns)

    def row(self, record: Any) -> tuple[Any, ...]:
        return tuple(_record_value(record, col.key) for col in self.columns)

    def rows(self, records: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
        for record in records:
            yield self.row(record)

    def col_map(self, start_col: int) -> dict[str, int]:
        return {col.key: start_col + idx for idx, col in enumerate(self.columns)}


@dataclass(frozen=True)
class GroupedExcelTableSpec:
    leading: tuple[ExcelColumn, ...]
    groups: tuple[ExcelColumnGroup, ...]

    @property
    def columns(self) -> tuple[ExcelColumn, ...]:
        return self.leading + tuple(
            col for group in self.groups for col in group.columns
        )

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(col.label for col in self.columns)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(col.key for col in self.columns)

    def row(self, record: Any) -> tuple[Any, ...]:
        return tuple(_record_value(record, col.key) for col in self.columns)

    def rows(self, records: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
        for record in records:
            yield self.row(record)

    def col_map(self, start_col: int) -> dict[str, int]:
        return {col.key: start_col + idx for idx, col in enumerate(self.columns)}


ROI_COST_COLUMNS = (
    ExcelColumn("roi_cost_min"),
    ExcelColumn("roi_cost_p25"),
    ExcelColumn("roi_cost_p50"),
    ExcelColumn("roi_cost_p75"),
    ExcelColumn("roi_cost_max"),
    ExcelColumn("roi_cost_mean"),
    ExcelColumn("roi_cost_std"),
)

ROI_BINDING_COLUMNS = (
    ExcelColumn("roi_binding_min"),
    ExcelColumn("roi_binding_p25"),
    ExcelColumn("roi_binding_p50"),
    ExcelColumn("roi_binding_p75"),
    ExcelColumn("roi_binding_max"),
    ExcelColumn("roi_binding_mean"),
    ExcelColumn("roi_binding_std"),
)

DESIGN_RUN_TABLE = GroupedExcelTableSpec(
    leading=(
        ExcelColumn("seq_id"),
        ExcelColumn("fsm_id"),
        ExcelColumn("fsm_size"),
        ExcelColumn("db_fsm_size"),
        ExcelColumn("reduce_fsm_by"),
        ExcelColumn("k"),
        ExcelColumn("alpha"),
        ExcelColumn("log_pos"),
        ExcelColumn("alpha_label"),
        ExcelColumn("sweeps"),
        ExcelColumn("n_solutions"),
        ExcelColumn("runtime_s"),
    ),
    groups=(
        ExcelColumnGroup(
            "FSM reduction",
            (
                ExcelColumn("fsm_binding_score_err"),
                ExcelColumn("kmer_binding_score_mse_mean"),
                ExcelColumn("kmer_binding_score_mse_solution_std"),
            ),
        ),
        ExcelColumnGroup(
            "Solution quality",
            (
                ExcelColumn("n_with_hits"),
                ExcelColumn("n_roi"),
                ExcelColumn("n_plateau"),
                ExcelColumn("n_with_nonsyn"),
            ),
        ),
        ExcelColumnGroup(
            "Region of interest",
            ROI_COST_COLUMNS + ROI_BINDING_COLUMNS,
        ),
    ),
)

SOLUTION_TABLE = ExcelTableSpec(
    columns=(
        ExcelColumn("seq_id"),
        ExcelColumn("fsm_id"),
        ExcelColumn("fsm_size"),
        ExcelColumn("reduce_fsm_by"),
        ExcelColumn("k"),
        ExcelColumn("alpha_label"),
        ExcelColumn("sweeps"),
        ExcelColumn("solution_id"),
        ExcelColumn("cost"),
        ExcelColumn("binding_score"),
        ExcelColumn("origin_binding_score"),
        ExcelColumn("kmer_binding_score_mse"),
        ExcelColumn("kmer_binding_score_err_std"),
        ExcelColumn("n_motif_hits"),
        ExcelColumn("n_nonsyn"),
        ExcelColumn("n_cost_items"),
        ExcelColumn("quality_region"),
    )
)

OVERVIEW_CHECKLIST_TABLE = ExcelTableSpec(
    columns=(
        ExcelColumn("seq_id"),
        ExcelColumn("sweep"),
        ExcelColumn("metadata_path"),
        ExcelColumn("exists"),
    )
)

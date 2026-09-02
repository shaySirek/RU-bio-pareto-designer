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
class ExcelTableSpec:
    columns: tuple[ExcelColumn, ...]

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(col.label for col in self.columns)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(col.key for col in self.columns)

    def row(self, record: Any) -> tuple[Any, ...]:
        return tuple(self._value(record, col.key) for col in self.columns)

    @staticmethod
    def _value(record: Any, key: str) -> Any:
        if isinstance(record, dict):
            return record[key]
        return getattr(record, key)

    def rows(self, records: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
        for record in records:
            yield self.row(record)

    def col_map(self, start_col: int) -> dict[str, int]:
        return {col.key: start_col + idx for idx, col in enumerate(self.columns)}


DESIGN_RUN_TABLE = ExcelTableSpec(
    columns=(
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
        ExcelColumn("kmer_binding_score_mse_mean"),
        ExcelColumn("kmer_binding_score_mse_solution_std"),
        ExcelColumn("fsm_binding_score_err"),
    )
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
        ExcelColumn("n_cost_items"),
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

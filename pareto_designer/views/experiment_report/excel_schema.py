from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pareto_designer.shared.seq_design_utils.solution_quality import SolutionRegion


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
    columns: tuple[ExcelColumn, ...] = ()
    subgroups: tuple["ExcelColumnGroup", ...] = ()

    def leaf_columns(self) -> tuple[ExcelColumn, ...]:
        if self.subgroups:
            return tuple(
                col for group in self.subgroups for col in group.leaf_columns()
            )
        return self.columns

    def walk_depth(self) -> int:
        if not self.subgroups:
            return 1
        return 1 + max(sub.walk_depth() for sub in self.subgroups)


def _record_value(record: Any, key: str) -> Any:
    current = record
    for part in key.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


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
    groups: tuple[ExcelColumnGroup, ...]

    @property
    def columns(self) -> tuple[ExcelColumn, ...]:
        return tuple(col for group in self.groups for col in group.leaf_columns())

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(col.label for col in self.columns)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(col.key for col in self.columns)

    @property
    def header_depth(self) -> int:
        return 1 + max((group.walk_depth() for group in self.groups), default=0)

    def row(self, record: Any) -> tuple[Any, ...]:
        return tuple(_record_value(record, col.key) for col in self.columns)

    def rows(self, records: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
        for record in records:
            yield self.row(record)

    def col_map(self, start_col: int) -> dict[str, int]:
        return {col.key: start_col + idx for idx, col in enumerate(self.columns)}


_ROI_STATS = ("min", "p25", "p50", "p75", "max", "mean", "std")

_QUALITY_COLUMNS = (
    (SolutionRegion.HITS, "n_with_hits", "with hits"),
    (SolutionRegion.ROI, "n_roi", "roi"),
    (SolutionRegion.PLATEAU, "n_plateau", "plateau"),
    (SolutionRegion.NONSYN, "n_with_nonsyn", "with nonsyn"),
)

_SWEEP_DOMINANCE_ATTR = {
    "alpha": "dominance_alpha",
    "k": "dominance_k",
    "fsm_size": "dominance_fsm",
}


def _roi_subgroup(prefix: str, title: str) -> ExcelColumnGroup:
    return ExcelColumnGroup(
        title,
        columns=tuple(ExcelColumn(f"roi_{prefix}_{stat}", stat) for stat in _ROI_STATS),
    )


def _quality_subgroups(
    dominance_sweep: str | None,
) -> tuple[ExcelColumnGroup, ...]:
    attr = _SWEEP_DOMINANCE_ATTR.get(dominance_sweep or "", "")
    groups: list[ExcelColumnGroup] = []
    for region, n_key, title in _QUALITY_COLUMNS:
        columns = [ExcelColumn(n_key, "n")]
        if attr:
            columns.append(
                ExcelColumn(f"{attr}.n_by_region.{region.value}", "n_dom_by_next")
            )
        groups.append(ExcelColumnGroup(title, columns=tuple(columns)))
    return tuple(groups)


def design_run_table(dominance_sweep: str | None = None) -> GroupedExcelTableSpec:
    return GroupedExcelTableSpec(
        groups=(
            ExcelColumnGroup("Sequence", columns=(ExcelColumn("seq_id"),)),
            ExcelColumnGroup(
                "Sampling",
                subgroups=(
                    ExcelColumnGroup(
                        "low-cost preference",
                        columns=(ExcelColumn("log_pos"), ExcelColumn("alpha")),
                    ),
                    ExcelColumnGroup("k", columns=(ExcelColumn("k"),)),
                ),
            ),
            ExcelColumnGroup(
                "FSM",
                subgroups=(
                    ExcelColumnGroup(
                        "model",
                        columns=(
                            ExcelColumn("fsm_id"),
                            ExcelColumn("fsm_size"),
                            ExcelColumn("reduce_fsm_by"),
                            ExcelColumn("db_fsm_size"),
                        ),
                    ),
                    ExcelColumnGroup(
                        "error",
                        columns=(
                            ExcelColumn("fsm_binding_score_err", "fsm_err"),
                            ExcelColumn("kmer_binding_score_mse_mean", "kmer_mse"),
                            ExcelColumn(
                                "kmer_binding_score_mse_solution_std", "kmer_mse_std"
                            ),
                        ),
                    ),
                ),
            ),
            ExcelColumnGroup(
                "Output",
                columns=(ExcelColumn("n_solutions"), ExcelColumn("runtime_s")),
            ),
            ExcelColumnGroup(
                "Solution quality",
                subgroups=_quality_subgroups(dominance_sweep),
            ),
            ExcelColumnGroup(
                "Objectives within ROI",
                subgroups=(
                    _roi_subgroup("cost", "cost"),
                    _roi_subgroup("binding", "binding"),
                ),
            ),
        )
    )


DESIGN_RUN_TABLE = design_run_table()

SOLUTION_TABLE = ExcelTableSpec(
    columns=(
        ExcelColumn("seq_id"),
        ExcelColumn("log_pos"),
        ExcelColumn("alpha"),
        ExcelColumn("k"),
        ExcelColumn("fsm_id"),
        ExcelColumn("fsm_size"),
        ExcelColumn("reduce_fsm_by"),
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

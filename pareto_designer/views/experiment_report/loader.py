from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pareto_designer.models.context import FSMContext, ParetoResult
from pareto_designer.views.experiment_report.kmer_binding import fill_kmer_binding
from pareto_designer.views.experiment_report.models import LoadedRun
from pareto_designer.views.experiment_report.paths import (
    parse_metadata_path,
    resolve_path,
)


def discover_runs(results_root: Path) -> list[Path]:
    root = resolve_path(results_root)
    if not root.exists():
        return []
    return sorted(root.rglob("results_metadata.json"))


def load_run(metadata_path: Path) -> LoadedRun:
    path = Path(metadata_path)
    params = parse_metadata_path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    solutions = []
    for row in data.get("results", []):
        row.pop("kmer_binding_score_mse", None)
        row.pop("kmer_binding_score_err_std", None)
        solutions.append(ParetoResult(**row))
    return LoadedRun(
        params=params,
        metadata=metadata,
        solutions=solutions,
        path=path,
    )


def load_all(
    results_root: Path,
    fsm_contexts: Sequence[FSMContext] | None = None,
) -> list[LoadedRun]:
    runs = [load_run(p) for p in discover_runs(results_root)]
    fill_kmer_binding(runs, fsm_contexts=fsm_contexts)
    return runs

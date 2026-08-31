from __future__ import annotations

import json
from pathlib import Path

from pareto_designer.models.context import ParetoResult
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
    solutions = [ParetoResult(**row) for row in data.get("results", [])]
    return LoadedRun(
        params=params,
        metadata=metadata,
        solutions=solutions,
        path=path,
    )


def load_all(results_root: Path) -> list[LoadedRun]:
    return [load_run(p) for p in discover_runs(results_root)]

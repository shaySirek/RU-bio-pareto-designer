from __future__ import annotations

from typing import Literal

from pareto_designer.views.experiment_report.config import effective_grid
from pareto_designer.views.experiment_report.models import (
    ExperimentConfig,
    LoadedRun,
    RunParams,
)

SWEEP_NAMES = ("alpha", "k", "fsm_size")


def alpha_regime(params: RunParams) -> Literal["const", "log_pos"]:
    return "log_pos" if params.sampler.log_pos else "const"


def _matches_grid(params: RunParams, grid) -> bool:
    alpha_label = params.sampler.alpha_label
    return (
        params.sampler.k in grid.k_values
        and alpha_label in grid.sampler_alpha
        and any(abs(params.reduce_fsm_by - r) < 1e-9 for r in grid.reduce_fsm_by)
    )


def sweep_membership(params: RunParams, config: ExperimentConfig | None) -> list[str]:
    if config is not None:
        return [
            name
            for name in SWEEP_NAMES
            if _matches_grid(params, effective_grid(config, name))
        ]
    return _infer_sweep_membership(params)


def _infer_sweep_membership(params: RunParams) -> list[str]:
    sweeps: list[str] = []
    if params.sampler.k == 100 and abs(params.reduce_fsm_by - 0.875) < 1e-9:
        sweeps.append("alpha")
    if (
        params.sampler.alpha_label == "1.0_log_pos"
        and abs(params.reduce_fsm_by - 0.875) < 1e-9
    ):
        sweeps.append("k")
    if params.sampler.k == 100 and params.sampler.alpha_label == "1.0_log_pos":
        sweeps.append("fsm_size")
    return sweeps


def matches_sweep(run: LoadedRun, sweep: str) -> bool:
    return sweep in getattr(run, "_sweeps", [])


def sweep_membership_from_run(run: LoadedRun) -> list[str]:
    return getattr(run, "_sweeps", [])


def set_sweep_membership(run: LoadedRun, sweeps: list[str]) -> None:
    run._sweeps = sweeps  # type: ignore[attr-defined]


def swept_param_value(params: RunParams, sweep: str) -> float:
    if sweep == "alpha":
        return params.sampler.alpha
    if sweep == "k":
        return float(params.sampler.k)
    return float(params.fsm_size)


def swept_param_label(params: RunParams, sweep: str) -> str:
    if sweep == "alpha":
        return params.sampler.alpha_label
    if sweep == "k":
        return str(params.sampler.k)
    if abs(params.reduce_fsm_by) < 1e-9:
        return "DB"
    fold = round(1 / (1 - params.reduce_fsm_by))
    return f"{fold}-fold ({params.fsm_size})"


def alpha_groups(runs: list[LoadedRun]) -> dict[str, list[LoadedRun]]:
    groups: dict[str, list[LoadedRun]] = {"const": [], "log_pos": []}
    for run in runs:
        groups[alpha_regime(run.params)].append(run)
    return groups


def assign_sweep_memberships(
    runs: list[LoadedRun], config: ExperimentConfig | None
) -> None:
    for run in runs:
        set_sweep_membership(run, sweep_membership(run.params, config))

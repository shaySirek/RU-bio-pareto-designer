from __future__ import annotations

from collections import defaultdict

import numpy as np

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.solution_quality import (
    SolutionRegion,
    classify_run_solutions,
)
from pareto_designer.views.experiment_report.models import (
    DesignRunSummary,
    LoadedRun,
    SweepDominance,
)

_SWEEP_ATTR = {
    "alpha": "dominance_alpha",
    "k": "dominance_k",
    "fsm_size": "dominance_fsm",
}


def n_dominated_by(front_a: list[ParetoResult], front_b: list[ParetoResult]) -> int:
    if not front_a or not front_b:
        return 0
    a_cost = np.asarray([p.cost for p in front_a], dtype=float)
    a_bind = np.asarray([p.binding_score for p in front_a], dtype=float)
    b_cost = np.asarray([q.cost for q in front_b], dtype=float)
    b_bind = np.asarray([q.binding_score for q in front_b], dtype=float)
    le_cost = b_cost[:, None] <= a_cost[None, :]
    le_bind = b_bind[:, None] <= a_bind[None, :]
    lt_cost = b_cost[:, None] < a_cost[None, :]
    lt_bind = b_bind[:, None] < a_bind[None, :]
    dominated = np.any(le_cost & le_bind & (lt_cost | lt_bind), axis=0)
    return int(np.count_nonzero(dominated))


def sweep_dominance(
    run_a: LoadedRun,
    run_b: LoadedRun | None,
    *,
    w: float | None = None,
) -> SweepDominance:
    if run_b is None:
        return SweepDominance()
    _, regions = classify_run_solutions(run_a.solutions, w=w)
    n_by_region: dict[str, int | None] = {}
    for region in SolutionRegion:
        subset = [sol for sol in run_a.solutions if regions[sol.id] == region]
        if not subset:
            n_by_region[region.value] = 0
        else:
            n_by_region[region.value] = n_dominated_by(subset, run_b.solutions)
    return SweepDominance(n_by_region=n_by_region)


def _runs_in_sweep(runs: list[LoadedRun], sweep: str) -> list[LoadedRun]:
    return [run for run in runs if sweep in getattr(run, "_sweeps", [])]


def _fill_chain(
    groups: dict[tuple, list[LoadedRun]],
    summaries: dict[tuple, DesignRunSummary],
    attr: str,
    *,
    sort_key,
    w: float | None,
) -> None:
    for group_runs in groups.values():
        ordered = sorted(group_runs, key=sort_key)
        for idx, run in enumerate(ordered):
            summary = summaries.get(run.params.run_key)
            if summary is None:
                continue
            nxt = ordered[idx + 1] if idx + 1 < len(ordered) else None
            setattr(summary, attr, sweep_dominance(run, nxt, w=w))


def attach_next_run_dominance(
    runs: list[LoadedRun],
    summaries: list[DesignRunSummary],
    *,
    w: float | None = None,
) -> None:
    summary_map = {(s.seq_id, s.fsm_id, s.k, s.alpha, s.log_pos): s for s in summaries}
    for summary in summaries:
        summary.dominance_alpha = SweepDominance()
        summary.dominance_k = SweepDominance()
        summary.dominance_fsm = SweepDominance()

    alpha_groups: dict[tuple, list[LoadedRun]] = defaultdict(list)
    for run in _runs_in_sweep(runs, "alpha"):
        if run.params.sampler.log_pos:
            continue
        p = run.params
        alpha_groups[(p.seq_id, p.sampler.k, p.fsm_size)].append(run)
    _fill_chain(
        alpha_groups,
        summary_map,
        _SWEEP_ATTR["alpha"],
        sort_key=lambda r: r.params.sampler.alpha,
        w=w,
    )

    k_groups: dict[tuple, list[LoadedRun]] = defaultdict(list)
    for run in _runs_in_sweep(runs, "k"):
        p = run.params
        s = p.sampler
        k_groups[(p.seq_id, s.alpha, s.log_pos, p.fsm_size)].append(run)
    _fill_chain(
        k_groups,
        summary_map,
        _SWEEP_ATTR["k"],
        sort_key=lambda r: r.params.sampler.k,
        w=w,
    )

    fsm_groups: dict[tuple, list[LoadedRun]] = defaultdict(list)
    for run in _runs_in_sweep(runs, "fsm_size"):
        p = run.params
        s = p.sampler
        fsm_groups[(p.seq_id, s.k, s.alpha, s.log_pos)].append(run)
    _fill_chain(
        fsm_groups,
        summary_map,
        _SWEEP_ATTR["fsm_size"],
        sort_key=lambda r: r.params.fsm_size,
        w=w,
    )

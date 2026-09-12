from __future__ import annotations

from collections import defaultdict

import numpy as np

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.solution_quality import (
    RoiDistribution,
    SolutionRegion,
    classify_run_solutions,
    distribution_stats,
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


def _tightest_partner_gains(
    front_a: list[ParetoResult], front_b: list[ParetoResult]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(front_a)
    if n == 0 or not front_b:
        return (
            np.zeros(n, dtype=bool),
            np.full(n, np.nan),
            np.full(n, np.nan),
        )
    a_cost = np.asarray([p.cost for p in front_a], dtype=float)
    a_bind = np.asarray([p.binding_score for p in front_a], dtype=float)
    b_cost = np.asarray([q.cost for q in front_b], dtype=float)
    b_bind = np.asarray([q.binding_score for q in front_b], dtype=float)
    dc = a_cost[None, :] - b_cost[:, None]
    db = a_bind[None, :] - b_bind[:, None]
    dominates = (dc >= 0) & (db >= 0) & ((dc > 0) | (db > 0))
    dist2 = np.where(dominates, dc * dc + db * db, np.inf)
    idx = np.arange(n)
    best_b = np.argmin(dist2, axis=0)
    dominated = np.isfinite(dist2[best_b, idx])
    cost_gain = np.full(n, np.nan)
    bind_gain = np.full(n, np.nan)
    if np.any(dominated):
        cost_gain[dominated] = dc[best_b[dominated], idx[dominated]]
        bind_gain[dominated] = db[best_b[dominated], idx[dominated]]
    return dominated, cost_gain, bind_gain


def n_dominated_by(front_a: list[ParetoResult], front_b: list[ParetoResult]) -> int:
    dominated, _, _ = _tightest_partner_gains(front_a, front_b)
    return int(np.count_nonzero(dominated))


def _stats_for(values: np.ndarray) -> RoiDistribution:
    return distribution_stats([float(v) for v in values])


def sweep_dominance(
    run_a: LoadedRun,
    run_b: LoadedRun | None,
    *,
    w: float | None = None,
) -> SweepDominance:
    if run_b is None:
        return SweepDominance()
    _, regions = classify_run_solutions(run_a.solutions, w=w)
    dominated, cost_gain, bind_gain = _tightest_partner_gains(
        run_a.solutions, run_b.solutions
    )
    region_ids = np.asarray(
        [regions[sol.id].value for sol in run_a.solutions], dtype=object
    )
    n_by_region: dict[str, int | None] = {}
    func_cost_gain_by_region: dict[str, RoiDistribution] = {}
    binding_gain_by_region: dict[str, RoiDistribution] = {}
    for region in SolutionRegion:
        in_region = region_ids == region.value
        dom = dominated & in_region
        n_by_region[region.value] = int(np.count_nonzero(dom))
        func_cost_gain_by_region[region.value] = _stats_for(cost_gain[dom])
        binding_gain_by_region[region.value] = _stats_for(bind_gain[dom])
    return SweepDominance(
        n_by_region=n_by_region,
        n_global=int(np.count_nonzero(dominated)),
        func_cost_gain_by_region=func_cost_gain_by_region,
        binding_gain_by_region=binding_gain_by_region,
        func_cost_gain=_stats_for(cost_gain[dominated]),
        binding_gain=_stats_for(bind_gain[dominated]),
    )


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

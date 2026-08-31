from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from pareto_designer.shared.seq_design_utils.binding_metrics import binding_score_sse
from pareto_designer.shared.seq_design_utils.pareto_utils import compare_frontiers
from pareto_designer.views.experiment_report.models import (
    CrossSequenceAggregate,
    DesignRunSummary,
    ExperimentConfig,
    LoadedRun,
    SolutionRecord,
)
from pareto_designer.views.experiment_report.sweeps import (
    assign_sweep_memberships,
    sweep_membership,
)


def solution_records(
    run: LoadedRun, sweeps: list[str] | None = None
) -> list[SolutionRecord]:
    sweeps_str = ",".join(sweeps or [])
    p = run.params
    s = p.sampler
    return [
        SolutionRecord(
            seq_id=p.seq_id,
            fsm_id=p.fsm_id,
            fsm_size=p.fsm_size,
            reduce_fsm_by=p.reduce_fsm_by,
            k=s.k,
            alpha=s.alpha,
            log_pos=s.log_pos,
            alpha_label=s.alpha_label,
            sweeps=sweeps_str,
            solution_id=r.id,
            cost=r.cost,
            binding_score=r.binding_score,
            origin_binding_score=r.origin_binding_score,
            n_motif_hits=r.n_motif_hits,
            n_cost_items=r.n_cost_items,
        )
        for r in run.solutions
    ]


def parse_runtime_seconds(metadata: dict) -> float:
    if "runtime_seconds" in metadata:
        return float(metadata["runtime_seconds"])
    runtime = metadata.get("runtime", "")
    if not runtime or runtime == "-":
        return float("nan")
    text = str(runtime)
    if text.endswith("s") and ":" not in text:
        try:
            return float(text[:-1])
        except ValueError:
            return float("nan")
    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (int(p) for p in parts)
        return float(hours * 3600 + minutes * 60 + seconds)
    return float("nan")


def design_run_summary(
    run: LoadedRun, hv: float | None, sweeps: list[str] | None = None
) -> DesignRunSummary:
    p = run.params
    s = p.sampler
    sse = binding_score_sse(run.solutions)
    n = len(run.solutions)
    binding_score_mse = sse / n if n else float("nan")
    binding_score_rmse = (
        math.sqrt(binding_score_mse) if n and binding_score_mse >= 0 else float("nan")
    )
    meta = run.metadata
    return DesignRunSummary(
        seq_id=p.seq_id,
        fsm_id=p.fsm_id,
        fsm_size=p.fsm_size,
        reduce_fsm_by=p.reduce_fsm_by,
        k=s.k,
        alpha=s.alpha,
        log_pos=s.log_pos,
        alpha_label=s.alpha_label,
        sweeps=",".join(sweeps or []),
        n_solutions=meta.get("n_solutions", n),
        runtime_s=parse_runtime_seconds(meta),
        binding_score_sse=sse,
        binding_score_mse=binding_score_mse,
        binding_score_rmse=binding_score_rmse,
        fsm_binding_score_err=float(meta.get("fsm_binding_score_err", float("nan"))),
        hypervolume=hv,
    )


def _frontier_key(run: LoadedRun) -> str:
    p = run.params
    return f"{p.fsm_id}__k_{p.sampler.k}__{p.sampler.alpha_label}"


def compute_hypervolumes(runs: list[LoadedRun]) -> dict[tuple, float]:
    grouped: dict[str, list[LoadedRun]] = defaultdict(list)
    for run in runs:
        grouped[run.params.seq_id].append(run)

    hv_map: dict[tuple, float] = {}
    for group in grouped.values():
        frontiers: dict[str, np.ndarray] = {}
        for run in group:
            if not run.solutions:
                continue
            arr = np.array(
                [[s.cost, s.binding_score] for s in run.solutions], dtype=float
            )
            frontiers[_frontier_key(run)] = arr
        if len(frontiers) < 2:
            for run in group:
                hv_map[run.params.run_key] = float("nan")
            continue
        hvs = compare_frontiers(frontiers, output_file=None)
        for run in group:
            hv_map[run.params.run_key] = hvs.get(_frontier_key(run), float("nan"))
    return hv_map


def build_design_run_summaries(
    runs: list[LoadedRun], config: ExperimentConfig | None = None
) -> list[DesignRunSummary]:
    assign_sweep_memberships(runs, config)
    summaries_by_key: dict[tuple, DesignRunSummary] = {}
    hv_by_sweep: dict[str, dict[tuple, float]] = {}

    for sweep in ("alpha", "k", "fsm_size"):
        sweep_runs = [r for r in runs if sweep in getattr(r, "_sweeps", [])]
        hv_by_sweep[sweep] = compute_hypervolumes(sweep_runs)

    for run in runs:
        sweeps = getattr(run, "_sweeps", sweep_membership(run.params, config))
        hv = float("nan")
        for sweep in sweeps:
            sweep_hv = hv_by_sweep.get(sweep, {}).get(run.params.run_key)
            if sweep_hv is not None and not math.isnan(sweep_hv):
                hv = sweep_hv
                break
        key = run.params.run_key
        if key not in summaries_by_key:
            summaries_by_key[key] = design_run_summary(run, hv, sweeps)
        else:
            existing = summaries_by_key[key]
            merged_sweeps = sorted(set(existing.sweeps.split(",")) | set(sweeps))
            summaries_by_key[key] = design_run_summary(
                run, hv if not math.isnan(hv) else existing.hypervolume, merged_sweeps
            )
    return list(summaries_by_key.values())


def _aggregate_bucket(
    rows: list[DesignRunSummary], sweep: str
) -> CrossSequenceAggregate | None:
    if not rows:
        return None
    ref = rows[0]

    def stats(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=float)
        if len(arr) == 0:
            return float("nan"), float("nan")
        if len(arr) == 1:
            return float(arr[0]), float("nan")
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    hv_mean, hv_std = stats([r.hypervolume for r in rows if r.hypervolume is not None])
    sse_mean, sse_std = stats([r.binding_score_sse for r in rows])
    mse_mean, mse_std = stats([r.binding_score_mse for r in rows])
    fsm_mean, fsm_std = stats([r.fsm_binding_score_err for r in rows])
    return CrossSequenceAggregate(
        sweep=sweep,
        swept_param=sweep,
        swept_value=swept_param_value_from_summary(ref, sweep),
        swept_label=swept_param_label_from_summary(ref, sweep),
        hv_mean=hv_mean,
        hv_std=hv_std,
        sse_mean=sse_mean,
        sse_std=sse_std,
        mse_mean=mse_mean,
        mse_std=mse_std,
        fsm_err_mean=fsm_mean,
        fsm_err_std=fsm_std,
        n_seq=len({r.seq_id for r in rows}),
    )


def swept_param_value_from_summary(row: DesignRunSummary, sweep: str) -> float:
    if sweep == "alpha":
        return row.alpha
    if sweep == "k":
        return float(row.k)
    return float(row.fsm_size)


def swept_param_label_from_summary(row: DesignRunSummary, sweep: str) -> str:
    if sweep == "alpha":
        return row.alpha_label
    if sweep == "k":
        return str(row.k)
    if abs(row.reduce_fsm_by) < 1e-9:
        return "DB"
    fold = round(1 / (1 - row.reduce_fsm_by))
    return f"{fold}-fold ({row.fsm_size})"


def aggregate_cross_sequence(
    design_runs: list[DesignRunSummary], sweep: str
) -> list[CrossSequenceAggregate]:
    filtered = [r for r in design_runs if r.sweeps and sweep in r.sweeps.split(",")]
    buckets: dict[str, list[DesignRunSummary]] = defaultdict(list)
    for row in filtered:
        label = swept_param_label_from_summary(row, sweep)
        buckets[label].append(row)

    out: list[CrossSequenceAggregate] = []
    for _label, rows in sorted(
        buckets.items(),
        key=lambda item: swept_param_value_from_summary(item[1][0], sweep),
    ):
        agg = _aggregate_bucket(rows, sweep)
        if agg:
            out.append(agg)
    return out


def aggregate_alpha_regime(
    design_runs: list[DesignRunSummary], regime: str
) -> list[CrossSequenceAggregate]:
    filtered = [
        r
        for r in design_runs
        if "alpha" in r.sweeps.split(",")
        and (("log_pos" if r.log_pos else "const") == regime)
    ]
    return aggregate_cross_sequence(filtered, "alpha")


SWEEP_CORREL_METRICS = [
    "hypervolume",
    "binding_score_sse",
    "binding_score_mse",
]

CORREL_METRIC_LABELS = {
    "hypervolume": "Hypervolume",
    "binding_score_sse": "binding_sse",
    "binding_score_mse": "binding_mse",
}

CORREL_ROW_LABEL = "Correlates with"


def filter_design_runs_by_sweep(
    design_runs: list[DesignRunSummary], sweep: str
) -> list[DesignRunSummary]:
    return [r for r in design_runs if r.sweeps and sweep in r.sweeps.split(",")]


def filter_design_runs_by_alpha_regime(
    design_runs: list[DesignRunSummary], regime: str
) -> list[DesignRunSummary]:
    return [
        r
        for r in design_runs
        if "alpha" in r.sweeps.split(",")
        and (("log_pos" if r.log_pos else "const") == regime)
    ]


def sort_design_runs(runs: list[DesignRunSummary]) -> list[DesignRunSummary]:
    return sorted(runs, key=lambda r: (r.seq_id, -r.fsm_size, -r.k))


def sort_solutions(solutions: list[SolutionRecord]) -> list[SolutionRecord]:
    return sorted(solutions, key=lambda r: (r.seq_id, -r.fsm_size, -r.k))


SWEEP_PARAM_ATTR = {
    "alpha": "alpha",
    "k": "k",
    "fsm_size": "fsm_size",
}

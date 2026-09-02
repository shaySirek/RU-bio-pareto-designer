from __future__ import annotations

from pareto_designer.shared.seq_design_utils.binding_metrics import (
    fsm_binding_score_mse,
    run_kmer_binding_score_mse_summary,
)
from pareto_designer.views.experiment_report.models import (
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
            kmer_binding_score_mse=r.kmer_binding_score_mse,
            kmer_binding_score_err_std=r.kmer_binding_score_err_std,
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
    run: LoadedRun, sweeps: list[str] | None = None
) -> DesignRunSummary:
    p = run.params
    s = p.sampler
    kmer_mse = run_kmer_binding_score_mse_summary(run.solutions)
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
        n_solutions=meta.get("n_solutions", len(run.solutions)),
        runtime_s=parse_runtime_seconds(meta),
        kmer_binding_score_mse_mean=kmer_mse.mse,
        kmer_binding_score_mse_solution_std=kmer_mse.err_std,
        db_fsm_size=int(meta.get("db_fsm_size", float("nan"))),
        fsm_binding_score_err=fsm_binding_score_mse(
            float(meta.get("fsm_binding_score_err", float("nan"))),
            int(meta.get("db_fsm_size", 0) or 0),
        ),
    )


def build_design_run_summaries(
    runs: list[LoadedRun], config: ExperimentConfig | None = None
) -> list[DesignRunSummary]:
    assign_sweep_memberships(runs, config)
    summaries_by_key: dict[tuple, DesignRunSummary] = {}

    for run in runs:
        sweeps = getattr(run, "_sweeps", sweep_membership(run.params, config))
        key = run.params.run_key
        if key not in summaries_by_key:
            summaries_by_key[key] = design_run_summary(run, sweeps)
        else:
            existing = summaries_by_key[key]
            merged_sweeps = sorted(set(existing.sweeps.split(",")) | set(sweeps))
            summaries_by_key[key] = design_run_summary(run, merged_sweeps)
    return list(summaries_by_key.values())


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

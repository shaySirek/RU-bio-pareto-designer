from __future__ import annotations

from pathlib import Path

from pareto_designer.algorithms.seq_design.sampling import SamplingMethod


def _short_cost_value(value: float) -> str:
    text = f"{value:g}"
    if "." not in text:
        text = f"{text}.0"
    return text


def format_cost_params_str(cost_params: dict[str, float]) -> str:
    return "_".join(
        f"{key}{_short_cost_value(float(cost_params[key]))}"
        for key in ("alpha", "beta", "w")
    )


def run_output_path(
    results_root: Path,
    target_sequence_id: str,
    cost_params_str: str,
    motif_id: str,
    fsm_id: str,
    sampler: SamplingMethod,
) -> Path:
    return (
        results_root
        / target_sequence_id
        / cost_params_str
        / motif_id
        / fsm_id
        / sampler.params
    )


def metadata_path(
    results_root: Path,
    target_sequence_id: str,
    cost_params_str: str,
    motif_id: str,
    fsm_id: str,
    sampler: SamplingMethod,
) -> Path:
    return (
        run_output_path(
            results_root,
            target_sequence_id,
            cost_params_str,
            motif_id,
            fsm_id,
            sampler,
        )
        / "results_metadata.json"
    )

from __future__ import annotations

from pathlib import Path

from pareto_designer.algorithms.seq_design.sampling import SamplingMethod


def norm_cost_param_key(key: str) -> str:
    return key.split(" ")[0].lower().replace("-", "_")


def format_cost_params_str(cost_params: dict[str, float]) -> str:
    return "__".join(
        f"{norm_cost_param_key(k)}_{p:.2f}"
        for k, p in cost_params.items()
        if isinstance(p, float)
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
        / type(sampler).__name__
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

import json
from os.path import commonpath
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

import numpy as np
from pymoo.indicators.hv import Hypervolume

from pareto_designer.shared.binding_utils import motif_hit_binding_thresholds
from pareto_designer.shared.csv_writer import write_results_stream
from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod
from pareto_designer.views.pareto_frontier.png_exporter import (
    render_pareto_frontiers,
    render_motif_cost_dist,
)


def compare_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path | None = None,
    pad: float = 0.1,
) -> dict[str, float]:
    if not frontiers:
        raise ValueError("The dictionary of frontiers cannot be empty.")
    if pad <= 0.0:
        raise ValueError("Pad must be strictly positive.")

    frontier_names = list(frontiers.keys())
    frontier_arrays = [np.atleast_2d(frontiers[name])[:, :2] for name in frontier_names]
    num_frontiers = len(frontier_arrays)
    all_points = np.vstack(frontier_arrays)
    num_objectives = all_points.shape[1]

    global_min = np.min(all_points, axis=0)
    global_max = np.max(all_points, axis=0)
    range_diff = global_max - global_min
    range_diff[range_diff == 0] = 1.0
    all_points_normalized = (all_points - global_min) / range_diff
    split_indices = np.cumsum([f.shape[0] for f in frontier_arrays])[:-1]
    normalized_frontiers = np.split(all_points_normalized, split_indices, axis=0)

    logger.debug(
        f"Comparing frontiers using Hypervolume [{global_min=}, {global_max=}, {pad=}]"
    )
    norm_ref_point = np.full(num_objectives, 1.0 + pad)
    hv_indicator = Hypervolume(ref_point=norm_ref_point)
    max_possible_volume = np.prod(norm_ref_point)
    normalized_hvs = [
        float(hv_indicator(norm_arr) / max_possible_volume)
        for norm_arr in normalized_frontiers
    ]

    coverage_matrix = np.zeros((num_frontiers, num_frontiers))
    for i in range(num_frontiers):
        for j in range(num_frontiers):
            if i == j:
                coverage_matrix[i, j] = 1.0
                continue

            combined_norm = np.vstack(
                (normalized_frontiers[i], normalized_frontiers[j])
            )
            hv_union = hv_indicator(combined_norm)
            hv_i_raw = normalized_hvs[i] * max_possible_volume
            hv_j_raw = normalized_hvs[j] * max_possible_volume
            intersection_volume = max(0.0, hv_i_raw + hv_j_raw - hv_union)
            if normalized_hvs[j] > 0:
                coverage_matrix[i, j] = float(intersection_volume / hv_j_raw)
            else:
                coverage_matrix[i, j] = 0.0

    output_data = {
        "global_bounds": {"min": global_min.tolist(), "max": global_max.tolist()},
        "metrics": {
            name: {
                "normalized_hypervolume": normalized_hvs[i],
                "coverage_over_others": {
                    frontier_names[j]: float(coverage_matrix[i, j])
                    for j in range(num_frontiers)
                },
            }
            for i, name in enumerate(frontier_names)
        },
    }

    if output_file is not None:
        with output_file.open("w") as f:
            json.dump(output_data, f, indent=4)

    return {name: normalized_hvs[i] for i, name in enumerate(frontier_names)}


def sampler_alpha_label(alpha: float, log_pos: bool) -> str:
    label = str(alpha)
    if log_pos:
        label += "_log_pos"
    return label


def parse_sampler_alpha(exp_str: str) -> tuple[float, bool]:
    log_pos = "_log_pos" in exp_str
    alpha = float(exp_str.split("_")[0])
    return alpha, log_pos


def render_and_compare(exporters: dict[str, ParetoExporter]) -> list[dict[str, Any]]:
    if not exporters:
        return []

    max_cost = 0.0
    max_positional_cost = 0.0
    min_binding = float("inf")
    max_binding = -float("inf")
    min_positional_binding = float("inf")
    max_positional_binding = -float("inf")
    for exporter in exporters.values():
        if not exporter._results:
            continue
        max_cost = max(max_cost, exporter.max_cost)
        max_positional_cost = max(max_positional_cost, exporter.max_positional_cost)
        min_binding = min(min_binding, exporter.min_binding)
        max_binding = max(max_binding, exporter.max_binding)
        min_positional_binding = min(
            min_positional_binding, exporter.min_positional_binding
        )
        max_positional_binding = max(
            max_positional_binding, exporter.max_positional_binding
        )

    first_exporter = list(exporters.values())[0]
    hit_thresholds = motif_hit_binding_thresholds(first_exporter.ctx.fsm_ctx)
    if min_binding == float("inf"):
        min_binding, max_binding = hit_thresholds[0], hit_thresholds[-1]
    if min_positional_binding == float("inf"):
        min_positional_binding, max_positional_binding = 0.0, 0.0
    binding_range = (min_binding, max_binding)
    positional_binding_range = (min_positional_binding, max_positional_binding)

    seen_fsms: set[str] = set()
    for exporter in exporters.values():
        fsm_id = exporter.ctx.fsm_ctx.fsm_id
        if fsm_id in seen_fsms:
            continue
        seen_fsms.add(fsm_id)
        exporter.render_target_sequence(max_positional_cost, positional_binding_range)

    frontiers: dict[str, np.ndarray] = {}
    for variant, exporter in exporters.items():
        logger.info(f"Rendering variant {variant}...")
        exporter.render(
            max_cost,
            binding_range,
            max_positional_cost,
            positional_binding_range,
            hit_thresholds,
        )
        if exporter._results:
            frontiers[variant] = exporter.frontier

    comparison_dir = Path(
        commonpath([str(e.output_path.resolve()) for e in exporters.values()])
    )
    labels = _display_labels(exporters)
    labeled_frontiers = {labels[name]: frontier for name, frontier in frontiers.items()}

    hypervolumes: dict[str, float] = {}
    if frontiers:
        hypervolumes = compare_frontiers(
            frontiers, comparison_dir / "pareto_comparison.json"
        )
        render_pareto_frontiers(
            labeled_frontiers,
            comparison_dir / "pareto_frontiers.png",
            max_cost,
            binding_range,
            hit_thresholds,
        )
        render_motif_cost_dist(
            labeled_frontiers,
            comparison_dir / "motif_cost_dists.png",
        )

    rows = list(_comparison_rows(exporters, hypervolumes))
    write_results_stream(
        iter(rows),
        comparison_dir / "pareto_comparison.csv",
    )
    return rows


def _sampler_alpha_label(sampler: SamplingMethod) -> str:
    return sampler_alpha_label(
        float(getattr(sampler, "alpha", 0.0)),
        bool(getattr(sampler, "use_dynamic_log_position_exponent", False)),
    )


def _fsm_fold_label(exporter: ParetoExporter) -> str:
    ctx = exporter.ctx.fsm_ctx
    db_size, n_states = ctx.db_fsm_size, ctx.size
    if n_states == db_size:
        return f"DB ({db_size})"
    if n_states and db_size % n_states == 0:
        return f"{db_size // n_states}-fold ({n_states})"
    return f"|V|={n_states}"


def _display_labels(exporters: dict[str, ParetoExporter]) -> dict[str, str]:
    vary_k = len({e.ctx.run_ctx.sampler.k for e in exporters.values()}) > 1
    vary_alpha = (
        len({_sampler_alpha_label(e.ctx.run_ctx.sampler) for e in exporters.values()})
        > 1
    )
    vary_fsm = len({e.ctx.fsm_ctx.fsm_id for e in exporters.values()}) > 1
    labels: dict[str, str] = {}
    for key, exporter in exporters.items():
        parts: list[str] = []
        if vary_k:
            parts.append(f"K={exporter.ctx.run_ctx.sampler.k}")
        if vary_alpha:
            parts.append(f"α={_sampler_alpha_label(exporter.ctx.run_ctx.sampler)}")
        if vary_fsm:
            parts.append(_fsm_fold_label(exporter))
        labels[key] = ", ".join(parts) if parts else key
    return labels


def _comparison_rows(
    exporters: dict[str, ParetoExporter], hypervolumes: dict[str, float]
) -> Iterator[dict[str, Any]]:
    for variant, exporter in exporters.items():
        sampler = exporter.ctx.run_ctx.sampler
        fsm_ctx = exporter.ctx.fsm_ctx
        yield {
            "seq_id": exporter.ctx.run_ctx.target_sequence_id,
            "variant": variant,
            "fsm_id": fsm_ctx.fsm_id,
            "fsm_size": fsm_ctx.size,
            "reduce_fsm_by": fsm_ctx.reduce_fsm_by,
            "k": sampler.k,
            "alpha": getattr(sampler, "alpha", None),
            "log_pos": getattr(sampler, "use_dynamic_log_position_exponent", False),
            "n_solutions": len(exporter._results),
            "binding_score_sse": exporter.binding_score_sse,
            "fsm_binding_score_err": exporter.fsm_binding_score_err,
            "normalized_hypervolume": hypervolumes.get(variant, float("nan")),
        }

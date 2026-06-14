import json
from pathlib import Path
from loguru import logger

import numpy as np
from pymoo.indicators.hv import Hypervolume

from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter
from pareto_designer.views.pareto_front.png_exporter import render_pareto_frontiers


def compare_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path,
    pad: float = 0.1,
) -> None:
    if not frontiers:
        raise ValueError("The dictionary of frontiers cannot be empty.")
    if pad <= 0.0:
        raise ValueError("Pad must be strictly positive.")

    frontier_names = list(frontiers.keys())
    frontier_arrays = [np.atleast_2d(frontiers[name]) for name in frontier_names]
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

    logger.info(
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

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=4)


def render_and_compare(exporters: dict[str, ParetoExporter]):
    max_cost = 0.0
    max_positional_cost = 0.0
    min_binding = float("inf")
    max_binding = -float("inf")
    min_positional_binding = float("inf")
    max_positional_binding = -float("inf")
    for exporter in exporters.values():
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
    binding_range = (min_binding, max_binding)
    positional_binding_range = (min_positional_binding, max_positional_binding)

    first_exporter = list(exporters.values())[0]
    first_exporter.render_target_sequence(max_positional_cost, positional_binding_range)

    frontiers = dict()
    for variant, exporter in exporters.items():
        logger.info(f"Rendering variant {variant}...")
        exporter.render(
            max_cost,
            binding_range,
            max_positional_cost,
            positional_binding_range,
        )
        frontiers[variant] = exporter.frontier

    compare_frontiers(
        frontiers, first_exporter.output_path.parent / "pareto_comparison.json"
    )
    render_pareto_frontiers(
        frontiers,
        first_exporter.output_path.parent / "pareto_frontiers.png",
        max_cost,
        binding_range,
    )

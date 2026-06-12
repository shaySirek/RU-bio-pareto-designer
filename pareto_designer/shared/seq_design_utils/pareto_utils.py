import json
from pathlib import Path
from loguru import logger

import numpy as np

from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter
from pareto_designer.views.pareto_front.png_exporter import render_pareto_frontiers


def compare_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path,
    grid_bins: int = 100,
    alpha: float = 1.1,
) -> dict:
    if not frontiers:
        raise ValueError("The dictionary of frontiers cannot be empty.")
    if alpha <= 1.0:
        raise ValueError("Alpha must be strictly greater than 1.0.")

    frontier_names = list(frontiers.keys())
    frontier_arrays = [np.atleast_2d(frontiers[name]) for name in frontier_names]

    num_frontiers = len(frontier_arrays)
    frontier_sizes = np.array([f.shape[0] for f in frontier_arrays])

    all_points = np.vstack(frontier_arrays)
    num_objectives = all_points.shape[1]

    grid_min = np.min(all_points, axis=0)
    grid_max = np.max(all_points, axis=0) * alpha
    logger.info(
        f"Compare frontiers using Grid Base Hyper Volume [{grid_bins=}, {grid_min=}, {grid_max=}]"
    )

    axes = [
        np.linspace(grid_min[d], grid_max[d], grid_bins) for d in range(num_objectives)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    flat_grid = np.stack([m.ravel() for m in mesh], axis=-1)

    less_equal = all_points[:, None, :] <= flat_grid[None, :, :]
    strictly_less = all_points[:, None, :] < flat_grid[None, :, :]
    global_cell_dominance = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)

    frontier_bounds = np.insert(np.cumsum(frontier_sizes)[:-1], 0, 0)
    frontier_dominates_cell = np.maximum.reduceat(
        global_cell_dominance, frontier_bounds, axis=0
    )

    grid_hypervolumes = np.mean(frontier_dominates_cell, axis=1)

    intersection_counts = np.dot(
        frontier_dominates_cell.astype(float), frontier_dominates_cell.T
    )
    cells_per_front = np.sum(frontier_dominates_cell, axis=1)

    coverage_matrix = np.where(
        cells_per_front[None, :] > 0,
        intersection_counts / cells_per_front[None, :],
        0.0,
    )
    np.fill_diagonal(coverage_matrix, 1.0)

    output_data = {
        "grid_bounds": {"min": grid_min.tolist(), "max": grid_max.tolist()},
        "metrics": {
            name: {
                "grid_hypervolume": float(grid_hypervolumes[i]),
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

    return output_data


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

import json
from pathlib import Path
from loguru import logger

import numpy as np

from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter


def compare_frontiers(
    fronts: dict[str, np.ndarray],
    output_file: Path,
    grid_bins: int = 100,
    alpha: float = 1.1,
) -> dict:
    if not fronts:
        raise ValueError("The dictionary of fronts cannot be empty.")
    if alpha <= 1.0:
        raise ValueError("Alpha must be strictly greater than 1.0.")

    front_names = list(fronts.keys())
    front_arrays = [np.atleast_2d(fronts[name]) for name in front_names]

    num_fronts = len(front_arrays)
    front_sizes = np.array([f.shape[0] for f in front_arrays])

    all_points = np.vstack(front_arrays)
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

    front_bounds = np.insert(np.cumsum(front_sizes)[:-1], 0, 0)
    front_dominates_cell = np.maximum.reduceat(
        global_cell_dominance, front_bounds, axis=0
    )

    grid_hypervolumes = np.mean(front_dominates_cell, axis=1)

    intersection_counts = np.dot(
        front_dominates_cell.astype(float), front_dominates_cell.T
    )
    cells_per_front = np.sum(front_dominates_cell, axis=1)

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
                    front_names[j]: float(coverage_matrix[i, j])
                    for j in range(num_fronts)
                },
            }
            for i, name in enumerate(front_names)
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

    first_exporter = list(exporters.values())[0]
    logger.info(f"[{min_positional_binding},{max_positional_binding}]")
    first_exporter.render_target_sequence(
        max_positional_cost, (min_positional_binding, max_positional_binding)
    )

    fronts_cmp_file = first_exporter.output_path.parent / "pareto_comparison.json"
    fronts = dict()
    for variant, exporter in exporters.items():
        logger.info(f"Rendering variant {variant}...")
        exporter.render(
            max_cost,
            (min_binding, max_binding),
            max_positional_cost,
            (min_positional_binding, max_positional_binding),
        )
        fronts[variant] = exporter.frontier

    compare_frontiers(fronts, fronts_cmp_file)

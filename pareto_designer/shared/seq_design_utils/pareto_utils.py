from os.path import commonpath
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

import numpy as np

from pareto_designer.shared.binding_utils import motif_hit_binding_thresholds
from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.csv_writer import write_results_stream
from pareto_designer.shared.seq_design_utils.binding_metrics import (
    run_kmer_binding_score_mse_summary,
)
from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter
from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
    export_alpha_sweep_roi_boxplot,
)
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod
from pareto_designer.views.pareto_frontier.png_exporter import (
    render_pareto_frontiers,
)


def sampler_alpha_label(alpha: float, log_pos: bool) -> str:
    label = str(alpha)
    if log_pos:
        label += "_log_pos"
    return label


def parse_sampler_alpha(exp_str: str) -> tuple[float, bool]:
    log_pos = "_log_pos" in exp_str
    alpha = float(exp_str.split("_")[0])
    return alpha, log_pos


def _long_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive:
        path_str = str(resolved)
        if not path_str.startswith("\\\\?\\"):
            return Path("\\\\?\\" + path_str)
    return resolved


def _is_design_run_dir(path: Path) -> bool:
    return (_long_path(path) / "results_metadata.json").exists()


def sweep_pareto_frontiers_filename(
    sweep_name: str, grid, *, alpha_group: str | None = None
) -> str:
    if sweep_name == "alpha":
        suffix = f"_{alpha_group}" if alpha_group else ""
        return f"sweep_alpha_K{grid.k_values[0]}{suffix}_pareto_frontiers.png"
    if sweep_name == "k":
        return f"sweep_K_alpha_{grid.sampler_alpha[0]}_pareto_frontiers.png"
    if sweep_name == "fsm_size":
        return (
            f"sweep_fsm_K{grid.k_values[0]}_alpha_{grid.sampler_alpha[0]}"
            "_pareto_frontiers.png"
        )
    raise ValueError(f"Unknown sweep name: {sweep_name!r}")


def _exporter_alpha_label(exporter: ParetoExporter) -> str:
    return _sampler_alpha_label(exporter.ctx.run_ctx.sampler)


def _ordered_labeled_frontiers_for_alphas(
    exporters: dict[str, ParetoExporter],
    frontiers: dict[str, np.ndarray],
    labels: dict[str, str],
    alpha_labels: tuple[str, ...],
) -> dict[str, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    for alpha in alpha_labels:
        for variant, exporter in exporters.items():
            if _exporter_alpha_label(exporter) != alpha:
                continue
            if variant not in frontiers:
                continue
            grouped[labels[variant]] = frontiers[variant]
            break
    return grouped


def _nonsyn_w_from_exporter(exporter: ParetoExporter) -> float | None:
    w = getattr(exporter.ctx.score_function, "w", None)
    if w is None:
        w = exporter.ctx.run_ctx.cost_params.get("w")
    if w is None or not np.isfinite(w):
        return None
    return float(w)


def _labeled_results(
    exporters: dict[str, ParetoExporter],
    labels: dict[str, str],
    variant_names: dict[str, np.ndarray] | None = None,
) -> dict[str, list[ParetoResult]]:
    out: dict[str, list[ParetoResult]] = {}
    for name, exporter in exporters.items():
        if variant_names is not None and name not in variant_names:
            continue
        if not exporter._results:
            continue
        out[labels[name]] = exporter._results
    return out


def _render_comparison_frontiers(
    labeled_frontiers: dict[str, np.ndarray],
    comparison_png: Path,
    max_cost: float,
    binding_range: tuple[float, float],
    hit_thresholds: list[float],
    *,
    origin_frontiers: dict[str, np.ndarray] | None = None,
    db_fsm_labels: set[str] | None = None,
    results_by_label: dict[str, list[ParetoResult]] | None = None,
    nonsyn_w: float | None = None,
) -> None:
    labeled_origin = None
    if origin_frontiers is not None:
        labeled_origin = {
            label: origin_frontiers[label]
            for label in labeled_frontiers
            if label in origin_frontiers
        }
    render_pareto_frontiers(
        labeled_frontiers,
        comparison_png,
        max_cost,
        binding_range,
        hit_thresholds,
        origin_frontiers=labeled_origin,
        db_fsm_labels=db_fsm_labels,
        results_by_label=results_by_label,
        nonsyn_w=nonsyn_w,
    )


def render_and_compare(
    exporters: dict[str, ParetoExporter],
    *,
    sweep_name: str | None = None,
    sweep_grid=None,
    skip_render_per_solution: bool = False,
    alpha_comparison_groups: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
) -> list[dict[str, Any]]:
    if not exporters:
        return []

    max_cost = 0.0
    max_positional_cost = 0.0
    min_binding = float("inf")
    max_binding = -float("inf")
    min_positional_binding = float("inf")
    max_positional_binding = -float("inf")
    show_origin_lines = sweep_name == "fsm_size"
    for exporter in exporters.values():
        if not exporter._results:
            continue
        max_cost = max(max_cost, exporter.max_cost)
        max_positional_cost = max(max_positional_cost, exporter.max_positional_cost)
        min_binding = min(min_binding, exporter.min_binding)
        max_binding = max(max_binding, exporter.max_binding)
        if show_origin_lines:
            min_binding = min(min_binding, exporter.min_origin_binding)
            max_binding = max(max_binding, exporter.max_origin_binding)
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
    if not skip_render_per_solution:
        for exporter in exporters.values():
            fsm_id = exporter.ctx.fsm_ctx.fsm_id
            if fsm_id in seen_fsms:
                continue
            seen_fsms.add(fsm_id)
            exporter.render_target_sequence(
                max_positional_cost, positional_binding_range
            )

    frontiers: dict[str, np.ndarray] = {}
    for variant, exporter in exporters.items():
        logger.info(f"Rendering variant {variant}...")
        exporter.render(
            max_cost,
            binding_range,
            max_positional_cost,
            positional_binding_range,
            hit_thresholds,
            skip_render_per_solution=skip_render_per_solution,
        )
        if exporter._results:
            frontiers[variant] = exporter.frontier

    comparison_dir = Path(
        commonpath([str(e.output_path.resolve()) for e in exporters.values()])
    )
    write_comparison = not _is_design_run_dir(comparison_dir)
    labels = _display_labels(exporters)
    labeled_frontiers = {labels[name]: frontier for name, frontier in frontiers.items()}
    nonsyn_w = _nonsyn_w_from_exporter(first_exporter)
    labeled_results = _labeled_results(exporters, labels, frontiers)

    origin_frontiers: dict[str, np.ndarray] | None = None
    db_fsm_labels: set[str] | None = None
    if show_origin_lines:
        origin_frontiers = {
            labels[name]: exporter.origin_frontier
            for name, exporter in exporters.items()
            if name in frontiers
        }
        db_fsm_labels = {
            labels[name]
            for name, exporter in exporters.items()
            if name in frontiers and exporter.ctx.fsm_ctx.reduce_fsm_by == 0
        }

    if write_comparison and frontiers:
        if sweep_name == "alpha" and sweep_grid is not None and alpha_comparison_groups:
            for group_name, alpha_labels in alpha_comparison_groups:
                group_frontiers = _ordered_labeled_frontiers_for_alphas(
                    exporters, frontiers, labels, alpha_labels
                )
                if not group_frontiers:
                    continue
                group_results = {
                    label: labeled_results[label]
                    for label in group_frontiers
                    if label in labeled_results
                }
                comparison_png = comparison_dir / sweep_pareto_frontiers_filename(
                    sweep_name, sweep_grid, alpha_group=group_name
                )
                _render_comparison_frontiers(
                    group_frontiers,
                    comparison_png,
                    max_cost,
                    binding_range,
                    hit_thresholds,
                    results_by_label=group_results,
                    nonsyn_w=nonsyn_w,
                )
        else:
            if sweep_name is not None and sweep_grid is not None:
                comparison_png = comparison_dir / sweep_pareto_frontiers_filename(
                    sweep_name, sweep_grid
                )
            else:
                comparison_png = comparison_dir / "pareto_frontiers.png"
            _render_comparison_frontiers(
                labeled_frontiers,
                comparison_png,
                max_cost,
                binding_range,
                hit_thresholds,
                origin_frontiers=origin_frontiers,
                db_fsm_labels=db_fsm_labels,
                results_by_label=(
                    labeled_results if sweep_name != "fsm_size" else None
                ),
                nonsyn_w=nonsyn_w,
            )

        if sweep_name == "alpha" and sweep_grid is not None:
            export_alpha_sweep_roi_boxplot(
                exporters,
                comparison_dir,
                k=sweep_grid.k_values[0],
                nonsyn_w=nonsyn_w,
            )

    rows = list(_comparison_rows(exporters))
    if write_comparison:
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
    return f"|V|={exporter.ctx.fsm_ctx.size}"


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


def _comparison_rows(exporters: dict[str, ParetoExporter]) -> Iterator[dict[str, Any]]:
    for variant, exporter in exporters.items():
        sampler = exporter.ctx.run_ctx.sampler
        fsm_ctx = exporter.ctx.fsm_ctx
        kmer_mse = run_kmer_binding_score_mse_summary(exporter._results)
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
            "kmer_binding_score_mse_mean": kmer_mse.mse,
            "kmer_binding_score_mse_solution_std": kmer_mse.err_std,
            "db_fsm_size": fsm_ctx.db_fsm_size,
            "fsm_binding_score_err": exporter.fsm_binding_score_err,
        }

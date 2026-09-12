from enum import StrEnum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import gaussian_kde

from pareto_designer.models.context import RunContext, ParetoResult
from pareto_designer.shared.seq_design_utils.solution_quality import (
    SolutionRegion,
    region_borders,
)
from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
    frontier_line_color,
    mark_first_hit_free,
    overlay_run_quality,
    region_legend_handle,
    regions_for_results,
    scatter_classified_points,
)


class FrontierPlotStyle(StrEnum):
    POINTS = "points"
    LINES = "lines"
    LINES_ANNO = "lines_anno"


def _frontier_line_color(index: int) -> str:
    return frontier_line_color(index)


def _sorted_by_cost(
    costs: np.ndarray, bindings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(costs)
    return costs[order], bindings[order]


def _frontier_legend_handles(
    frontiers: dict[str, np.ndarray], line_colors: dict[str, str]
) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=line_colors[key],
            linewidth=1.5,
            label=key,
        )
        for key in frontiers.keys()
    ]


def _frontier_point_legend_handles(
    frontiers: dict[str, np.ndarray], line_colors: dict[str, str]
) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=line_colors[key],
            marker="o",
            linestyle="None",
            markersize=5,
            label=key,
        )
        for key in frontiers.keys()
    ]


def _filter_frontier_by_binding(
    frontier: np.ndarray, min_binding: float = 0.0
) -> np.ndarray:
    if frontier.size == 0:
        return frontier
    return frontier[frontier[:, 1] >= min_binding]


_KDE_GRID_SIZE = 256


def _gaussian_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 1 or np.allclose(values, values[0]):
        span = float(grid[-1] - grid[0])
        bandwidth = max(span / 40.0, 1e-6)
        return np.exp(-0.5 * ((grid - values[0]) / bandwidth) ** 2) / (
            bandwidth * np.sqrt(2 * np.pi)
        )
    return gaussian_kde(values)(grid)


def cost_histogram_lines(
    costs_by_label: dict[str, list[float] | np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    series = {
        label: np.asarray(costs, dtype=float)
        for label, costs in costs_by_label.items()
        if len(costs) > 0
    }
    if not series:
        return None
    all_costs = np.concatenate(list(series.values()))
    lo = float(all_costs.min())
    hi = float(all_costs.max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, _KDE_GRID_SIZE)
    densities = {
        label: _gaussian_density(costs, grid) for label, costs in series.items()
    }
    return grid, densities


def render_cost_hist_lines(
    costs_by_label: dict[str, list[float] | np.ndarray],
    output_file: Path,
) -> Path | None:
    hist = cost_histogram_lines(costs_by_label)
    if hist is None:
        return None
    grid, density_by_label = hist
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, (label, density) in enumerate(density_by_label.items()):
        ax.plot(
            grid,
            density,
            color=_frontier_line_color(idx),
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Functional cost")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def render_pareto_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path,
    max_cost: float,
    binding_range: tuple[float, float],
    *,
    origin_frontiers: dict[str, np.ndarray] | None = None,
    db_fsm_labels: set[str] | None = None,
    results_by_label: dict[str, list[ParetoResult]] | None = None,
    nonsyn_w: float | None = None,
    plot_style: FrontierPlotStyle = FrontierPlotStyle.LINES_ANNO,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted_frontiers: dict[str, np.ndarray] = {}
    line_colors: dict[str, str] = {}
    plotted_costs: list[float] = []
    plotted_bindings: list[float] = []
    regions_in_legend: set[SolutionRegion] = set()
    draw_lines = plot_style != FrontierPlotStyle.POINTS
    draw_classified = plot_style == FrontierPlotStyle.LINES_ANNO
    draw_unclassified = plot_style == FrontierPlotStyle.POINTS
    for idx, (key, frontier) in enumerate(frontiers.items()):
        color = _frontier_line_color(idx)
        filtered = _filter_frontier_by_binding(frontier)
        if filtered.size == 0:
            continue
        plotted_frontiers[key] = filtered
        line_colors[key] = color
        costs, bindings = _sorted_by_cost(filtered[:, 0], filtered[:, 1])
        plotted_costs.extend(costs.tolist())
        plotted_bindings.extend(bindings.tolist())
        if draw_lines:
            ax.plot(costs, bindings, color=color, linewidth=1.5, label=key, zorder=1)
        if draw_unclassified:
            ax.scatter(
                costs,
                bindings,
                s=12,
                color=color,
                edgecolors="none",
                alpha=0.85,
                zorder=3,
            )

        if results_by_label and key in results_by_label:
            run_results = results_by_label[key]
            if draw_classified:
                regions = regions_for_results(run_results, nonsyn_w=nonsyn_w)
                for region in scatter_classified_points(ax, run_results, regions):
                    regions_in_legend.add(region)
            if draw_unclassified:
                borders = region_borders(run_results, w=nonsyn_w)
                hit_free_cost = borders.first_hit_free_cost
                hit_free_binding = borders.first_hit_free_binding
                if mark_first_hit_free(ax, borders, color=color):
                    if hit_free_cost is not None and hit_free_binding is not None:
                        plotted_costs.append(hit_free_cost)
                        plotted_bindings.append(hit_free_binding)

        if origin_frontiers is not None and key in origin_frontiers:
            if db_fsm_labels and key in db_fsm_labels:
                continue
            if not draw_lines:
                continue
            mask = frontier[:, 1] >= 0.0
            origin = origin_frontiers[key][mask]
            if origin.size == 0:
                continue
            origin_costs, origin_bindings = _sorted_by_cost(origin[:, 0], origin[:, 1])
            plotted_costs.extend(origin_costs.tolist())
            plotted_bindings.extend(origin_bindings.tolist())
            ax.plot(
                origin_costs,
                origin_bindings,
                color=color,
                linewidth=1.5,
                linestyle="--",
                zorder=1,
            )

    plot_max_cost = max(plotted_costs) if plotted_costs else max_cost
    if plotted_bindings:
        plot_binding_range = (min(plotted_bindings), max(plotted_bindings))
    else:
        plot_binding_range = binding_range

    _set_pareto_axes(ax, plot_max_cost, plot_binding_range)
    if plotted_frontiers:
        if plot_style == FrontierPlotStyle.POINTS:
            legend_handles = _frontier_point_legend_handles(
                plotted_frontiers, line_colors
            )
        else:
            legend_handles = _frontier_legend_handles(plotted_frontiers, line_colors)
            legend_handles.extend(
                region_legend_handle(region)
                for region in SolutionRegion
                if region in regions_in_legend
            )
        ax.legend(handles=legend_handles, loc="upper right")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def render_pareto_frontier_png(
    ctx: RunContext,
    results: list[ParetoResult],
    max_cost: float,
    binding_range: tuple[float, float],
    *,
    is_db_fsm: bool = True,
    nonsyn_w: float | None = None,
):
    sorted_results = sorted(results, key=lambda r: r.cost)
    costs = [r.cost for r in sorted_results]
    bindings = [r.binding_score for r in sorted_results]
    origin_bindings = [r.origin_binding_score for r in sorted_results]

    y_values = list(bindings)
    if not is_db_fsm:
        y_values.extend(origin_bindings)
    plot_max_cost = max(costs) if costs else max_cost
    if y_values:
        plot_binding_range = (min(y_values), max(y_values))
    else:
        plot_binding_range = binding_range

    fig, ax = plt.subplots(figsize=(5, 4))
    line_color = _frontier_line_color(0)
    fsm_label = "Binding score" if is_db_fsm else "Reduced FSM"
    ax.plot(costs, bindings, color=line_color, linewidth=1.5, label=fsm_label, zorder=1)

    if not is_db_fsm:
        ax.plot(
            costs,
            origin_bindings,
            color=line_color,
            linewidth=1.5,
            linestyle="--",
            label="Origin (DB FSM)",
            zorder=1,
        )

    w = nonsyn_w
    if w is None:
        w = ctx.cost_params.get("w")
    region_point_handles = overlay_run_quality(ax, sorted_results, nonsyn_w=w)

    _set_pareto_axes(ax, plot_max_cost, plot_binding_range)
    legend_handles = [
        mlines.Line2D([], [], color=line_color, linewidth=1.5, label=fsm_label)
    ]
    if not is_db_fsm:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=line_color,
                linewidth=1.5,
                linestyle="--",
                label="Origin (DB FSM)",
            )
        )
    legend_handles.extend(region_point_handles)
    ax.legend(handles=legend_handles, loc="upper right")

    fig.savefig(
        ctx.output_path / "pareto_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _set_pareto_axes(
    ax: Axes,
    max_cost: float,
    binding_range: tuple[float, float],
):
    x_max = max_cost * 1.05
    min_binding, max_binding = binding_range
    y_margin = (max_binding - min_binding) * 0.05
    if y_margin == 0:
        y_margin = 1.0
    y_min = min_binding - y_margin
    y_max = max_binding + y_margin
    ax.set_xlabel("Functional cost")
    ax.set_ylabel("Binding score")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)


def render_heatmap_png(
    ctx: RunContext,
    seq_id: str,
    costs: np.ndarray | None,
    binding: np.ndarray,
    motif_hits: list[tuple[int, int]],
    max_cost: float,
    binding_range: tuple[float, float],
):
    seq_len = len(binding)
    width = min(max(10, seq_len * 0.02), 40)
    cmaps = _get_cmaps(max_cost, binding_range)
    folder = ctx.output_path

    if costs is not None:
        fig, axes = plt.subplots(2, 1, figsize=(width, 2.0), sharex=True)
        fig.subplots_adjust(hspace=0.1)
        ax_cost, ax_binding = axes[0], axes[1]
        sns.heatmap(
            costs.reshape(1, -1),
            **cmaps["Functinal cost"],
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax_cost,
        )
    else:
        folder = folder.parent
        fig, axes = plt.subplots(1, 1, figsize=(width, 1.0), sharex=True)
        ax_binding = axes
        for s, e in ctx.orfs:
            ax_binding.axvspan(s - 0.5, e - 0.5, color="darkblue", alpha=0.1, zorder=0)
            ax_binding.plot(
                [s - 0.5, e - 0.5],
                [-0.4, -0.4],
                color="darkblue",
                lw=4,
                transform=ax_binding.get_xaxis_transform(),
                clip_on=False,
            )

    sns.heatmap(
        binding.reshape(1, -1),
        **cmaps["Binding score"],
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax_binding,
    )
    hit_mask = np.full(len(binding), np.nan)
    for start, end in motif_hits:
        hit_mask[start:end] = 1
    if not np.all(np.isnan(hit_mask)):
        sns.heatmap(
            hit_mask.reshape(1, -1),
            cmap=mcolors.ListedColormap(["black"]),
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax_binding,
            zorder=3,
        )

    ticks = np.arange(0, seq_len, 100)
    ax_binding.set_xticks(ticks + 0.5)
    ax_binding.set_xticklabels(ticks, fontsize=12)

    fig.savefig(folder / f"{seq_id}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_heatmap_legend(
    ctx: RunContext,
    max_cost: float,
    binding_range: tuple[float, float],
):
    bars = _get_cmaps(max_cost, binding_range)
    n = len(bars)
    fig, axes = plt.subplots(1, len(bars), figsize=(4 * n, 0.5), squeeze=False)

    i = 0
    for name, cmap_kwargs in bars.items():
        ax = axes[0, i]
        fig.colorbar(
            plt.cm.ScalarMappable(**cmap_kwargs),
            cax=ax,
            orientation="horizontal",
        )
        ax.set_title(name, fontsize=10, pad=5)
        i += 1

    fig.savefig(
        ctx.output_path.parent / "heatmap_legend.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def _get_cmaps(
    max_cost: float,
    binding_range: tuple[float, float],
):
    return {
        "Functinal cost": dict(cmap="Reds", norm=mcolors.Normalize(0.0, max_cost)),
        "Binding score": dict(
            cmap="Purples", norm=mcolors.TwoSlopeNorm(0.0, *binding_range)
        ),
    }


def kmer_binding_score_mse_values(results: list[ParetoResult]) -> list[float]:
    return [
        r.kmer_binding_score_mse
        for r in results
        if np.isfinite(r.kmer_binding_score_mse)
    ]


def kmer_mse_histogram_series(
    values_by_label: dict[str, list[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        label: np.asarray(values, dtype=float)
        for label, values in values_by_label.items()
        if len(values) > 0
    }


def shared_hist_bins(series: dict[str, np.ndarray]) -> np.ndarray:
    all_values = np.concatenate(list(series.values()))
    n_bins = min(50, max(5, min(len(values) for values in series.values())))
    return np.histogram_bin_edges(all_values, bins=n_bins)


def draw_labeled_histograms(
    ax: Axes,
    series: dict[str, np.ndarray],
    bins: np.ndarray,
) -> None:
    for idx, (label, values) in enumerate(series.items()):
        ax.hist(
            values,
            bins=bins,
            color=_frontier_line_color(idx),
            alpha=0.7,
            **({"label": label} if label else {}),
        )


def render_kmer_binding_score_mse_histograms(
    values_by_label: dict[str, list[float] | np.ndarray],
    output_file: Path,
    *,
    vlines: list[tuple[float, str, str]] | None = None,
) -> Path | None:
    series = kmer_mse_histogram_series(values_by_label)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_labeled_histograms(ax, series, shared_hist_bins(series))
    for value, linestyle, label in vlines or ():
        ax.axvline(value, linestyle=linestyle, color="black", linewidth=1, label=label)
    ax.set_xlabel("per-solution binding score MSE")
    ax.set_ylabel("frequency")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def render_kmer_binding_score_mse_histogram(
    ctx: RunContext,
    results: list[ParetoResult],
    fsm_mse: float | None = None,
):
    values = kmer_binding_score_mse_values(results)
    if not values:
        return
    mean = float(np.mean(values))
    extra: list[tuple[float, str, str]] = [
        (mean, "-", "mean per-solution MSE"),
    ]
    if fsm_mse is not None and np.isfinite(fsm_mse):
        extra.append((fsm_mse, "--", "FSM MSE"))
    render_kmer_binding_score_mse_histograms(
        {"": values},
        ctx.output_path / "kmer_binding_score_mse_histogram.png",
        vlines=extra,
    )


def render_scatter_binding_scores(
    ctx: RunContext,
    results: list[ParetoResult],
):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = [r.origin_binding_score for r in results]
    y = [r.binding_score for r in results]
    ax.scatter(x, y, s=15, linewidths=0.3, edgecolors="black", alpha=0.8)
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    margin = (hi - lo) * 0.05
    if margin == 0:
        margin = 1.0
    lo -= margin
    hi += margin
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Binding score")
    ax.set_ylabel("FSM Binding score")
    fig.savefig(
        ctx.output_path / "binding_scores_scatter.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

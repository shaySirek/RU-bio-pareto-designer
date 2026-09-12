from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.solution_quality.classify import (
    RegionBorders,
    SolutionRegion,
    classify_run_solutions,
    region_borders,
)

if TYPE_CHECKING:
    from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter

_PLOT_DPI = 150
RoiPoint = tuple[float, float]

REGION_COLORS = {
    SolutionRegion.HITS: "tab:red",
    SolutionRegion.NONSYN: "tab:purple",
    SolutionRegion.ROI: "tab:green",
    SolutionRegion.PLATEAU: "tab:orange",
}

# Line/box colors chosen to avoid overlap with REGION_COLORS.
FRONTIER_LINE_COLORS = [
    "#333333",
    "#1f77b4",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#005f8a",
    "#aec7e8",
    "#4a4a4a",
]


def frontier_line_color(index: int) -> str:
    return FRONTIER_LINE_COLORS[index % len(FRONTIER_LINE_COLORS)]


def _color_boxplot(box: dict, colors: list[str]) -> None:
    for i, color in enumerate(colors):
        if i < len(box["boxes"]):
            box["boxes"][i].set_facecolor(color)
            box["boxes"][i].set_edgecolor(color)
            box["boxes"][i].set_alpha(0.55)
        if i < len(box["medians"]):
            box["medians"][i].set_color(color)
        for key in ("whiskers", "caps"):
            left = box[key][2 * i] if 2 * i < len(box[key]) else None
            right = box[key][2 * i + 1] if 2 * i + 1 < len(box[key]) else None
            for artist in (left, right):
                if artist is not None:
                    artist.set_color(color)
        if i < len(box["fliers"]):
            box["fliers"][i].set_markeredgecolor(color)
            box["fliers"][i].set_markerfacecolor(color)


BORDER_NO_HITS_COLOR = "#1f77b4"
BORDER_PLATEAU_COLOR = "#8c564b"


def regions_for_results(
    results: list[ParetoResult],
    *,
    nonsyn_w: float | None = None,
) -> dict[str, SolutionRegion]:
    _, regions = classify_run_solutions(results, w=nonsyn_w)
    return regions


def scatter_classified_points(
    ax: Axes,
    results: list[ParetoResult],
    regions: dict[str, SolutionRegion],
) -> set[SolutionRegion]:
    drawn: set[SolutionRegion] = set()
    for region in SolutionRegion:
        points = [(r.cost, r.binding_score) for r in results if regions[r.id] == region]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(
            xs,
            ys,
            s=12,
            color=REGION_COLORS[region],
            edgecolors="none",
            alpha=0.85,
            zorder=3,
        )
        drawn.add(region)
    return drawn


def mark_first_hit_free(
    ax: Axes,
    borders: RegionBorders,
    *,
    color: str,
) -> bool:
    if borders.first_hit_free_cost is None or borders.first_hit_free_binding is None:
        return False
    ax.scatter(
        [borders.first_hit_free_cost],
        [borders.first_hit_free_binding],
        marker="*",
        s=80,
        color=color,
        edgecolors="black",
        linewidths=0.4,
        zorder=4,
    )
    return True


def draw_region_borders(ax: Axes, borders: RegionBorders) -> None:
    if borders.first_hit_free_cost is not None:
        ax.axvline(
            borders.first_hit_free_cost,
            linestyle=":",
            color=BORDER_NO_HITS_COLOR,
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            borders.first_hit_free_cost,
            1.01,
            "no hits",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="center",
            fontsize=8,
            color=BORDER_NO_HITS_COLOR,
            rotation=90,
        )

    if (
        borders.plateau_onset_cost is not None
        and borders.plateau_onset_cost != borders.first_hit_free_cost
    ):
        ax.axvline(
            borders.plateau_onset_cost,
            linestyle=":",
            color=BORDER_PLATEAU_COLOR,
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            borders.plateau_onset_cost,
            1.01,
            "plateau",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            ha="center",
            fontsize=8,
            color=BORDER_PLATEAU_COLOR,
            rotation=90,
        )


def region_legend_handle(region: SolutionRegion) -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=5,
        color=REGION_COLORS[region],
        label=region.value,
    )


def region_legend_handles(regions: set[SolutionRegion]) -> list[mlines.Line2D]:
    return [
        region_legend_handle(region) for region in SolutionRegion if region in regions
    ]


def overlay_run_quality(
    ax: Axes,
    results: list[ParetoResult],
    *,
    nonsyn_w: float | None = None,
) -> list[mlines.Line2D]:
    regions = regions_for_results(results, nonsyn_w=nonsyn_w)
    draw_region_borders(ax, region_borders(results, w=nonsyn_w))
    drawn = scatter_classified_points(ax, results, regions)
    return region_legend_handles(drawn)


def alpha_roi_boxplot_filename(k: int, group_name: str) -> str:
    return f"sweep_alpha_K{k}_{group_name}_roi_boxwhisker.png"


def alpha_roi_violin_filename(k: int, group_name: str) -> str:
    return f"sweep_alpha_K{k}_{group_name}_roi_violin.png"


def k_roi_boxplot_filename(alpha) -> str:
    return f"sweep_K_alpha_{alpha}_roi_boxwhisker.png"


def k_roi_violin_filename(alpha) -> str:
    return f"sweep_K_alpha_{alpha}_roi_violin.png"


def _alpha_label_sort_key(label: str) -> tuple[float, bool]:
    log_pos = "_log_pos" in label
    alpha = float(label.split("_")[0])
    return (alpha, log_pos)


def _alpha_label_from_exporter(exporter: ParetoExporter) -> str:
    sampler = exporter.ctx.run_ctx.sampler
    alpha = float(getattr(sampler, "alpha", 0.0))
    log_pos = bool(getattr(sampler, "use_dynamic_log_position_exponent", False))
    label = str(alpha)
    if log_pos:
        label += "_log_pos"
    return label


def roi_points_grouped(
    exporters: dict[str, ParetoExporter],
    label_of: Callable[[ParetoExporter], str],
    *,
    nonsyn_w: float | None = None,
) -> tuple[str | None, dict[str, list[RoiPoint]]]:
    roi_by_label: dict[str, list[RoiPoint]] = defaultdict(list)
    seq_id: str | None = None
    for exporter in exporters.values():
        if not exporter._results:
            continue
        seq_id = exporter.ctx.run_ctx.target_sequence_id
        label = label_of(exporter)
        regions = regions_for_results(exporter._results, nonsyn_w=nonsyn_w)
        for result in exporter._results:
            if regions[result.id] == SolutionRegion.ROI:
                roi_by_label[label].append((result.cost, result.binding_score))
    return seq_id, dict(roi_by_label)


def _ordered_roi_labels(
    roi_by_label: dict[str, list[RoiPoint]],
    *,
    label_order: tuple[str, ...] | None = None,
    sort_key: Callable[[str], object] | None = None,
) -> tuple[list[str], list[str]] | None:
    if not roi_by_label:
        return None
    if label_order is None:
        labels = (
            sorted(roi_by_label, key=sort_key) if sort_key else list(roi_by_label)
        )
        color_indices = list(range(len(labels)))
    else:
        labels = [label for label in label_order if roi_by_label.get(label)]
        color_indices = [label_order.index(label) for label in labels]
    if not labels:
        return None
    colors = [frontier_line_color(idx) for idx in color_indices]
    return labels, colors


def _draw_roi_box(
    ax: Axes, groups: list[list[float]], labels: list[str], colors: list[str]
) -> None:
    box = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
    )
    _color_boxplot(box, colors)


def _draw_roi_density_2d(
    ax: Axes, points: list[RoiPoint], color: str
) -> None:
    costs = [cost for cost, _binding in points]
    bindings = [binding for _cost, binding in points]
    if len(points) >= 3 and len({(c, b) for c, b in points}) >= 3:
        try:
            sns.kdeplot(
                x=costs,
                y=bindings,
                ax=ax,
                color=color,
                fill=True,
                alpha=0.35,
                thresh=0.05,
                levels=6,
                warn_singular=False,
            )
            sns.kdeplot(
                x=costs,
                y=bindings,
                ax=ax,
                color=color,
                fill=False,
                linewidths=1.2,
                thresh=0.05,
                levels=6,
                warn_singular=False,
            )
            return
        except ValueError:
            pass
    ax.scatter(costs, bindings, s=12, color=color, edgecolors="none", alpha=0.85)


def render_roi_boxplot(
    roi_by_label: dict[str, list[RoiPoint]],
    output_path: Path,
    *,
    label_order: tuple[str, ...] | None = None,
    xlabel: str,
    sort_key: Callable[[str], object] | None = None,
) -> Path | None:
    ordered = _ordered_roi_labels(
        roi_by_label, label_order=label_order, sort_key=sort_key
    )
    if ordered is None:
        return None
    labels, colors = ordered
    cost_groups = [[cost for cost, _binding in roi_by_label[label]] for label in labels]
    binding_groups = [
        [binding for _cost, binding in roi_by_label[label]] for label in labels
    ]
    fig, (ax_cost, ax_binding) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, groups, ylabel in (
        (ax_cost, cost_groups, "Functional cost"),
        (ax_binding, binding_groups, "Binding score"),
    ):
        _draw_roi_box(ax, groups, labels, colors)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _roi_series_legend_label(label: str, prefix: str | None) -> str:
    if prefix:
        return f"{prefix}={label}"
    return label


def render_roi_violin(
    roi_by_label: dict[str, list[RoiPoint]],
    output_path: Path,
    *,
    label_order: tuple[str, ...] | None = None,
    sort_key: Callable[[str], object] | None = None,
    series_prefix: str | None = None,
) -> Path | None:
    ordered = _ordered_roi_labels(
        roi_by_label, label_order=label_order, sort_key=sort_key
    )
    if ordered is None:
        return None
    labels, colors = ordered
    fig, ax = plt.subplots(figsize=(5, 4))
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            linewidth=1.5,
            label=_roi_series_legend_label(label, series_prefix),
        )
        for label, color in zip(labels, colors)
    ]
    for label, color in zip(labels, colors):
        _draw_roi_density_2d(ax, roi_by_label[label], color)
    ax.set_xlabel("Functional cost")
    ax.set_ylabel("Binding score")
    ax.legend(handles=legend_handles, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_roi_plots(
    exporters: dict[str, ParetoExporter],
    comparison_dir: Path,
    *,
    label_of: Callable[[ParetoExporter], str],
    label_order: tuple[str, ...],
    box_filename: str,
    violin_filename: str,
    xlabel: str,
    series_prefix: str,
    sort_key: Callable[[str], object] | None = None,
    nonsyn_w: float | None = None,
) -> tuple[Path | None, Path | None]:
    seq_id, roi_by_label = roi_points_grouped(
        exporters, label_of, nonsyn_w=nonsyn_w
    )
    grouped = {
        label: roi_by_label[label]
        for label in label_order
        if label in roi_by_label and roi_by_label[label]
    }
    if not grouped or seq_id is None:
        return None, None
    return (
        render_roi_boxplot(
            grouped,
            comparison_dir / box_filename,
            label_order=label_order,
            xlabel=xlabel,
            sort_key=sort_key,
        ),
        render_roi_violin(
            grouped,
            comparison_dir / violin_filename,
            label_order=label_order,
            sort_key=sort_key,
            series_prefix=series_prefix,
        ),
    )


def export_alpha_sweep_roi_boxplot(
    exporters: dict[str, ParetoExporter],
    comparison_dir: Path,
    *,
    k: int,
    group_name: str,
    alpha_labels: tuple[str, ...],
    nonsyn_w: float | None = None,
) -> tuple[Path | None, Path | None]:
    return export_roi_plots(
        exporters,
        comparison_dir,
        label_of=_alpha_label_from_exporter,
        label_order=alpha_labels,
        box_filename=alpha_roi_boxplot_filename(k, group_name),
        violin_filename=alpha_roi_violin_filename(k, group_name),
        xlabel="alpha",
        series_prefix="α",
        sort_key=_alpha_label_sort_key,
        nonsyn_w=nonsyn_w,
    )


def export_k_sweep_roi_boxplot(
    exporters: dict[str, ParetoExporter],
    comparison_dir: Path,
    *,
    alpha,
    k_values: list[int] | tuple[int, ...],
    nonsyn_w: float | None = None,
) -> tuple[Path | None, Path | None]:
    k_order = tuple(str(k) for k in k_values)
    return export_roi_plots(
        exporters,
        comparison_dir,
        label_of=lambda exporter: str(exporter.ctx.run_ctx.sampler.k),
        label_order=k_order,
        box_filename=k_roi_boxplot_filename(alpha),
        violin_filename=k_roi_violin_filename(alpha),
        xlabel="K",
        series_prefix="K",
        nonsyn_w=nonsyn_w,
    )

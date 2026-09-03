from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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


def alpha_roi_boxplot_filename(k: int) -> str:
    return f"sweep_alpha_K{k}_alpha_roi_boxwhisker.png"


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


def roi_points_by_alpha(
    exporters: dict[str, ParetoExporter],
    *,
    nonsyn_w: float | None = None,
) -> tuple[str | None, dict[str, list[RoiPoint]]]:
    roi_by_alpha: dict[str, list[RoiPoint]] = defaultdict(list)
    seq_id: str | None = None
    for exporter in exporters.values():
        if not exporter._results:
            continue
        seq_id = exporter.ctx.run_ctx.target_sequence_id
        alpha_label = _alpha_label_from_exporter(exporter)
        regions = regions_for_results(exporter._results, nonsyn_w=nonsyn_w)
        for result in exporter._results:
            if regions[result.id] == SolutionRegion.ROI:
                roi_by_alpha[alpha_label].append((result.cost, result.binding_score))
    return seq_id, dict(roi_by_alpha)


def render_alpha_roi_boxplot(
    seq_id: str,
    roi_by_alpha: dict[str, list[RoiPoint]],
    output_path: Path,
) -> Path | None:
    if not roi_by_alpha:
        return None

    labels = sorted(roi_by_alpha, key=_alpha_label_sort_key)
    cost_groups = [[cost for cost, _binding in roi_by_alpha[label]] for label in labels]
    binding_groups = [
        [binding for _cost, binding in roi_by_alpha[label]] for label in labels
    ]

    fig, (ax_cost, ax_binding) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    colors = plt.cm.tab10.colors

    for ax, groups, ylabel in (
        (ax_cost, cost_groups, "Functional cost"),
        (ax_binding, binding_groups, "Binding score"),
    ):
        box = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=True,
            widths=0.6,
        )
        for patch, color in zip(box["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(f"ROI distributions — {seq_id}", y=1.02, fontsize=11)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_alpha_sweep_roi_boxplot(
    exporters: dict[str, ParetoExporter],
    comparison_dir: Path,
    *,
    k: int,
    nonsyn_w: float | None = None,
) -> Path | None:
    seq_id, roi_by_alpha = roi_points_by_alpha(exporters, nonsyn_w=nonsyn_w)
    if not roi_by_alpha or seq_id is None:
        return None
    output_path = comparison_dir / alpha_roi_boxplot_filename(k)
    return render_alpha_roi_boxplot(seq_id, roi_by_alpha, output_path)

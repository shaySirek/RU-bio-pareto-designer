from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns

from pareto_designer.models.context import RunContext, ParetoResult


def _sorted_by_cost(
    costs: np.ndarray, bindings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(costs)
    return costs[order], bindings[order]


def _frontier_legend_handles(frontiers: dict[str, np.ndarray]) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=f"C{idx % 10}",
            linewidth=1.5,
            label=key,
        )
        for idx, key in enumerate(frontiers.keys())
    ]


def _filter_frontier_by_binding(
    frontier: np.ndarray, min_binding: float = 0.0
) -> np.ndarray:
    if frontier.size == 0:
        return frontier
    return frontier[frontier[:, 1] >= min_binding]


def render_pareto_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path,
    max_cost: float,
    binding_range: tuple[float, float],
    hit_thresholds: list[float] | None = None,
    *,
    origin_frontiers: dict[str, np.ndarray] | None = None,
    db_fsm_labels: set[str] | None = None,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted_frontiers: dict[str, np.ndarray] = {}
    plotted_costs: list[float] = []
    plotted_bindings: list[float] = []
    for idx, (key, frontier) in enumerate(frontiers.items()):
        color = f"C{idx % 10}"
        filtered = _filter_frontier_by_binding(frontier)
        if filtered.size == 0:
            continue
        plotted_frontiers[key] = filtered
        costs, bindings = _sorted_by_cost(filtered[:, 0], filtered[:, 1])
        plotted_costs.extend(costs.tolist())
        plotted_bindings.extend(bindings.tolist())
        ax.plot(costs, bindings, color=color, linewidth=1.5, label=key)

        if origin_frontiers is not None and key in origin_frontiers:
            if db_fsm_labels and key in db_fsm_labels:
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
            )

    plot_max_cost = max(plotted_costs) if plotted_costs else max_cost
    if plotted_bindings:
        plot_binding_range = (min(plotted_bindings), max(plotted_bindings))
    else:
        plot_binding_range = binding_range

    _draw_hit_thresholds(ax, hit_thresholds)
    _set_pareto_axes(ax, plot_max_cost, plot_binding_range, hit_thresholds)
    if plotted_frontiers:
        ax.legend(
            handles=_frontier_legend_handles(plotted_frontiers),
            loc="upper right",
        )

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
    hit_thresholds: list[float] | None = None,
    *,
    is_db_fsm: bool = True,
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
    line_color = "C0"
    fsm_label = "Binding score" if is_db_fsm else "Reduced FSM"
    ax.plot(costs, bindings, color=line_color, linewidth=1.5, label=fsm_label)

    if not is_db_fsm:
        ax.plot(
            costs,
            origin_bindings,
            color=line_color,
            linewidth=1.5,
            linestyle="--",
            label="Origin (DB FSM)",
        )

    _draw_hit_thresholds(ax, hit_thresholds)
    _set_pareto_axes(ax, plot_max_cost, plot_binding_range, hit_thresholds)
    ax.legend(loc="upper right")

    fig.savefig(
        ctx.output_path / "pareto_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _draw_hit_thresholds(ax: Axes, hit_thresholds: list[float] | None):
    if not hit_thresholds:
        return
    for n_hits, threshold in enumerate(hit_thresholds, start=1):
        ax.axhline(threshold, linestyle="--", color="gray", linewidth=0.8, zorder=1)
        ax.text(
            0.99,
            threshold,
            f"{n_hits} hit" if n_hits == 1 else f"{n_hits} hits",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=8,
            color="gray",
        )


def _set_pareto_axes(
    ax: Axes,
    max_cost: float,
    binding_range: tuple[float, float],
    hit_thresholds: list[float] | None = None,
):
    x_max = max_cost * 1.05
    min_binding, max_binding = binding_range
    if hit_thresholds:
        min_binding = min(min_binding, *hit_thresholds)
        max_binding = max(max_binding, *hit_thresholds)
    y_margin = (max_binding - min_binding) * 0.05
    if y_margin == 0:
        y_margin = 1.0
    y_min = min_binding - y_margin
    y_max = max_binding + y_margin
    ax.set_xlabel("Functional Cost")
    ax.set_ylabel("Binding Score")
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


def render_kmer_binding_score_mse_histogram(
    ctx: RunContext,
    results: list[ParetoResult],
    fsm_mse: float | None = None,
):
    values = [
        r.kmer_binding_score_mse
        for r in results
        if np.isfinite(r.kmer_binding_score_mse)
    ]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(values, bins=min(50, max(5, len(values))), color="steelblue", alpha=0.8)
    mean = float(np.mean(values))
    ax.axvline(
        mean, linestyle="-", color="black", linewidth=1, label=f"mean={mean:.4g}"
    )
    if fsm_mse is not None and np.isfinite(fsm_mse):
        ax.axvline(
            fsm_mse,
            linestyle="--",
            color="black",
            linewidth=1,
            label=f"FSM MSE={fsm_mse:.4g}",
        )
    ax.set_xlabel("K-mer binding score MSE (per solution)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(
        ctx.output_path / "kmer_binding_score_mse_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("FSM Binding Score")
    fig.savefig(
        ctx.output_path / "binding_scores_scatter.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

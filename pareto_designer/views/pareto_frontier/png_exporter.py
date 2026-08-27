from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

from pareto_designer.models.context import RunContext, ParetoResult

_HIT_LEVELS = {
    "0": (0, "#cbd5e0"),
    "1": (1, "#d8b4fe"),
    "2-5": (5, "#a855f7"),
    "6+": (float("inf"), "#6b21a8"),
}
_FRONTIER_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "h", "p"]


def _hit_color(n_motif_hits: int) -> str:
    for _, (thr, col) in _HIT_LEVELS.items():
        if n_motif_hits <= thr:
            return col
    return "#6b21a8"


def _hit_legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=col,
            edgecolor="black",
            linewidth=0.5,
            label=lbl,
        )
        for lbl, (_, col) in _HIT_LEVELS.items()
    ]


def _frontier_legend_handles(frontiers: dict[str, np.ndarray]) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            marker=_FRONTIER_MARKERS[idx % len(_FRONTIER_MARKERS)],
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            linestyle="None",
            label=key,
        )
        for idx, key in enumerate(frontiers.keys())
    ]


def render_pareto_frontiers(
    frontiers: dict[str, np.ndarray],
    output_file: Path,
    max_cost: float,
    binding_range: tuple[float, float],
    hit_thresholds: list[float] | None = None,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, (key, frontier) in enumerate(frontiers.items()):
        marker = _FRONTIER_MARKERS[idx % len(_FRONTIER_MARKERS)]
        costs = frontier[:, 0]
        bindings = frontier[:, 1]
        hits = frontier[:, 2].astype(int)
        ax.scatter(
            costs,
            bindings,
            c=[_hit_color(h) for h in hits],
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
            s=15,
        )

    _draw_hit_thresholds(ax, hit_thresholds)
    _set_pareto_axes(ax, max_cost, binding_range, hit_thresholds)
    ax.legend(
        handles=_frontier_legend_handles(frontiers),
        title="Frontiers",
        loc="upper right",
    )
    fig.subplots_adjust(bottom=0.18)
    fig.legend(
        handles=_hit_legend_handles(),
        title="Motif Hits",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=len(_HIT_LEVELS),
        frameon=True,
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
):
    min_hits = min(r.n_motif_hits for r in results)
    buckets = {key: [] for key in _HIT_LEVELS}
    star = None
    for r in results:
        for lbl, (thr, col) in _HIT_LEVELS.items():
            if r.n_motif_hits <= thr:
                buckets[lbl].append(r)
                if star is None and r.n_motif_hits == min_hits:
                    star = (r, col)
                break

    groups = {key: (buckets[key], _HIT_LEVELS[key][1]) for key in _HIT_LEVELS}
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, (group_results, color) in groups.items():
        if not group_results:
            continue
        costs = [r.cost for r in group_results]
        bindings = [r.binding_score for r in group_results]
        yerr_lower = [
            max(0.0, r.binding_score - r.origin_binding_score) for r in group_results
        ]
        yerr_upper = [
            max(0.0, r.origin_binding_score - r.binding_score) for r in group_results
        ]
        ax.errorbar(
            costs,
            bindings,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="red",
            elinewidth=0.8,
            capsize=2,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            costs,
            bindings,
            label=label,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            s=30,
        )
    if star:
        ax.scatter(
            star[0].cost,
            star[0].binding_score,
            color="#ffd700",
            marker="*",
            s=90,
            edgecolors=star[1],
            linewidths=1.0,
            zorder=5,
        )

    _draw_hit_thresholds(ax, hit_thresholds)
    _set_pareto_axes(ax, max_cost, binding_range, hit_thresholds)
    ax.legend(title="Motif Hits", loc="upper right")

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
            0.01,
            threshold,
            f"{n_hits} hit" if n_hits == 1 else f"{n_hits} hits",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
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


def render_motif_cost_dist(frontiers: dict[str, np.ndarray], output_file: Path):
    fig, ax = plt.subplots(figsize=(5, 4))

    all_hits = set()
    for frontier in frontiers.values():
        all_hits.update(frontier[:, 2].astype(int))

    sorted_hits = np.sort(list(all_hits))
    x_positions = np.arange(len(sorted_hits))
    legend_elements = []

    for idx, (key, frontier) in enumerate(frontiers.items()):
        costs = frontier[:, 0]
        motif_hits = frontier[:, 2].astype(int)
        current_color = f"C{idx % 10}"

        grouped_data = []
        positions_to_plot = []

        for pos, hit in zip(x_positions, sorted_hits):
            mask = motif_hits == hit
            if np.any(mask):
                sub_costs = costs[mask]
                offset = (idx - (len(frontiers) - 1) / 2) * 0.15
                actual_pos = pos + offset

                grouped_data.append(sub_costs)
                positions_to_plot.append(actual_pos)

                jitter = np.random.uniform(-0.02, 0.02, size=len(sub_costs))
                ax.scatter(
                    np.repeat(actual_pos, len(sub_costs)) + jitter,
                    sub_costs,
                    color=current_color,
                    alpha=0.5,
                    s=12,
                    edgecolors="#2c3e50",
                    linewidths=0.5,
                    zorder=3,
                )

        bp = ax.boxplot(
            grouped_data,
            positions=positions_to_plot,
            widths=0.12,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )

        for box in bp["boxes"]:
            box.set_facecolor(current_color)
            box.set_alpha(0.4)
            box.set_edgecolor("#2c3e50")

        for whisker in bp["whiskers"]:
            whisker.set(color="#7f8c8d", linestyle="-", alpha=0.5)
        for cap in bp["caps"]:
            cap.set(color="#7f8c8d", alpha=0.5)
        for median in bp["medians"]:
            median.set(color="#2c3e50", linewidth=1.5)

        legend_elements.append(
            mpatches.Patch(
                facecolor=current_color,
                alpha=0.6,
                edgecolor="#2c3e50",
                label=key,
            )
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(h) for h in sorted_hits])
    ax.set_xlabel("# Motif Hits")
    ax.set_ylabel("Functional Cost")
    ax.legend(handles=legend_elements, loc="upper right")

    fig.savefig(
        output_file,
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

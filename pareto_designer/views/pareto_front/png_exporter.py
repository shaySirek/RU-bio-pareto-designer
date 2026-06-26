import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns

from pareto_designer.models.context import RunContext, ParetoResult


def _get_front_label(key: str) -> str | None:
    return f"K={m.group(1)}" if (m := re.search(r"(?:^|__)k_([^\s_]+)", key)) else None


def render_pareto_fronts(
    fronts: dict[str, np.ndarray],
    output_file: Path,
    max_cost: float,
    binding_range: tuple[float, float],
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for key, front in fronts.items():
        ax.scatter(front[:, 0], front[:, 1], label=_get_front_label(key), alpha=0.8)

    _set_pareto_axes(ax, max_cost, binding_range)
    ax.legend(loc="upper right")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def render_pareto_front_png(
    ctx: RunContext,
    results: list[ParetoResult],
    max_cost: float,
    binding_range: tuple[float, float],
):
    min_hits = min(r.n_motif_hits for r in results)
    levels = {
        "0": (0, "#cbd5e0"),
        "1": (1, "#d8b4fe"),
        "2-5": (5, "#a855f7"),
        "6+": (float("inf"), "#6b21a8"),
    }
    buckets = {key: [] for key in levels}
    star = None
    for r in results:
        for lbl, (thr, col) in levels.items():
            if r.n_motif_hits <= thr:
                buckets[lbl].append(r)
                if star is None and r.n_motif_hits == min_hits:
                    star = (r, col)
                break

    groups = {key: (buckets[key], levels[key][1]) for key in levels}
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, (group_results, color) in groups.items():
        ax.scatter(
            [r.cost for r in group_results],
            [r.binding_score for r in group_results],
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

    _set_pareto_axes(ax, max_cost, binding_range)
    ax.legend(title="Motif Hits", loc="upper right")

    fig.savefig(
        ctx.output_path / "pareto_front.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _set_pareto_axes(ax: Axes, max_cost: float, binding_range: tuple[float, float]):
    x_max = max_cost * 1.05
    min_binding, max_binding = binding_range
    y_margin = (max_binding - min_binding) * 0.05
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


def render_motif_cost_dist(fronts: dict[str, np.ndarray], output_file: Path):
    fig, ax = plt.subplots(figsize=(5, 4))

    all_hits = set()
    for front in fronts.values():
        all_hits.update(front[:, 2].astype(int))

    sorted_hits = np.sort(list(all_hits))
    x_positions = np.arange(len(sorted_hits))
    legend_elements = []

    for idx, (key, front) in enumerate(fronts.items()):
        costs = front[:, 0]
        motif_hits = front[:, 2].astype(int)
        current_color = f"C{idx % 10}"

        grouped_data = []
        positions_to_plot = []

        for pos, hit in zip(x_positions, sorted_hits):
            mask = motif_hits == hit
            if np.any(mask):
                sub_costs = costs[mask]
                offset = (idx - (len(fronts) - 1) / 2) * 0.15
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
                label=_get_front_label(key),
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

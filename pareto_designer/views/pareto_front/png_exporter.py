import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

from pareto_designer.models.context import RunContext, ParetoResult


def render_pareto_front_png(ctx: RunContext, results: list[ParetoResult]):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"0": "#cbd5e0", "1": "#d8b4fe", "2-5": "#a855f7", "6+": "#6b21a8"}
    buckets = {key: [] for key in colors}
    for r in results:
        hits = r.n_motif_hits
        if hits == 0:
            buckets["0"].append(r)
        elif hits == 1:
            buckets["1"].append(r)
        elif 2 <= hits <= 5:
            buckets["2-5"].append(r)
        else:
            buckets["6+"].append(r)

    groups = {key: (buckets[key], colors[key]) for key in colors}

    for label, (group_results, color) in groups.items():
        if not group_results:
            continue
        ax.scatter(
            [r.cost for r in group_results],
            [r.binding_score for r in group_results],
            label=label,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            s=60,
        )

    ax.set_xlabel("Functional Cost")
    ax.set_ylabel("Binding Score")
    ax.legend(title="Motif Hits", loc="upper right")

    fig.savefig(
        ctx.output_path / "pareto_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def render_heatmap_png(ctx: RunContext, res: ParetoResult, max_cost: float):
    data = res.costs.reshape(1, -1)
    norm = mcolors.SymLogNorm(
        linthresh=1.0, linscale=1.0, vmin=0, vmax=max_cost, base=10
    )

    width = min(max(10, len(res.costs) * 0.02), 40)
    fig, ax = plt.subplots(figsize=(width, 0.8))

    sns.heatmap(
        data,
        cmap="Reds",
        norm=norm,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    for start, end in ctx.orfs:
        ax.axvspan(start - 0.5, end - 0.5, color="blue", alpha=0.1, zorder=0)
        ax.plot(
            [start - 0.5, end - 0.5],
            [-0.05, -0.05],
            color="blue",
            lw=3,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    for start, end in res.motif_hits:
        ax.axvspan(start - 0.5, end - 0.5, color="blue", alpha=0.1, zorder=0)
        ax.plot(
            [start - 0.5, end - 0.5],
            [1.05, 1.05],
            color="darkblue",
            lw=3,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    legend_elements = [
        Line2D([0], [0], color="blue", lw=3, label="ORF", alpha=1.0),
        Line2D([0], [0], color="darkblue", lw=3, label="Motif Hits", alpha=1.0),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.8),
        ncol=2,
        frameon=False,
        fontsize="small",
    )

    ax.set_axis_off()

    fig.savefig(
        ctx.output_path / f"{res.id}_heatmap.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.close(fig)

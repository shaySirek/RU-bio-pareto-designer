import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from pareto_designer.models.context import RunContext, ParetoResult


def render_pareto_front_png(ctx: RunContext, results: list[ParetoResult]):
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = {
        "0": ([r for r in results if r.n_motif_hits == 0], "#cbd5e0"),
        "1": ([r for r in results if r.n_motif_hits == 1], "#d8b4fe"),
        "2-5": ([r for r in results if 2 <= r.n_motif_hits <= 5], "#a855f7"),
        "5+": ([r for r in results if r.n_motif_hits > 5], "#6b21a8"),
    }

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
            [1.05, 1.05],
            color="blue",
            lw=3,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        ax.plot(
            [start - 0.5, end - 0.5],
            [-0.05, -0.05],
            color="blue",
            lw=3,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
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

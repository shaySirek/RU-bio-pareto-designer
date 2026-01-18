from typing import Callable
from operator import itemgetter
import os

import warnings
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpld3
from scipy.stats import pearsonr

from pareto_designer.shared.fsm_utils.coloring import PartitioningMethod
from pareto_designer.bio_fetcher.motif import BindingMotif, StrandForBindingScore

warnings.filterwarnings("ignore")


def plot_histogram(
    matrix_id: str,
    method: PartitioningMethod,
    short_desc: str,
    scores: np.ndarray,
    approx_scores: np.ndarray,
    errors: dict[str, float],
    complexity_display: str,
    bins: int = 50,
    out_folder: str = "plots",
):
    filename = os.path.join(out_folder, method.value, f"{matrix_id}_{short_desc}.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(scores, bins=bins, alpha=0.6, color="steelblue", label="Origin")
    ax.hist(
        approx_scores, bins=bins, alpha=0.6, color="darkorange", label="Irreducible"
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()

    error_str = ", ".join([f"{k}={v:.3g}" for k, v in errors.items()])
    fig.suptitle(
        f"{matrix_id} | {short_desc}" f"\n{error_str}\n{complexity_display}",
        fontsize=10,
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_elbow_and_hist(
    matrix_id: str,
    method: PartitioningMethod,
    scores: np.ndarray,
    series_n_colors: np.ndarray,
    series_coloring_mse: np.ndarray,
    series_binding_mse: np.ndarray,
    series_n_states_irreducible: np.ndarray,
    hist_bins: int = 50,
    out_folder: str = "plots",
    description: str = "",
):
    filename = os.path.join(out_folder, method.value, f"{matrix_id}.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, (ax_hist, ax_elbow) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f"{matrix_id} | {method.value}\n{description}", fontsize=14)

    ax_hist.hist(scores, bins=hist_bins, alpha=0.6, color="steelblue")
    ax_hist.set_xlabel("Binding score")
    ax_hist.set_ylabel("Frequency")

    ax_elbow.plot(
        series_n_colors,
        series_coloring_mse,
        marker="o",
        label="Coloring MSE = SSE / |V(G)|",
        color="tab:red",
        linestyle="--",
    )
    ax_elbow.plot(
        series_n_colors,
        series_binding_mse,
        marker="s",
        label="Binding score MSE",
        color="tab:red",
    )
    ax_elbow.set_xlabel("Number of Colors")
    ax_elbow.set_xticks(series_n_colors)
    ax_elbow.set_ylabel("MSE", color="tab:red")
    ax_elbow.tick_params(axis="y", labelcolor="tab:red")

    ax_elbow_twin = ax_elbow.twinx()
    ax_elbow_twin.plot(
        series_n_colors,
        series_n_states_irreducible,
        marker="^",
        label="|V(G')|",
        color="tab:green",
    )
    ax_elbow_twin.set_ylabel("Number of States in Irreducible FSM", color="tab:green")
    ax_elbow_twin.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax_elbow.get_legend_handles_labels()
    lines2, labels2 = ax_elbow_twin.get_legend_handles_labels()
    ax_elbow.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_colored_state_reduction(
    approx_binding_score_map: dict[str, float],
    inverse_states_mapping: dict[str, list[str]],
    color_map: dict[str, str],
    filename: str,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    unique_colors = sorted(list(set(color_map.values())))
    num_colors = len(unique_colors)
    cmap = cm.get_cmap("viridis", num_colors)
    gradient_colors = [cmap(i) for i in range(num_colors)]
    fsm_color_to_color = {c: gradient_colors[i] for i, c in enumerate(unique_colors)}

    all_scores: list[float] = []
    all_sizes: list[int] = []
    all_colors_for_plot: list[tuple] = []
    for v, score in approx_binding_score_map.items():
        gradient_color = fsm_color_to_color[color_map[v]]
        merged_set_size = len(inverse_states_mapping[v])
        all_scores.append(score)
        all_sizes.append(merged_set_size)
        all_colors_for_plot.append(gradient_color)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(all_scores, all_sizes, c=all_colors_for_plot, alpha=0.7)

    ax.set_title("State Reduction", fontsize=14)
    ax.set_xlabel("Approximate Binding Score")
    ax.set_ylabel("Size of Merged Sets")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scores_scatter(
    binding_score_map: dict[str, float],
    approx_binding_score_map: dict[str, float],
    states_mapping: dict[str, str],
    inverse_states_mapping: dict[str, list[str]],
    color_map: dict[str, str],
    potential_merging_sets: list[list[str]],
    filename: str,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    sorted_states = list(
        map(
            itemgetter(0),
            sorted(list(approx_binding_score_map.items()), key=itemgetter(1)),
        )
    )
    n_states_in_reduced_fsm = len(sorted_states)
    states_rank = {v_in_reduced: i for i, v_in_reduced in enumerate(sorted_states)}
    states_cmap = cm.get_cmap(
        "gray", n_states_in_reduced_fsm + n_states_in_reduced_fsm // 4
    )
    states_colors = [states_cmap(i) for i in range(n_states_in_reduced_fsm)]
    fsm_coloring_cmap = cm.get_cmap("viridis", len(set(color_map.values())))

    pms_by_first: dict[str, list[str]] = {pms[0]: pms for pms in potential_merging_sets}

    scores_in_origin: list[float] = []
    scores_in_reduced: list[float] = []
    colors_for_scatter = []
    labels: list[str] = []
    merged_states_lines = []
    pms_lines = []
    for v_in_reduced, v_set in inverse_states_mapping.items():
        idx = states_rank[v_in_reduced]
        score_in_reduced = approx_binding_score_map[v_in_reduced]
        color_for_plot = fsm_coloring_cmap(int(color_map[v_in_reduced]))
        for v in v_set:
            score = binding_score_map[v]
            scores_in_origin.append(score)
            scores_in_reduced.append(score_in_reduced)
            colors_for_scatter.append(color_for_plot)
            labels.append(
                f"State in Origin FSM: {v} ({score:.2f})<br>"
                f"State in Reduced FSM: {v_in_reduced} ({score_in_reduced:.2f})<br>"
                f"{idx+1}/{n_states_in_reduced_fsm}"
            )

            if v in pms_by_first:
                pms = pms_by_first[v]
                v_min = min(pms, key=binding_score_map.get)
                v_max = max(pms, key=binding_score_map.get)
                pms_lines.append(
                    (
                        [binding_score_map[v_min], binding_score_map[v_max]],
                        [
                            approx_binding_score_map[states_mapping[v_min]],
                            approx_binding_score_map[states_mapping[v_max]],
                        ],
                    )
                )

        if len(v_set) > 1:
            group_scores = scores_in_origin[-len(v_set) :]
            merged_states_lines.append(
                (
                    idx,
                    [min(group_scores), max(group_scores)],
                    [score_in_reduced, score_in_reduced],
                )
            )

    # min_score = min(scores_in_origin)
    # max_score = max(scores_in_origin)

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(
        scores_in_origin, scores_in_reduced, c=colors_for_scatter, alpha=0.7
    )
    for idx, x, y in merged_states_lines:
        ax.plot(x, y, color=states_colors[idx], linestyle="--", linewidth=1)
    for x, y in pms_lines:
        ax.plot(x, y, color="red", linestyle="--", linewidth=0.5)

    # ax.set_xlim(min_score, max_score)
    # ax.set_ylim(min_score, max_score)

    fig.tight_layout()
    fig.savefig(filename.replace(".html", ".png"), dpi=300, bbox_inches="tight")

    tooltip = mpld3.plugins.PointHTMLTooltip(
        scatter,
        labels=labels,
        voffset=10,
        hoffset=10,
        css="""
                                             background-color: white;
                                             border: 1px solid black;
                                             border-radius: 5px;
                                             padding: 5px;
                                             font-family: sans-serif;
                                             font-size: 12px;
                                             """,
    )
    mpld3.plugins.connect(fig, tooltip)
    mpld3.save_html(fig, filename)

    plt.close(fig)


def plot_state_reduction_process_cmp_colored(
    motif_ctx: BindingMotif,
    mse_by_n_states: dict[int, float],
    colored_fsm_by_n_colors: dict[int, tuple[int, float]],
    out_folder: str,
):
    sorted_colorless_fsm_info = sorted(mse_by_n_states.items(), key=itemgetter(0))
    series_colorless_n_states = np.array(
        list(map(itemgetter(0), sorted_colorless_fsm_info))
    )
    series_colorless_mse_series = np.array(
        list(map(itemgetter(1), sorted_colorless_fsm_info))
    )

    filename = os.path.join(
        out_folder, f"{motif_ctx.matrix_id}_state_reduction_cmp.png"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        series_colorless_n_states,
        series_colorless_mse_series,
        label="colorless",
        color="black",
    )
    for n_colors, (n_states, mse) in colored_fsm_by_n_colors.items():
        ax.scatter(n_states, mse, s=40, edgecolor="black", label=f"{n_colors} colors")
    ax.set_xlabel("# States")
    ax.set_ylabel("MSE")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    filename = os.path.join(out_folder, f"{motif_ctx.matrix_id}_state_reduction.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    pow2_mask = (series_colorless_n_states & (series_colorless_n_states - 1)) == 0
    mask = pow2_mask & (
        series_colorless_n_states >= (1 << motif_ctx.length)
    )  # >= sqrt(4^m) = 2^m
    ax.scatter(
        series_colorless_n_states[mask],
        series_colorless_mse_series[mask],
        color="black",
    )
    ax.set_xlabel("# States")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("MSE")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_motifs_reductions_scatter(
    stats_per_motif_len: dict[int, list[dict[StrandForBindingScore, float]]],
    out_folder: str,
):
    for single_strand in (
        StrandForBindingScore.Forward,
        StrandForBindingScore.Backward,
    ):
        filename = os.path.join(
            out_folder,
            f"{single_strand.value}_strand_double_strand_reduction_efficiencies_scatter.png",
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for motif_length, stats in sorted(
            stats_per_motif_len.items(), key=itemgetter(0)
        ):
            ax.scatter(
                [strands_effs[single_strand] for strands_effs in stats],
                [strands_effs[StrandForBindingScore.Double] for strands_effs in stats],
                label=f"m={motif_length}",
                alpha=0.7,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{single_strand.value} strand reduction efficiency")
        ax.set_ylabel("double-stranded reduction efficiency")
        ax.legend(loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    for agg_name, agg_func in (
        ("avg", lambda x: sum(x) / len(x)),
        ("max", max),
    ):
        filename = os.path.join(
            out_folder,
            f"{agg_name}_single_stranded_double_strand_reduction_efficiencies_scatter.png",
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for motif_length, stats in sorted(
            stats_per_motif_len.items(), key=itemgetter(0)
        ):
            ax.scatter(
                [
                    agg_func(
                        (
                            strands_effs[StrandForBindingScore.Forward],
                            strands_effs[StrandForBindingScore.Backward],
                        )
                    )
                    for strands_effs in stats
                ],
                [strands_effs[StrandForBindingScore.Double] for strands_effs in stats],
                label=f"m={motif_length}",
                alpha=0.7,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{agg_name} single-stranded reduction efficiency")
        ax.set_ylabel("double-stranded reduction efficiency")
        ax.legend(loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_state_reduction_processes_across_strands(
    motif_ctx: BindingMotif,
    strand_mse_by_n_states: dict[StrandForBindingScore, dict[int, float]],
    reduction_efficiency_fn: Callable[[dict[int, float]], float],
    out_folder: str,
) -> dict[StrandForBindingScore, float]:
    filename = os.path.join(out_folder, f"{motif_ctx.matrix_id}_state_reduction.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    strands_effs: dict[StrandForBindingScore, float] = {}
    for strand_for_score, mse_by_n_states in strand_mse_by_n_states.items():
        sorted_colorless_fsm_info = sorted(mse_by_n_states.items(), key=itemgetter(0))
        series_colorless_n_states = np.array(
            list(map(itemgetter(0), sorted_colorless_fsm_info))
        )
        series_colorless_mse_series = np.array(
            list(map(itemgetter(1), sorted_colorless_fsm_info))
        )
        eff = reduction_efficiency_fn(mse_by_n_states)
        strands_effs[strand_for_score] = eff
        ax.plot(
            series_colorless_n_states,
            series_colorless_mse_series,
            label=strand_for_score.get_label(eff),
            color=strand_for_score.get_color(),
        )

    ax.set_xlabel("# States")
    ax.set_ylabel("MSE")
    ax.legend(prop={"size": 14, "family": "DejaVu Sans Mono"})

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return strands_effs


def plot_mse_hist(
    colored_mse_series: np.ndarray,
    delta_mse_series: np.ndarray,
    out_folder: str = "plots",
):
    filename = os.path.join(out_folder, "mse_hist.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].hist(colored_mse_series, bins=50, color="steelblue", alpha=0.7)
    axes[0].axvline(colored_mse_series.mean(), color="red", ls="--")
    axes[0].set_title("MSE(colored)")

    axes[1].hist(delta_mse_series, bins=50, color="orange", alpha=0.7)
    axes[1].axvline(delta_mse_series.mean(), color="red", ls="--")
    axes[1].set_title("ΔMSE")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_motifs_scatter(
    stats_per_motif_len: dict[int, list[tuple[BindingMotif, float]]],
    out_folder: str = "plots",
):
    filename = os.path.join(out_folder, "avg_ic_reduction_efficiency_scatter.html")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    average_ics = []
    reduction_efficiencies = []
    all_labels = []
    all_targets = []
    for motif_length, stats in sorted(stats_per_motif_len.items(), key=itemgetter(0)):
        current_average_ics = []
        current_reduction_efficiencies = []
        for motif_ctx, reduction_efficiency in stats:
            average_ic = motif_ctx.avg_ic
            current_average_ics.append(average_ic)
            current_reduction_efficiencies.append(reduction_efficiency)
            all_labels.append(
                f"{motif_ctx} ({average_ic:.3f},{reduction_efficiency:.4f})"
            )
            all_targets.append(
                f"https://jaspar.elixir.no/matrix/{motif_ctx.matrix_id}/"
            )

        ax.scatter(
            current_average_ics,
            current_reduction_efficiencies,
            label=f"m={motif_length}",
            alpha=0.7,
        )
        average_ics.extend(current_average_ics)
        reduction_efficiencies.extend(current_reduction_efficiencies)

    ax.set_xlabel("average IC")
    ax.set_ylabel("reduction efficiency")
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename.replace(".html", ".png"), dpi=300, bbox_inches="tight")

    current_index = 0
    for collection in ax.collections:
        num_points = len(collection.get_offsets())
        labels_chunk = all_labels[current_index : current_index + num_points]
        targets_chunk = all_targets[current_index : current_index + num_points]
        tooltip = mpld3.plugins.PointHTMLTooltip(
            collection,
            labels=labels_chunk,
            targets=targets_chunk,
            voffset=10,
            hoffset=10,
            css="""
                background-color: white;
                border: 1px solid black;
                border-radius: 5px;
                padding: 5px;
                font-family: sans-serif;
                font-size: 12px;
                """,
        )
        mpld3.plugins.connect(fig, tooltip)
        current_index += num_points

    mpld3.save_html(fig, filename)
    plt.close(fig)

    corr, p = pearsonr(average_ics, reduction_efficiencies)
    logger.info(
        f"Correlation between average IC and reduction efficiency: r={corr:.2f}, p-value={p:.3f}"
    )

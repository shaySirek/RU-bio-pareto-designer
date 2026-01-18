from collections import Counter
from operator import itemgetter
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

from pareto_designer.algorithms.fsm import FSM, T_STATE, T_CHAR
from pareto_designer.bio_fetcher.motif import BindingMotif

DNA_BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "green", "C": "blue", "G": "orange", "T": "red"}


def _compute_position_freqs(sequences: list[str]) -> np.ndarray:
    """Compute base frequencies per position."""
    if not sequences:
        return np.zeros((4, 1))

    seq_len = len(sequences[0])
    freqs = np.zeros((4, seq_len), dtype=float)
    for pos in range(seq_len):
        col = [s[pos] for s in sequences]
        counts = Counter(col)
        total = sum(counts.get(b, 0) for b in DNA_BASES)
        for i, b in enumerate(DNA_BASES):
            freqs[i, pos] = counts.get(b, 0) / total if total > 0 else 0.0
    return freqs


def _plot_sequence_logo(ax, sequences: list[str]):
    """Plot histogram-style DNA sequence logo on given axis."""
    if not sequences:
        ax.axis("off")
        return

    freqs = _compute_position_freqs(sequences)
    seq_len = freqs.shape[1]
    x = np.arange(seq_len)
    bottom = np.zeros(seq_len)

    for i, base in enumerate(DNA_BASES):
        ax.bar(
            x,
            freqs[i],
            bottom=bottom,
            color=BASE_COLORS[base],
            width=0.8,
        )
        bottom += freqs[i]

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def visualise_fsm(
    motif_ctx: BindingMotif,
    binding_score_map: dict[T_STATE, float],
    reduced_fsm: FSM[T_STATE, T_CHAR],
    mse: float,
    reduced_fsm_f_inverse: dict[T_STATE, list[T_STATE]],
    out_folder: str = "plots",
):
    """
    - Violins plot of conditional distribution of size and MSE across states in the reduced FSM.
    - Visualise states in the reduced FSM as stacked sequence logos.
      - along each state v, we note its size on the left, and plot its SSE as bar on the right.
    """
    n_states = len(reduced_fsm.V)
    filename = os.path.join(
        out_folder, f"{motif_ctx.matrix_id}_reduced_fsm_{n_states}_states.png"
    )
    violin_filename = os.path.join(
        out_folder,
        f"{motif_ctx.matrix_id}_reduced_fsm_{n_states}_states__violin_size_vs_mse.png",
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = []
    for v in reduced_fsm.V:
        origin_states = reduced_fsm_f_inverse[v]
        origin_scores = [binding_score_map[w] for w in origin_states]
        partial_sse = np.var(origin_scores) * len(origin_scores)
        data.append((v, origin_states, partial_sse))
    data = sorted(data, key=lambda item: len(item[1]), reverse=True)
    state_size_series = np.array(list(map(lambda x: len(x[1]), data)))
    state_sse_series = np.array(list(map(itemgetter(2), data)))
    state_mse_series = state_sse_series / state_size_series

    # Violins plot of n(v) and MSE(v)
    n_states_origin = len(binding_score_map)
    avg_state_size = n_states_origin / n_states
    bins = np.array([1, avg_state_size // 4, avg_state_size // 2])
    bins = np.append(
        bins, avg_state_size * (2 ** np.arange(0, min(3, np.log2(n_states)) + 1))
    )
    labels = [
        rf"${int(bins[i])}\!-\!{int(bins[i+1]) - 1}$" for i in range(len(bins) - 1)
    ]

    df = pd.DataFrame({"state_size": state_size_series, "mse": state_mse_series})
    df["size_bin"] = pd.cut(
        state_size_series, bins=bins, right=False, include_lowest=True
    )
    df = df.dropna(subset=["size_bin"])
    max_mse = df["mse"].max()
    print(
        f"Reduced FSM with {n_states} states of motif {motif_ctx.matrix_id} has maximum per-state MSE of {max_mse:.3f}"
    )

    violin_fig, ax = plt.subplots(figsize=(5, 4))
    sns.violinplot(
        data=df, x="size_bin", y="mse", scale="width", inner="quartile", cut=0, ax=ax
    )
    ax.set_xticklabels(labels, rotation=30)
    y_max = 22.0
    ax.set_ylim(0.0, y_max)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_xlabel("n(v)")
    ax.set_ylabel("MSE(v)")

    bin_counts = df["size_bin"].value_counts().sort_index()
    for i, count in enumerate(bin_counts):
        ax.text(i, 1.05 * y_max, f"({count})", ha="center", va="bottom", fontsize=10)

    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    violin_fig.savefig(violin_filename, dpi=300, bbox_inches="tight")
    plt.close(violin_fig)

    # FSM states visualisation
    sse_min, sse_max = state_sse_series.min(), state_sse_series.max()
    sse_range = sse_max - sse_min if sse_max != sse_min else 1.0

    fig = plt.figure(figsize=(8, max(4, 0.3 * n_states)))
    gs = gridspec.GridSpec(
        n_states,
        2,
        width_ratios=[4, 1],
        wspace=0.05,
        hspace=0.1,
    )

    # For each state v in the reduced FSM,
    # plot sequence logo of the represented states in the origin FSM,
    # and plot bar for its SSE, SSE(v).
    for i, (v, origin_states, partial_sse) in enumerate(data):
        # Sequence logo
        ax_logo = fig.add_subplot(gs[i, 0])
        _plot_sequence_logo(ax_logo, origin_states)
        ax_logo.set_ylabel(
            f"{len(origin_states)}", rotation=0, labelpad=10, fontsize=8, va="center"
        )

        # SSE bar
        ax_bar = fig.add_subplot(gs[i, 1])
        ax_bar.barh(
            [0],
            [partial_sse],
            color="gray",
            height=0.5,
            alpha=0.7,
        )
        ax_bar.set_ylabel(
            f"{partial_sse:.1f}", rotation=0, labelpad=-20, fontsize=8, va="center"
        )
        ax_bar.set_xlim(sse_min - 0.05 * sse_range, sse_max + 0.05 * sse_range)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        ax_bar.set_frame_on(False)

        if i == 0:
            ax_logo.set_title("n(v)", fontsize=8, loc="left")
            ax_bar.set_title("SSE(v)", fontsize=8)

    # Legend for bases
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE_COLORS[b]) for b in DNA_BASES]
    fig.legend(handles, DNA_BASES, loc="lower center", ncol=4, fontsize=8)

    fig.suptitle(f"MSE={mse:.3f}")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

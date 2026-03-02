import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from pareto_designer.shared.parsing import read_sequence
from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.bio_fetcher.motif import BindingMotif

FUNC_CMAP = "Reds"
FUNC_RANGE = (0.0, 1.0)
BINDING_CMAP = "Purples"


class SequencePlotData(NamedTuple):
    sequence: str
    costs: np.ndarray | None
    binding_data: np.ndarray
    hits: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Pareto-optimal solutions for multiple result folders."
    )
    parser.add_argument("parent_dir", type=Path)
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["k_50__alpha_0.7", "k_150__alpha_0.7", "k_250__alpha_0.7"],
    )
    parser.add_argument("--matrix-id", "-m", type=str, default="MA0267.1")
    parser.add_argument("--hit-threshold", type=float, default=4.8)
    parser.add_argument(
        "--codon-usage",
        type=Path,
        default=Path("bio_data/codon_usage/saccharomyces_cerevisiae.txt"),
    )
    parser.add_argument("-alpha", type=float, default=0.5)
    parser.add_argument("-beta", type=float, default=1.0)
    parser.add_argument("-w", type=float, default=500.0)

    return parser.parse_args()


def get_global_bounds(
    results_list: list[list[dict]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    all_costs = [r["cost"] for sublist in results_list for r in sublist]
    all_scores = [r["binding_score"] for sublist in results_list for r in sublist]
    if not all_costs:
        return (0, 500), (-23500, -16000)
    c_max, s_min, s_max = max(all_costs), min(all_scores), max(all_scores)
    return (0, int(round(c_max * 1.05, -1))), (
        int(round(s_min - (s_max - s_min) * 0.05, -1)),
        int(round(s_max + (s_max - s_min) * 0.05, -1)),
    )


def calc_seq_data(
    args: argparse.Namespace,
    target_path: Path,
    solutions: dict[str, Path],
) -> tuple[dict[str, SequencePlotData], list[tuple[int, int]]]:
    score_function = (
        ScoreFunctionBuilder()
        .with_codon_usage(args.codon_usage)
        .with_params(alpha=args.alpha, beta=args.beta, w=args.w)
        .with_target_sequence(target_path)
        .build()
    )
    motif = BindingMotif(args.matrix_id)
    binding_map = motif.get_binding_score_map()
    seq_data = {}

    for sid, path in solutions.items():
        seq = read_sequence(path)
        seq_data[sid] = SequencePlotData(
            sequence=seq,
            costs=np.array(score_function.get_costs(seq), dtype=float),
            **get_binding_data(seq, binding_map, motif.length, args.hit_threshold),
        )

    target_seq = score_function.target_sequence
    seq_data["target"] = SequencePlotData(
        sequence=target_seq,
        costs=None,
        **get_binding_data(target_seq, binding_map, motif.length, args.hit_threshold),
    )
    return seq_data, score_function.orfs


def get_binding_data(
    seq: str,
    binding_score_map: dict[str, float],
    m: int,
    hit_threshold: float,
):
    data = np.full(len(seq), np.nan)
    hits = []
    for i in range(len(seq) - m + 1):
        data[i] = binding_score_map[seq[i : i + m]]
        if data[i] > hit_threshold:
            hits.append((i, i + m - 1))
    return dict(binding_data=data, hits=hits)


def plot_solutions(
    results: list[dict],
    fig_file: Path,
    x_lim: tuple[int, int],
    y_lim: tuple[int, int],
) -> dict:
    min_hits = min(len(r["motif_hits"]) for r in results)
    configs = (
        {
            "0": (0, "#cbd5e0"),
            "1": (1, "#d8b4fe"),
            "2-5": (5, "#a855f7"),
            "6+": (float("inf"), "#6b21a8"),
        }
        if min_hits <= 3
        else {
            "0": (0, "#cbd5e0"),
            "1": (1, "#d8b4fe"),
            "2-5": (5, "#a855f7"),
            "6-10": (10, "#6b21a8"),
            "11-20": (20, "#3b1b6b"),
            "21+": (float("inf"), "#1b0f30"),
        }
    )
    buckets = {label: [] for label in configs}
    star = None
    for r in results:
        hits = len(r["motif_hits"])
        for lbl, (thr, col) in configs.items():
            if hits <= thr:
                buckets[lbl].append((r["cost"], r["binding_score"]))
                if star is None and hits == min_hits:
                    star = (r, col)
                break

    fig, ax = plt.subplots(figsize=(9, 4))
    for lbl, pts in buckets.items():
        if not pts:
            continue
        x, y = zip(*pts)
        ax.scatter(
            x,
            y,
            label=lbl,
            c=configs[lbl][1],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            s=60,
            zorder=2,
        )
    if star:
        ax.scatter(
            star[0]["cost"],
            star[0]["binding_score"],
            color="#ffd700",
            marker="*",
            s=400,
            edgecolors=star[1],
            linewidths=1.0,
            zorder=5,
        )

    ax.set_xlabel("Functional Cost")
    ax.set_ylabel("Binding Score")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend(title="Motif Hits")

    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return star[0]


def plot_heapmap(
    orfs: list[tuple[int, int]],
    data: SequencePlotData,
    binding_score_heapmap_range: tuple[float, float],
    fig_file: Path,
) -> None:
    width = min(max(10, len(data.binding_data) * 0.02), 40)

    if data.costs is not None:
        fig, axes = plt.subplots(2, 1, figsize=(width, 2.0), sharex=True)
        fig.subplots_adjust(hspace=0.1)
        ax_cost, ax_binding = axes[0], axes[1]
        sns.heatmap(
            data.costs.reshape(1, -1),
            cmap=FUNC_CMAP,
            norm=mcolors.Normalize(*FUNC_RANGE),
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax_cost,
        )
    else:
        fig, axes = plt.subplots(1, 1, figsize=(width, 1.0), sharex=True)
        ax_binding = axes
        for s, e in orfs:
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
        data.binding_data.reshape(1, -1),
        cmap=BINDING_CMAP,
        norm=mcolors.Normalize(*binding_score_heapmap_range),
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax_binding,
    )
    hit_mask = np.full(len(data.binding_data), np.nan)
    for start, end in data.hits:
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

    seq_len = len(data.binding_data)
    ticks = np.arange(0, seq_len, 100)
    ax_binding.set_xticks(ticks + 0.5)
    ax_binding.set_xticklabels(ticks, fontsize=12)

    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Heatmap rendered: {fig_file}")


def export_colorbars(bars: dict[str, tuple[str, tuple[float, float]]], out_file: Path):
    n = len(bars)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 0.5), squeeze=False)

    i = 0
    for name, (cmap_name, norm_range) in bars.items():
        ax = axes[0, i]
        norm = mcolors.Normalize(vmin=norm_range[0], vmax=norm_range[1])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_name),
            cax=ax,
            orientation="horizontal",
        )
        ax.set_title(name, fontsize=10, pad=5)
        i += 1

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Heatmap legend rendered: {out_file}")


def main() -> None:
    args = parse_args()
    root = Path(args.parent_dir)
    target = Path(f"bio_data/zea_mays_genes/{root.parts[1]}.txt")
    sets, dirs = [], []
    for fld in args.folders:
        if (p := root / fld / "results_metadata.json").exists():
            with p.open() as f:
                sets.append(json.load(f)["results"])
                dirs.append(root / fld)
    if not sets:
        return

    x_lim, y_lim = get_global_bounds(sets)
    solution_files: dict[str, Path] = {}
    with ThreadPoolExecutor() as exe:
        futs = [
            (
                exe.submit(
                    plot_solutions,
                    results,
                    folder / "pareto_optimal_solutions.png",
                    x_lim,
                    y_lim,
                ),
                folder,
            )
            for results, folder in zip(sets, dirs)
        ]
        for future, folder in futs:
            star = future.result()
            sid = star["id"]
            solution_files[sid] = folder / f"{sid}_sequence.txt"

    seq_data, orfs = calc_seq_data(args, target, solution_files)
    binding_score_heapmap_range: tuple[float, float] = (
        -args.hit_threshold,
        0.95 * args.hit_threshold,
    )
    export_colorbars(
        {
            "Functional cost": (FUNC_CMAP, FUNC_RANGE),
            "Binding score": (BINDING_CMAP, binding_score_heapmap_range),
        },
        root / "heatmap_legend.png",
    )
    for sid, data in seq_data.items():
        out = (
            root / "target_sequence_binding.png"
            if sid == "target"
            else solution_files[sid].with_suffix(".png")
        )
        logger.info(f"{len(data.hits)} hits in {sid}")
        plot_heapmap(orfs, data, binding_score_heapmap_range, out)


if __name__ == "__main__":
    main()

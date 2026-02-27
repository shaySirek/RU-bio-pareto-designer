import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    parser.add_argument("--hit-threshold", type=float, default=4.7)
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
    args: argparse.Namespace, target_path: Path, solutions: dict[str, Path]
) -> tuple[dict[str, tuple[np.ndarray | None, np.ndarray]], list[tuple[int, int]]]:
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
        seq_data[sid] = (
            np.array(score_function.get_costs(seq), dtype=float),
            get_binding_data(seq, binding_map, motif.length),
        )
    seq_data["target"] = (
        None,
        get_binding_data(score_function.target_sequence, binding_map, motif.length),
    )
    return seq_data, score_function.orfs


def get_binding_data(
    seq: str, binding_score_map: dict[str, float], m: int
) -> np.ndarray:
    data = np.full(len(seq), np.nan)
    for i in range(len(seq) - m + 1):
        data[i] = binding_score_map[seq[i : i + m]]
    return data


def plot_solutions(
    results: list[dict], fig_file: Path, x_lim: tuple[int, int], y_lim: tuple[int, int]
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
    costs: np.ndarray | None,
    binding_data: np.ndarray,
    motif_hits: list[tuple[int, int]],
    fig_file: Path,
) -> None:
    width = min(max(10, len(binding_data) * 0.02), 40)
    fig, axes = plt.subplots(
        2 if costs is not None else 1,
        1,
        figsize=(width, 2.0 if costs is not None else 1.0),
        sharex=True,
    )
    if costs is not None:
        fig.subplots_adjust(hspace=0.1)
    ax_cost, ax_binding = (axes[0], axes[1]) if costs is not None else (None, axes)

    if costs is not None:
        sns.heatmap(
            costs.reshape(1, -1),
            cmap="Reds",
            norm=mcolors.SymLogNorm(1.0, 1.0, 0, np.nanmax(costs), 10),
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax_cost,
        )

    sns.heatmap(
        binding_data.reshape(1, -1),
        cmap="YlGnBu",
        norm=mcolors.Normalize(
            0, np.nanmax(binding_data) if not np.all(np.isnan(binding_data)) else 1
        ),
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax_binding,
    )

    hit_mask = np.full(len(binding_data), np.nan)
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

    axs = [ax_cost, ax_binding] if costs is not None else [ax_binding]

    for s, e in orfs:
        for ax in axs:
            ax.axvspan(s - 0.5, e - 0.5, color="blue", alpha=0.1, zorder=0)
        ax_binding.plot(
            [s - 0.5, e - 0.5],
            [-0.1, -0.1],
            color="blue",
            lw=4,
            transform=ax_binding.get_xaxis_transform(),
            clip_on=False,
        )

    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"Heatmap rendered: {fig_file}")


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
    solution_hits: dict[str, list[tuple[int, int]]] = {}
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
            solution_hits[sid] = star["motif_hits"]

    seq_data, orfs = calc_seq_data(args, target, solution_files)
    for sid, (costs, binding_data) in seq_data.items():
        if sid == "target":
            out = root / "target_sequence_binding.png"
            hits = []
            for i, score in enumerate(binding_data):
                if score > args.hit_threshold:
                    hits.append((i, i+6))
        else:
            out = solution_files[sid].with_suffix(".png")
            hits = solution_hits[sid]
        
        logger.info(f"{len(hits)} hits in {sid}")
        plot_heapmap(orfs, costs, binding_data, hits, out)


if __name__ == "__main__":
    main()

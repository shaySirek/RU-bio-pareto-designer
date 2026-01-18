from typing import Iterator, Any
from collections import defaultdict
import argparse
import time
import yaml
from pathlib import Path

from loguru import logger
import pandas as pd
import numpy as np

from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.shared.fsm_utils.fsm_factory import (
    get_binding_motif_fsm,
    get_colored_fsm,
)
from pareto_designer.shared.plot import (
    plot_mse_hist,
    plot_state_reduction_process_cmp_colored,
    plot_motifs_scatter,
)
from pareto_designer.shared.csv_writer import write_results_stream
from pareto_designer.shared.fsm_utils.coloring import PartitioningMethod
from pareto_designer.algorithms.fsm_reduction.colored_db_fsm_reducer import (
    Colored_DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm_reduction.util import get_reduction_efficiency


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("experiments") / "base.yml",
        help="Configuration file",
    )
    parser.add_argument("--colorless", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    return parser.parse_args()


def read_config(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def colored_run_all(
    matrix_id_list: list[str],
    n_colors_list: list[int],
    coloring_method: PartitioningMethod,
) -> Iterator[dict[str, Any]]:
    logger.info(f"Running experiments on {len(matrix_id_list)} binding motifs...")
    for matrix_id in matrix_id_list:
        # motif with different number of colors
        yield from colored_run_on_motif(
            matrix_id, False, n_colors_list, coloring_method
        )
        # reverse complement of the motif with different number of colors
        # yield from colored_run_on_motif(matrix_id, True, n_colors_list, coloring_method)


def colored_run_on_motif(
    matrix_id: str,
    reverse_complement: bool,
    n_colors_list: list[int],
    coloring_method: PartitioningMethod,
) -> Iterator[dict[str, Any]]:
    logger.info(f"\nRunning {len(n_colors_list)} experiments for motif {matrix_id}...")
    motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(
        matrix_id, reverse_complement
    )
    matrix_id = motif_ctx.matrix_id
    n_states_db_fsm = len(db_fsm.V)

    logger.info(f"IC: {motif_ctx.ic_vector}")
    motif_ic_info = {
        f"motif_ic_{i+1}": round(motif_ctx.get_ic_at(i), 3)
        for i in range(min(motif_ctx.length, 5))
    }
    motif_ic_info.update(
        {
            "motif_ic_mean": round(motif_ctx.mean, 3),
            "motif_ic_std": round(motif_ctx.std, 3),
        }
    )

    for n_colors in n_colors_list:
        try:
            logger.info(100 * "-")
            colored_fsm = get_colored_fsm(
                db_fsm, binding_score_map, n_colors, coloring_method
            )
            logger.info(
                f"Finding irreducible FSM for colored DB FSM [matrix_id={matrix_id}, n_colors={n_colors}]..."
            )
            start: float = time.perf_counter()

            fsm_reducer = Colored_DB_FSM_Reducer[str, str, str](colored_fsm, matrix_id)
            irreducible_fsm = fsm_reducer.find_irreducible_fsm()
            duration: float = round(time.perf_counter() - start, 3)

            fsm_reducer.validate()
            n_states_irreducible_fsm = len(irreducible_fsm.V)
            efficiency_gain_perc = round(
                100 * (1 - (n_states_irreducible_fsm / n_states_db_fsm)), 3
            )
            binding_mse = fsm_reducer.with_binding_score_map(
                binding_score_map
            ).get_binding_score_mse()

            yield {
                "matrix_id": matrix_id,
                "motif_length": motif_ctx.length,
                "n_states_db_fsm": n_states_db_fsm,
                "number_of_colors": n_colors,
                "n_states_irreducible_fsm": n_states_irreducible_fsm,
                "efficiency_gain_perc": efficiency_gain_perc,
                "reducing_fsm_duration_sec": duration,
                "binding_mse": binding_mse,
                **motif_ic_info,
            }
        except Exception:
            logger.info(f"[WARN] Skipped: matrix_id={matrix_id}, n_colors={n_colors}")


def fsms_iter_as_dict_iter(fsms_iter):
    for i, (reduced_fsm, mse, _) in enumerate(fsms_iter):
        yield {
            "step": i + 1,
            "n_states": len(reduced_fsm.V),
            "binding_mse": mse,
        }


def get_colorless_results_file(out_folder: Path, m: int, matrix_id: str):
    return out_folder / "colorless" / str(m) / f"{matrix_id}_state_reduction.csv"


def plot_results(matrix_id_list: list[str], colored_results_file: Path):
    out_folder: Path = colored_results_file.parent
    stats_per_ncolors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    stats_per_motif_len: dict[int, list[tuple[BindingMotif, float, float]]] = (
        defaultdict(list)
    )

    # collect results and plot curve (state reduction: number of states vs MSE) for each motif
    colored_results_df = pd.read_csv(colored_results_file)
    for matrix_id, motif_colored_results in colored_results_df.groupby("matrix_id"):
        if matrix_id not in matrix_id_list:
            continue
        motif_length = motif_colored_results["motif_length"].iloc[0]
        motif_ctx = BindingMotif(matrix_id)

        mse_by_n_states: dict[int, float] = {}
        colored_fsm_by_n_colors: dict[int, tuple[int, float]] = {}

        colorless_results_file = get_colorless_results_file(
            out_folder, motif_length, matrix_id
        )
        colorless_reduced_fsms = pd.read_csv(colorless_results_file)
        for colored_entry in motif_colored_results.to_dict("records"):
            n_colors = colored_entry["number_of_colors"]
            n_states_irreducible_fsm = colored_entry["n_states_irreducible_fsm"]
            mse_colored = colored_entry["binding_mse"]
            mse_colorless = colorless_reduced_fsms.loc[
                colorless_reduced_fsms["n_states"] == n_states_irreducible_fsm,
                "binding_mse",
            ].item()

            # print outlier cases
            if mse_colored < mse_colorless:
                logger.info(
                    f"\nOutlier for motif {matrix_id} of length {motif_length} with c={n_colors}:"
                    f"\t|V'|={n_states_irreducible_fsm}"
                    f"\n\tMSE(colored irreducible)={mse_colored:.2f}"
                    f"\n\tMSE(colorless reduced)={mse_colorless:.2f}"
                )

            stats_per_ncolors[n_colors].append((mse_colored, mse_colorless))
            colored_fsm_by_n_colors[n_colors] = (n_states_irreducible_fsm, mse_colored)

        for colorless_entry in colorless_reduced_fsms.to_dict("records"):
            mse_by_n_states[colorless_entry["n_states"]] = colorless_entry[
                "binding_mse"
            ]
        reduction_efficiency = get_reduction_efficiency(mse_by_n_states)
        plot_state_reduction_process_cmp_colored(
            motif_ctx,
            mse_by_n_states,
            colored_fsm_by_n_colors,
            str(out_folder / "colorless" / str(motif_length)),
        )
        stats_per_motif_len[motif_length].append((motif_ctx, reduction_efficiency))
        logger.info(
            f"{matrix_id} (m={motif_length}):"
            f"\n\tIC:                                                                                     {motif_ctx.mean:.2f}"
            f"\n\t\t{motif_ctx.ic_vector}"
            f"\n\tStd. deviation:                                                                         {motif_ctx.std:.2f}"
            f"\n\t\t{motif_ctx.var_vector}"
            f"\n\treduction efficiency:                                                                   {reduction_efficiency:.3f}"
        )

    # print stats per number of colors
    for n_colors, mse_pairs in stats_per_ncolors.items():
        colored_mse_series = np.array([mse_colored for mse_colored, _ in mse_pairs])
        delta_mse_series = np.array(
            [mse_colored - mse_colorless for mse_colored, mse_colorless in mse_pairs]
        )
        colored_mse_mean, colored_mse_std = (
            colored_mse_series.mean(),
            colored_mse_series.std(),
        )
        delta_mse_mean, delta_mse_std = delta_mse_series.mean(), delta_mse_series.std()
        logger.info(
            f"\n\nc={n_colors}"
            f"\t\tMSE(colored)={colored_mse_mean:.2f} ± {colored_mse_std:.2f}"
            f"\t\tΔMSE={delta_mse_mean:.2f} ± {delta_mse_std:.2f}"
        )

    # histogram
    colored_mse_series = np.array(
        [
            mse_colored
            for mse_pairs in stats_per_ncolors.values()
            for mse_colored, _ in mse_pairs
        ]
    )
    delta_mse_series = np.array(
        [
            mse_colored - mse_colorless
            for mse_pairs in stats_per_ncolors.values()
            for mse_colored, mse_colorless in mse_pairs
        ]
    )
    plot_mse_hist(colored_mse_series, delta_mse_series, str(out_folder / "colorless"))

    # scatter
    plot_motifs_scatter(stats_per_motif_len, str(out_folder / "colorless"))


def main():
    args = parse_args()
    config_file: Path = args.config
    out_folder: Path = config_file.parent
    colored_results_file: Path = out_folder / f"{config_file.stem}.csv"

    logger.info(f"Reading configuration of '{config_file.name}' in {out_folder}...")
    config: dict[str, Any] = read_config(config_file)
    coloring_method: PartitioningMethod = PartitioningMethod(
        config.get("coloring_method")
    )
    n_colors_list: list[int] = config.get("number_of_colors", [])
    matrix_id_list: list[str] = config.get("motifs", [])
    logger.info(f"\nConfiguration:\n" f"\t{len(matrix_id_list)} different motifs")

    if args.visualize:
        logger.info(
            f"\nReading results from {colored_results_file}, printing stats, plotting figures..."
        )
        plot_results(matrix_id_list, colored_results_file)
    elif args.colorless:
        for matrix_id in matrix_id_list:
            motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(matrix_id)
            fsm_reducer = DB_FSM_Reducer[str, str](
                db_fsm, binding_score_map, motif_ctx.matrix_id
            )
            fsms_iter = fsm_reducer.find_reduced_fsms()
            generator = fsms_iter_as_dict_iter(fsms_iter)
            colorless_results_file = get_colorless_results_file(
                out_folder, motif_ctx.length, motif_ctx.matrix_id
            )
            write_results_stream(generator, colorless_results_file)
    else:
        logger.info(
            f"\t{len(n_colors_list)} different coloring levels:\n\t{n_colors_list}\n"
        )
        generator = colored_run_all(matrix_id_list, n_colors_list, coloring_method)
        write_results_stream(generator, colored_results_file)

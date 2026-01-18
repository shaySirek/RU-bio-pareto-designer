import argparse
import time
import os

from loguru import logger
import numpy as np

from pareto_designer.shared.fsm_utils.coloring import (
    partition_to_colors,
    PartitioningMethod,
)
from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.algorithms.fsm_reduction.colored_db_fsm_reducer import (
    Colored_DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm_reduction.util import get_reduction_efficiency
from pareto_designer.algorithms.fsm import FSM, ColoredFSM
from pareto_designer.shared.plot import (
    plot_histogram,
    plot_elbow_and_hist,
    plot_state_reduction_process_cmp_colored,
)
from pareto_designer.shared.fsm_utils.fsm_vis import visualise_fsm


def parse_args():
    parser = argparse.ArgumentParser()

    # motif Matrix ID
    parser.add_argument("matrix_id", type=str, help="Binding motif matrix id")

    # params for algorithm with coloring
    parser.add_argument(
        "--n-colors-list",
        "-k",
        type=str,
        default=",".join(map(str, [3, 6, 9, 15])),
        help="comma separated list of number of colors",
    )
    parser.add_argument(
        "--coloring-method",
        "-cm",
        type=str,
        default=PartitioningMethod.KMEANS_1D.value,
        choices=[x.value for x in list(PartitioningMethod)],
        help="Coloring method",
    )

    # validation flag
    parser.add_argument("--validate", default=False, action="store_true")

    # motif modification options
    parser.add_argument(
        "--override-consensus-prefix",
        type=str,
        default=None,
        help="Modify motif with consensus prefix",
    )
    parser.add_argument(
        "--reverse-complement", "-rc", default=False, action="store_true"
    )

    # debug and outputs
    parser.add_argument("--print-states", "-v", default=False, action="store_true")
    parser.add_argument(
        "--out-folder", "-o", type=str, default="plots", help="Output folder"
    )
    parser.add_argument("--plot-hist", "-ph", default=False, action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # counts_override = None
    motif_kwargs = {
        "reverse_complement": args.reverse_complement,
    }
    # if counts_override:
    #     motif_kwargs["counts_override"] = counts_override
    if args.override_consensus_prefix:
        motif_kwargs["consensus_override"] = {
            i: letter for i, letter in enumerate(args.override_consensus_prefix)
        }

    motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(
        args.matrix_id, **motif_kwargs
    )
    out_folder = os.path.join(args.out_folder, str(motif_ctx.length))
    sqrt_n_states = np.sqrt(len(db_fsm.V))

    n_colors_list = list(map(int, args.n_colors_list.split(",")))
    coloring_method = PartitioningMethod(args.coloring_method)
    colored_fsm_by_n_colors: dict[int, tuple[int, float]] = {}

    mse_by_n_states, min_n_states_by_mse = run_colorless_reduction(
        motif_ctx,
        db_fsm,
        binding_score_map,
        args.validate,
        {sqrt_n_states / (2**i) for i in range(motif_ctx.length)},
        os.path.join(out_folder, "colorless"),
    )
    colored_stats = list(
        run_reduction_with_coloring(
            motif_ctx,
            db_fsm,
            binding_score_map,
            n_colors_list,
            coloring_method,
            args.validate,
            args.print_states,
            out_folder,
            args.plot_hist,
        )
    )

    logger.info("\nSummary\n" + 100 * "=")
    for n_colors, n_states_irreducible_fsm, mse in colored_stats:
        colored_fsm_by_n_colors[n_colors] = (n_states_irreducible_fsm, mse)
        # by same size
        same_size_reduced_fsm_mse = mse_by_n_states[n_states_irreducible_fsm]
        # if there is a smaller reduced FSM whose MSE is equal to
        # the MSE of the reduced FSM of the same size as the colored irreducible FSM,
        # we want to consider it when comparing the methods.
        smallest_reduced_fsm = min_n_states_by_mse[same_size_reduced_fsm_mse]
        # by closet mse
        closest_reduced_fsm_mse, closet_mse_n_states = min(
            min_n_states_by_mse.items(), key=lambda x: abs(x[0] - mse)
        )
        logger.info(
            f"\ncolored irreducible FSM with c={n_colors}\t--> |V'| = {n_states_irreducible_fsm}\tMSE = {mse:.6f}"
            f"\ncolorless reduced FSMs"
            f"\n\twith same number of states\t--> |V'| = {n_states_irreducible_fsm}\tMSE = {same_size_reduced_fsm_mse:.6f}"
            f"\t(can be reduced to |V'| = {smallest_reduced_fsm} with same MSE)"
            f"\n\twith the closet MSE\t\t--> |V'| = {closet_mse_n_states}\tMSE = {closest_reduced_fsm_mse:.6f}"
            f"\n{80 * '='}"
        )
    reduction_efficiency = get_reduction_efficiency(mse_by_n_states)
    plot_state_reduction_process_cmp_colored(
        motif_ctx,
        mse_by_n_states,
        colored_fsm_by_n_colors,
        os.path.join(out_folder, "colorless"),
    )
    logger.info(
        f"{motif_ctx.matrix_id} (m={motif_ctx.length}):"
        f"\n\tIC:                                                                                     {motif_ctx.mean:.2f}"
        f"\n\t\t{motif_ctx.ic_vector}"
        f"\n\tStd. deviation:                                                                         {motif_ctx.std:.2f}"
        f"\n\t\t{motif_ctx.var_vector}"
        f"\n\treduction efficiency:                                                                   {reduction_efficiency:.3f}"
    )


def run_reduction_with_coloring(
    motif_ctx: BindingMotif,
    db_fsm: FSM[str, str],
    binding_score_map: dict[str, float],
    n_colors_list: list[int],
    coloring_method: PartitioningMethod,
    validate: bool,
    print_states: bool,
    out_folder: str,
    plot_hist: bool,
):
    matrix_id = motif_ctx.matrix_id
    logger.info(f"Running {len(n_colors_list)} experiments for motif {matrix_id}...")
    description = (
        f"IC = {motif_ctx.ic_vector}"
        f" | mean = {motif_ctx.mean:.3f}"
        f" | std = {motif_ctx.std:.3f}"
    )

    origin_scores = np.array(list(binding_score_map.values()))
    n_states_db_fsm = len(db_fsm.V)

    series_n_colors: list[int] = []
    series_coloring_mse: list[float] = []
    series_binding_mse: list[float] = []
    series_n_states_irreducible: list[int] = []
    sse_dict: dict[int, float] = {}

    for n_colors in n_colors_list:
        try:
            logger.info(
                f"Coloring DB FSM by {coloring_method.value} clustering with k={n_colors}..."
            )
            colored_patterns, sse = partition_to_colors(
                binding_score_map, n_colors, coloring_method
            )
            coloring_mse = round(sse / n_states_db_fsm, 6)
            sse_dict[n_colors] = sse
            colored_fsm: ColoredFSM = ColoredFSM[str, str, str].from_coloring(
                db_fsm,
                colored_patterns,
            )

            logger.info(
                "\nFinding irreducible FSM for the obtained colored DB FSM"
                f"\n\tmatrix_id={matrix_id}"
                f"\n\tn_colors={n_colors}"
                f"\n\tcoloring_method={coloring_method.value}"
                f"\n{100*'*'}"
            )
            start = time.perf_counter()
            fsm_reducer = Colored_DB_FSM_Reducer[str, str, str](
                colored_fsm, f"{matrix_id}__{n_colors}"
            )
            irreducible_fsm = fsm_reducer.find_irreducible_fsm()
            end = time.perf_counter()
            reducing_fsm_duration = round(end - start, 3)

            if validate:
                fsm_reducer.validate()

            n_states_irreducible_fsm = len(irreducible_fsm.V)
            efficiency_gain_perc = round(
                100 * (1 - (n_states_irreducible_fsm / n_states_db_fsm)), 3
            )
            binding_mse = fsm_reducer.with_binding_score_map(
                binding_score_map
            ).get_binding_score_mse()

            if plot_hist:
                origin_scores, approx_scores = fsm_reducer.get_scores()
                plot_histogram(
                    matrix_id,
                    coloring_method,
                    str(n_colors),
                    origin_scores,
                    approx_scores,
                    {"MSE": binding_mse},
                    f"{n_states_irreducible_fsm} / {n_states_db_fsm} states (gain {efficiency_gain_perc:.3f}%)",
                    out_folder=out_folder,
                )

            if print_states:
                logger.info(f"Irreducible FSM:\n{str(fsm_reducer)}")

            logger.info(
                f"\nNumber of states in the irreducible FSM:    {n_states_irreducible_fsm}"
                f"\nEfficiency gain:                            {efficiency_gain_perc:.3f}%"
                f"\nBinding score MSE:                          {binding_mse:.6f}"
                f"\nDuration of finding irreducible FSM:        {reducing_fsm_duration:.3f}s"
                f"\n{100*'='}"
            )

            series_n_colors.append(n_colors)
            series_coloring_mse.append(coloring_mse)
            series_binding_mse.append(binding_mse)
            series_n_states_irreducible.append(n_states_irreducible_fsm)

            yield n_colors, n_states_irreducible_fsm, binding_mse

        except Exception:
            logger.warning(f"Skipped: matrix_id={matrix_id}, n_colors={n_colors}")

    if plot_hist:
        plot_elbow_and_hist(
            matrix_id,
            coloring_method,
            origin_scores,
            np.array(series_n_colors),
            np.array(series_coloring_mse),
            np.array(series_binding_mse),
            np.array(series_n_states_irreducible),
            out_folder=out_folder,
            description=description,
        )


def run_colorless_reduction(
    motif_ctx: BindingMotif,
    db_fsm: FSM[str, str],
    binding_score_map: dict[str, float],
    validate: bool,
    vis_at: set[int],
    out_folder: str,
):
    matrix_id = motif_ctx.matrix_id
    fsm_reducer = DB_FSM_Reducer[str, str](
        db_fsm, binding_score_map, matrix_id, validate=validate
    )
    fsms_iter = fsm_reducer.find_reduced_fsms()

    mse_by_n_states: dict[int, float] = {}
    min_n_states_by_mse: dict[float, int] = {}
    for reduced_fsm, mse, (_, _, reduced_fsm_f_inverse) in fsms_iter:
        n_states = len(reduced_fsm.V)
        mse_by_n_states[n_states] = mse
        min_n_states_by_mse[mse] = n_states
        if n_states in vis_at:
            visualise_fsm(
                motif_ctx,
                binding_score_map,
                reduced_fsm,
                mse,
                reduced_fsm_f_inverse,
                out_folder=out_folder,
            )

    return mse_by_n_states, min_n_states_by_mse

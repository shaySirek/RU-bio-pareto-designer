from typing import Optional
import argparse
import time
import os
import json

from loguru import logger

from pareto_designer.shared.fsm_utils.coloring import (
    partition_to_colors,
    PartitioningMethod,
    PartitioningMap,
)
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.algorithms.fsm_reduction.colored_db_fsm_reducer import (
    Colored_DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm import FSM, ColoredFSM
from pareto_designer.shared.fsm_utils.colored_fsm_compare import cmp_reductions
from pareto_designer.shared.plot import (
    plot_histogram,
    plot_colored_state_reduction,
    plot_scores_scatter,
)

FSMs_DIR = "FSMs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_id", type=str, help="Binding motif matrix id")
    parser.add_argument(
        "--target-n-colors",
        "-k",
        type=int,
        required=True,
        help="Target number of colors",
    )
    parser.add_argument(
        "--origin-n-colors", "-ok", type=int, help="Origin number of colors"
    )
    parser.add_argument(
        "--coloring-method",
        "-cm",
        type=str,
        default=PartitioningMethod.KMEANS_1D.value,
        choices=[x.value for x in list(PartitioningMethod)],
        help="Coloring method",
    )
    parser.add_argument(
        "--reverse-complement", "-rc", default=False, action="store_true"
    )
    parser.add_argument("--print-states", "-v", default=False, action="store_true")

    args = parser.parse_args()

    return args


def run(
    matrix_id: str,
    origin_n_colors: int,
    target_n_colors: int,
    coloring_method: PartitioningMethod,
    reverse_complement: bool,
    counts_override: Optional[dict[tuple[str, int], int]],
    print_states: bool,
):
    logger.info(
        "\nRunning hierarachical clustering experiment"
        f" [{origin_n_colors} colors -> {target_n_colors} colors]"
        f" for motif {matrix_id}..."
    )
    motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(
        matrix_id, reverse_complement, counts_override
    )
    matrix_id = motif_ctx.matrix_id

    cmp_file_path = os.path.join(
        FSMs_DIR,
        coloring_method.value,
        f"{matrix_id}_cmp_{origin_n_colors}---{target_n_colors}",
    )
    os.makedirs(os.path.dirname(cmp_file_path), exist_ok=True)

    partitioning_results = partition_to_colors(
        binding_score_map,
        target_n_colors,
        coloring_method,
        origin_n_colors=origin_n_colors,
        return_baseline=True,
        cmp_file_path=cmp_file_path,
    )

    if origin_n_colors != target_n_colors:
        (colored_patterns, _), (baseline_colored_patterns, _) = partitioning_results
        reducer = find_irreducible_fsm(
            matrix_id,
            db_fsm,
            binding_score_map,
            colored_patterns,
            coloring_method,
            f"{coloring_method.value} clustering with k={origin_n_colors} then transforming to k={target_n_colors}",
            f"{origin_n_colors}---{target_n_colors}",
            print_states,
        )
        baseline_reducer = find_irreducible_fsm(
            matrix_id,
            db_fsm,
            binding_score_map,
            baseline_colored_patterns,
            coloring_method,
            f"{coloring_method.value} clustering with k={target_n_colors}",
            str(target_n_colors),
            print_states,
        )

        write_mapping(
            reducer.inverse_states_mapping,
            coloring_method,
            f"{matrix_id}_{origin_n_colors}---{target_n_colors}.json",
        )
        write_mapping(
            baseline_reducer.inverse_states_mapping,
            coloring_method,
            f"{matrix_id}_{target_n_colors}.json",
        )

        cmp_reductions(
            reducer,
            baseline_reducer,
            f"{cmp_file_path}_fsm.json",
            reducer_label=f"{origin_n_colors}_{target_n_colors}",
            baseline_reducer_label=str(target_n_colors),
        )

    else:
        colored_patterns, _ = partitioning_results
        find_irreducible_fsm(
            matrix_id,
            db_fsm,
            binding_score_map,
            colored_patterns,
            coloring_method,
            f"{coloring_method.value} clustering with k={target_n_colors}",
            str(target_n_colors),
            print_states,
        )


def write_mapping(mapping: dict, coloring_method: PartitioningMethod, filename: str):
    filepath = os.path.join(FSMs_DIR, coloring_method.value, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wt") as f:
        json.dump(mapping, f, indent=4)


def find_irreducible_fsm(
    matrix_id: str,
    db_fsm: FSM[str, str],
    binding_score_map: dict[str, float],
    colored_patterns: PartitioningMap,
    coloring_method: PartitioningMethod,
    coloring_method_desc: str,
    short_desc: str,
    print_states: bool,
) -> Colored_DB_FSM_Reducer:
    n_states_db_fsm = len(db_fsm.V)

    logger.info(f"Coloring DB FSM by {coloring_method_desc}...")
    colored_fsm: ColoredFSM = ColoredFSM[str, str, str].from_coloring(
        db_fsm,
        colored_patterns,
    )

    logger.info("Finding irreducible FSM for the obtained colored DB FSM")
    start = time.perf_counter()
    fsm_reducer = Colored_DB_FSM_Reducer[str, str, str](colored_fsm, matrix_id)
    irreducible_fsm = fsm_reducer.find_irreducible_fsm()
    end = time.perf_counter()
    reducing_fsm_duration = round(end - start, 3)

    fsm_reducer.validate()
    n_states_irreducible_fsm = len(irreducible_fsm.V)
    efficiency_gain_perc = round(
        100 * (1 - (n_states_irreducible_fsm / n_states_db_fsm)), 3
    )
    binding_mse = fsm_reducer.with_binding_score_map(
        binding_score_map
    ).get_binding_score_mse()

    if print_states:
        filepath = os.path.join(
            FSMs_DIR, coloring_method.value, f"{matrix_id}_{short_desc}"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wt") as f:
            f.write(str(fsm_reducer))

    logger.info(
        f"\nNumber of states in the irreducible FSM:    {n_states_irreducible_fsm}"
        f"\nEfficiency gain:                            {efficiency_gain_perc:.3f}%"
        f"\nBinding score MSE:                          {binding_mse:.6f}"
        f"\nDuration of finding irreducible FSM:        {reducing_fsm_duration:.3f}s"
        f"\n{100*'='}"
    )

    approx_binding_score_map = fsm_reducer.get_approx_binding_score_map()
    potential_merging_sets = fsm_reducer.get_initial_potential_merging_sets()
    if len(potential_merging_sets) > 0:
        for pms in potential_merging_sets:
            s = ", ".join(
                map(
                    lambda v: f"{v} ({binding_score_map[v]:.3f}) [{colored_fsm.c(v)}]",
                    pms,
                )
            )
            logger.info(f"\n\t{s}")
        logger.info(
            f"There are {len(potential_merging_sets)} sets of states that 'could be merged'."
            "\n\tIn such set, there are states from the origin FSM "
            "that are represented by different states in the reduced FSM "
            "that have identical outgoing transitions (but different color) "
            "and the 'difference' in their binding scores is smaller than 1."
        )
    else:
        logger.info("There are no sets of states that 'could be merged' (eps=1.).")

    origin_scores, approx_scores = fsm_reducer.get_scores()
    plot_histogram(
        matrix_id,
        coloring_method,
        short_desc,
        origin_scores,
        approx_scores,
        {"MSE": binding_mse},
        f"{n_states_irreducible_fsm} / {n_states_db_fsm} states (gain {efficiency_gain_perc:.3f}%)",
        out_folder=FSMs_DIR,
    )

    plot_colored_state_reduction(
        approx_binding_score_map,
        fsm_reducer.inverse_states_mapping,
        irreducible_fsm._c,
        os.path.join(
            FSMs_DIR,
            coloring_method.value,
            f"{matrix_id}_{short_desc}_fsm_state_reduction_process.png",
        ),
    )

    plot_scores_scatter(
        binding_score_map,
        approx_binding_score_map,
        fsm_reducer.states_mapping,
        fsm_reducer.inverse_states_mapping,
        irreducible_fsm._c,
        potential_merging_sets,
        os.path.join(
            FSMs_DIR,
            coloring_method.value,
            f"{matrix_id}_{short_desc}_fsm_scores_scatter.html",
        ),
    )

    return fsm_reducer


def main():
    args = parse_args()
    coloring_method = PartitioningMethod(args.coloring_method)
    counts_override: Optional[dict[tuple[str, int], int]] = None
    ######### MA0166.1 #########
    # counts_override = {
    #     ("A", 0): 0,
    #     ("C", 0): 0,
    # }
    ############################

    run(
        args.matrix_id,
        args.origin_n_colors or args.target_n_colors,
        args.target_n_colors,
        coloring_method,
        args.reverse_complement,
        counts_override,
        args.print_states,
    )

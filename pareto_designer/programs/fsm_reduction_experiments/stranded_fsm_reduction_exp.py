from typing import Any
from collections import defaultdict
import argparse
import yaml
from pathlib import Path

from loguru import logger

from pareto_designer.models.motif import BindingMotif, StrandForBindingScore
from pareto_designer.shared.fsm_utils.fsm_factory import (
    get_binding_motif_fsm_all_strands,
)
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.algorithms.fsm_reduction.util import get_reduction_efficiency
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.shared.plot import (
    plot_state_reduction_processes_across_strands,
    plot_motifs_reductions_scatter,
    plot_motifs_reductions_hists,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("experiments") / "base.yml",
        help="Configuration file",
    )
    args = parser.parse_args()
    return args


def read_config(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def run_fsm_reductions_across_strands(
    motif_ctx: BindingMotif,
    db_fsm: FSM[str, str],
    binding_score_maps: dict[StrandForBindingScore, dict[str, float]],
) -> dict[StrandForBindingScore, dict[int, float]]:
    strand_mse_by_n_states: dict[StrandForBindingScore, dict[int, float]] = {}

    for strand_for_score, binding_score_map in binding_score_maps.items():
        logger.info(
            f"Running FSM state reduction with binding scores of motif {motif_ctx.matrix_id} [{strand_for_score.get_description()}]"
        )
        fsm_reducer = DB_FSM_Reducer[str, str](
            db_fsm, binding_score_map, motif_ctx.matrix_id
        )
        fsms_iter = fsm_reducer.find_reduced_fsms()

        strand_mse_by_n_states[strand_for_score] = {}
        for reduced_fsm, mse, _ in fsms_iter:
            n_states = len(reduced_fsm.V)
            strand_mse_by_n_states[strand_for_score][n_states] = mse

        reduction_efficiency = get_reduction_efficiency(
            strand_mse_by_n_states[strand_for_score]
        )
        logger.info(
            f"{motif_ctx.matrix_id} (m={motif_ctx.length}) [{strand_for_score.get_description()}]:"
            f"\n\tIC:                                                                                     {motif_ctx.mean:.2f}"
            f"\n\t\t{motif_ctx.ic_vector}"
            f"\n\tStd. deviation:                                                                         {motif_ctx.std:.2f}"
            f"\n\t\t{motif_ctx.var_vector}"
            f"\n\treduction efficiency:                                                                   {reduction_efficiency:.3f}"
        )

    return strand_mse_by_n_states


def main():
    args = parse_args()
    config_file: Path = args.config
    out_folder: Path = config_file.parent

    logger.info(f"Reading configuration of '{config_file.name}' in {out_folder}...")
    config: dict[str, Any] = read_config(config_file)
    matrix_id_list: list[str] = config.get("motifs", [])
    logger.info(f"\nConfiguration:\n" f"\t{len(matrix_id_list)} different motifs")

    stats_per_motif_len: dict[int, list[dict[StrandForBindingScore, float]]] = (
        defaultdict(list)
    )
    for matrix_id in matrix_id_list:
        motif_ctx, db_fsm, binding_score_maps = get_binding_motif_fsm_all_strands(
            matrix_id
        )
        strand_mse_by_n_states = run_fsm_reductions_across_strands(
            motif_ctx,
            db_fsm,
            binding_score_maps,
        )
        strands_effs = plot_state_reduction_processes_across_strands(
            motif_ctx,
            strand_mse_by_n_states,
            get_reduction_efficiency,
            str(out_folder / "stranded" / str(motif_ctx.length)),
        )
        stats_per_motif_len[motif_ctx.length].append(strands_effs)

    plot_motifs_reductions_scatter(
        stats_per_motif_len,
        str(out_folder / "stranded"),
    )

    plot_motifs_reductions_hists(
        stats_per_motif_len,
        str(Path("experiments") / "stranded"),
    )

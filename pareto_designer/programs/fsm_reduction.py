#!/usr/bin/env python3

import argparse
from pathlib import Path

from pareto_designer.models.motif import BindingMotif, StrandForBindingScore
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.fsm_reduction.colorless_db_fsm_reducer import (
    DB_FSM_Reducer,
)
from pareto_designer.shared.fsm_utils.fsm_factory import get_binding_motif_fsm
from pareto_designer.shared.csv_writer import write_results_stream

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_id", type=str, help="Binding motif matrix id")
    parser.add_argument(
        "--output-folder",
        "-o",
        type=Path,
        default=Path("FSMs") / "state_reductions",
        help="Output folder",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(
        args.matrix_id, StrandForBindingScore.Double
    )
    base_folder = Path(args.output_folder) / motif_ctx.matrix_id
    base_folder.mkdir(parents=True, exist_ok=True)
    reduced_fsms_file = base_folder / "trace.csv"
    plot_file = base_folder / "reduction_process.png"

    reduced_fsms_gen = fsm_reduction_gen(motif_ctx, db_fsm, binding_score_map)
    write_results_stream(reduced_fsms_gen, reduced_fsms_file)
    plot(reduced_fsms_file, plot_file)


def fsm_reduction_gen(
    motif_ctx: BindingMotif,
    db_fsm: FSM[str, str],
    binding_score_map: dict[str, float],
):
    db_fsm_size = len(db_fsm.V)
    fsm_reducer = DB_FSM_Reducer[str, str](
        db_fsm,
        binding_score_map,
        motif_ctx.matrix_id,
    )
    for i, (reduced_fsm, mse, _) in enumerate(fsm_reducer.find_reduced_fsms()):
        yield {
            "step": i + 1,
            "n_states": len(reduced_fsm.V),
            "sse": mse * db_fsm_size,
        }


def plot(reduced_fsms_file: Path, plot_file: Path):
    df = pd.read_csv(reduced_fsms_file)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["n_states"], df["sse"])
    ax.set_xlabel("# States")
    ax.set_ylabel("SSE")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

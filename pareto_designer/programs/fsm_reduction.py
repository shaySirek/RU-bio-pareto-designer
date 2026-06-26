#!/usr/bin/env python3

import argparse
from pathlib import Path

from pareto_designer.models.motif import BindingMotif, StrandForBindingScore
from pareto_designer.algorithms.fsm import FSM
from pareto_designer.algorithms.spaces import ScoreSpaceOption
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
        "--binding-score-space",
        type=str,
        choices=[x.value for x in ScoreSpaceOption],
        default=ScoreSpaceOption.LogExp.value,
        help="Space of binding scores",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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

    base_folder = Path(args.output_folder) / args.binding_score_space / args.matrix_id
    base_folder.mkdir(parents=True, exist_ok=True)
    reduced_fsms_file = base_folder / "trace.csv"
    plot_file = base_folder / "reduction_process.png"

    if not args.dry_run:
        motif_ctx, db_fsm, binding_score_map = get_binding_motif_fsm(
            args.matrix_id, StrandForBindingScore.Double
        )
        reduced_fsms_gen = fsm_reduction_gen(
            motif_ctx,
            db_fsm,
            binding_score_map,
            ScoreSpaceOption(args.binding_score_space),
            args.validate,
        )
        write_results_stream(reduced_fsms_gen, reduced_fsms_file)

    plot(reduced_fsms_file, plot_file)


def fsm_reduction_gen(
    motif_ctx: BindingMotif,
    db_fsm: FSM[str, str],
    binding_score_map: dict[str, float],
    binding_score_space_option: ScoreSpaceOption,
    validate: bool,
):
    fsm_reducer = DB_FSM_Reducer[str, str](
        db_fsm,
        binding_score_map,
        binding_score_space_option.get_space(),
        motif_ctx.matrix_id,
        validate=validate,
    )
    for i, (reduced_fsm, err, _) in enumerate(fsm_reducer.find_reduced_fsms()):
        yield {
            "step": i + 1,
            "n_states": len(reduced_fsm.V),
            "err": err,
        }


def plot(
    reduced_fsms_file: Path,
    plot_file: Path,
    part_at: int = 2048,
    min_size: int = 256,
):
    df = pd.read_csv(reduced_fsms_file)
    msk = df["n_states"] >= part_at
    df_plot(df[msk], plot_file.with_stem(f"{plot_file.stem}_beyond_{part_at}"))
    df_plot(
        df[~msk & (df["n_states"] >= min_size)],
        plot_file.with_stem(f"{plot_file.stem}_until_{part_at}"),
    )


def df_plot(df: pd.DataFrame, plot_file: Path):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(df["n_states"], df["err"])
    ax.set_xlabel("# States")
    ax.set_ylabel("SSE")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.algorithms.spaces import ScoreSpaceOption
from pareto_designer.shared.seq_design_utils.fsm_builder import FSMBuilder
from pareto_designer.shared.seq_design_utils.pareto_utils import render_and_compare
from pareto_designer.shared.seq_design_utils.run_grid import GridMode, run_design_grid
from pareto_designer.shared.csv_writer import write_results_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sequence",
        "-s",
        type=Path,
        required=True,
        help="Target sequence (fasta file or directory of *.txt files)",
    )
    parser.add_argument(
        "--matrix-id",
        "-m",
        type=str,
        required=True,
        help="Matrix ID of the binding motif",
    )
    parser.add_argument(
        "--binding-score-space",
        type=str,
        choices=[x.value for x in ScoreSpaceOption],
        default=ScoreSpaceOption.LogExp.value,
        help="Space of binding scores",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip optimization and re-render existing results",
    )
    parser.add_argument(
        "--reduce-fsm-max-error",
        type=float,
        default=None,
        help="Specifies the maximum error of the reduced FSM",
    )
    parser.add_argument(
        "--reduce-fsm-by",
        type=float,
        nargs="+",
        default=[0.875],
        help="FSM size reduction ratios (0 = DB FSM). Default: 0.875 (8-fold)",
    )
    parser.add_argument(
        "--hit-pval",
        type=float,
        default=2e-3,
        help="Motif hit p-value threshold (default: 2e-3)",
    )
    parser.add_argument(
        "--budgets",
        "-k",
        type=int,
        nargs="+",
        default=[50, 100, 150],
        help="Specifies the maximum number of non-dominated scores in each cell of the DP matrix (default: 50, 100, 150)",
    )
    parser.add_argument(
        "--sampler-alpha",
        type=str,
        nargs="+",
        default=["0.0", "1.0", "1.0_log_pos", "2.0_log_pos"],
        help="Specifies the exponent for the inverse Power-Law weighting of functional costs (default: 0.0, 1.0, 1.0_log_pos, 2.0_log_pos)",
    )
    parser.add_argument(
        "--exact-match-cost",
        action="store_true",
        help="Use the exact match dummy cost function",
    )
    parser.add_argument(
        "--codon-usage",
        type=Path,
        help="Codon usage file for calculating synonymous substitution costs",
    )
    parser.add_argument(
        "-alpha",
        type=float,
        default=0.5,
        help="Specifies the value for transition substitution cost (default is 0.5)",
    )
    parser.add_argument(
        "-beta",
        type=float,
        default=1.0,
        help="Specifies the value for transversion substitution cost (default is 1.0)",
    )
    parser.add_argument(
        "-w",
        type=float,
        default=500.0,
        help="Specifies the value for non-synonymous substitution cost (default is 500.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.target_sequence)
    seq_files: list[Path] = (
        [input_path] if input_path.is_file() else sorted(input_path.glob("*.txt"))
    )

    score_function_builder = (
        ScoreFunctionBuilder()
        .with_is_exact_match_cost(args.exact_match_cost)
        .with_codon_usage(args.codon_usage)
        .with_params(alpha=args.alpha, beta=args.beta, w=args.w)
    )
    fsm_builder = (
        FSMBuilder()
        .with_matrix_id(args.matrix_id)
        .with_binding_score_space(ScoreSpaceOption(args.binding_score_space))
        .with_hit_pvalue(args.hit_pval)
        .with_fsm_reduction(
            max_total_error=args.reduce_fsm_max_error,
            reduction_ratio_threshold=args.reduce_fsm_by,
        )
    )

    batches = run_design_grid(
        seq_files,
        score_function_builder,
        fsm_builder,
        k_values=args.budgets,
        sampler_alpha=args.sampler_alpha,
        reduce_fsm_by=args.reduce_fsm_by,
        mode=GridMode.DRY_RUN if args.dry_run else GridMode.RUN,
    )

    all_rows: list[dict[str, Any]] = []
    for exporters in batches.values():
        if exporters:
            all_rows.extend(render_and_compare(exporters))

    if all_rows:
        write_results_stream(
            iter(all_rows), Path("designer_results") / "pareto_comparison.csv"
        )


if __name__ == "__main__":
    main()

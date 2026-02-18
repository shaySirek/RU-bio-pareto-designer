#!/usr/bin/env python3

import argparse
from pathlib import Path

from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.shared.seq_design_utils.fsm_builder import FSMBuilder
from pareto_designer.shared.seq_design_utils.seq_designer import SequenceDesigner
from pareto_designer.algorithms.seq_design.sampling import SUS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sequence",
        "-s",
        type=Path,
        required=True,
        help="Target sequence (fasta file)",
    )
    parser.add_argument(
        "--matrix-id",
        "-m",
        type=str,
        required=True,
        help="Matrix ID of the binding motif",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip optimization and re-render existing results",
    )
    parser.add_argument(
        "--reduce-fsm-by",
        type=float,
        default=0.875,
        help="Specifies the percentage of reduction in size of the FSM of the binding motif (default is 0.875)",
    )
    parser.add_argument(
        "--budgets",
        "-k",
        type=int,
        nargs="+",
        default=[150, 200, 250],
        help="Specifies the maximum number of Pareto-optimal scores in each cell of the DP matrix (default: 150, 200, 250)",
    )
    parser.add_argument(
        "--sampler-alpha",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.2, 0.1],
        help="Specifies the scaling factor for functional costs in the sampler (default: 1.0, 0.5, 0.2, 0.1)",
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
        default=1.0,
        help="Specifies the value for transition substitution cost (default is 1.0)",
    )
    parser.add_argument(
        "-beta",
        type=float,
        default=2.0,
        help="Specifies the value for transversion substitution cost (default is 2.0)",
    )
    parser.add_argument(
        "-w",
        type=float,
        default=500.0,
        help="Specifies the value for non-synonymous substitution cost (default is 500.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.target_sequence)
    seq_files = [input_path] if input_path.is_file() else input_path.glob("*.txt")

    score_function_builder = (
        ScoreFunctionBuilder()
        .with_is_exact_match_cost(args.exact_match_cost)
        .with_codon_usage(args.codon_usage)
        .with_params(alpha=args.alpha, beta=args.beta, w=args.w)
    )
    fsm_builder = (
        FSMBuilder()
        .with_matrix_id(args.matrix_id)
        .with_fsm_reduction(reduction_ratio_threshold=args.reduce_fsm_by)
    )
    seq_designer = (
        SequenceDesigner()
        .with_score_function_builder(score_function_builder)
        .with_fsm_builder(fsm_builder)
    )

    for seq_file in seq_files:
        for k in args.budgets:
            for alpha in args.sampler_alpha:
                (
                    seq_designer.with_target_sequence(seq_file)
                    .with_sampler(SUS(k, alpha))
                    .run(dry_run=args.dry_run)
                )

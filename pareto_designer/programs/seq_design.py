#!/usr/bin/env python3

import argparse
from pathlib import Path

from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.shared.seq_design_utils.seq_designer import SequenceDesigner


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
        "--reduce-fsm-by",
        type=float,
        default=0.875,
        help="Specifies the percentage of reduction in size of the FSM of the binding motif (default is 0.875)",
    )
    parser.add_argument(
        "--limit-solutions",
        "-l",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Specifies the maximum number of Pareto-optimal sequences (default: 64, 128)",
    )
    parser.add_argument(
        "--exact-match-cost",
        action="store_true",
        help="Use the exact match dummy cost function",
    )
    parser.add_argument(
        "--codon-usage",
        type=Path,
        help="Codon usage file for cost function",
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
        default=100.0,
        help="Specifies the value for non-synonymous substitution cost (default is 100.0)",
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
    seq_designer = (
        SequenceDesigner()
        .with_binding_motif(args.matrix_id)
        .with_reduced_fsm(reduction_ratio_threshold=args.reduce_fsm_by)
    )

    for seq_file in seq_files:
        for po_limit in args.limit_solutions:
            (
                seq_designer.with_target_sequence(seq_file, score_function_builder)
                .with_solutions_limit(po_limit)
                .run()
            )

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from pareto_designer.algorithms.spaces import ScoreSpaceOption
from pareto_designer.shared.seq_design_utils.fsm_builder import FSMBuilder
from pareto_designer.shared.seq_design_utils.pareto_utils import render_and_compare
from pareto_designer.shared.seq_design_utils.run_grid import GridMode, run_design_grid
from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.views.experiment_report.config import (
    ConfigError,
    effective_grid,
    load_experiment_config,
    seq_files,
)
from pareto_designer.views.experiment_report.xlsx_exporter import (
    ExperimentReportExporter,
)

SWEEP_NAMES = ("alpha", "k", "fsm_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="YAML experiment config",
    )
    parser.add_argument(
        "--sweep",
        choices=SWEEP_NAMES,
        help="Run a single sweep (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Re-render existing results only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when results_metadata.json exists",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export Excel report after sweeps",
    )
    return parser.parse_args()


def _build_score_function_builder(config) -> ScoreFunctionBuilder:
    return (
        ScoreFunctionBuilder()
        .with_codon_usage(Path(config.fixed["codon_usage"]))
        .with_params(**config.fixed["cost_params"])
    )


def _build_fsm_builder(config, reduce_fsm_by: list[float]) -> FSMBuilder:
    return (
        FSMBuilder()
        .with_matrix_id(config.fixed["matrix_id"])
        .with_binding_score_space(ScoreSpaceOption(config.fixed["binding_score_space"]))
        .with_hit_pvalue(float(config.fixed.get("hit_pval", 0.002)))
        .with_fsm_reduction(None, reduce_fsm_by)
    )


def run_sweep(config, sweep_name: str, *, dry_run: bool, force: bool) -> None:
    grid = effective_grid(config, sweep_name)
    score_builder = _build_score_function_builder(config)
    fsm_builder = _build_fsm_builder(config, grid.reduce_fsm_by)
    results_root = Path(config.fixed["results_root"])

    logger.info(f"Running sweep {sweep_name!r} on {len(seq_files(config))} sequence(s)")
    batches = run_design_grid(
        seq_files(config),
        score_builder,
        fsm_builder,
        k_values=grid.k_values,
        sampler_alpha=grid.sampler_alpha,
        reduce_fsm_by=grid.reduce_fsm_by,
        mode=(
            GridMode.DRY_RUN
            if dry_run
            else GridMode.FORCE if force else GridMode.SKIP_EXISTING
        ),
        results_root=results_root,
    )

    for seq_id, exporters in batches.items():
        if exporters:
            render_and_compare(exporters)
            logger.info(f"Completed {sweep_name} sweep for {seq_id}")


def main() -> None:
    args = parse_args()
    try:
        config = load_experiment_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    sweeps = [args.sweep] if args.sweep else list(SWEEP_NAMES)
    for sweep_name in sweeps:
        run_sweep(config, sweep_name, dry_run=args.dry_run, force=args.force)

    if args.export:
        exporter = ExperimentReportExporter(Path(config.fixed["results_root"]), config)
        out = exporter.export()
        logger.info(f"Exported report to {out}")


if __name__ == "__main__":
    main()

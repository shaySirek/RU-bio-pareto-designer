#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from pareto_designer.views.experiment_report.config import (
    ConfigError,
    load_experiment_config,
)
from pareto_designer.views.experiment_report.xlsx_exporter import (
    ExperimentReportExporter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export designer results to Excel report"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="YAML experiment config (optional)",
    )
    parser.add_argument(
        "--results-root",
        "-r",
        type=Path,
        default=Path("designer_results"),
        help="Root directory containing run results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output xlsx path (default from config or results root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = None
    results_root = args.results_root

    if args.config:
        try:
            config = load_experiment_config(args.config)
        except ConfigError as exc:
            raise SystemExit(str(exc)) from exc
        results_root = Path(config.fixed["results_root"])

    exporter = ExperimentReportExporter(results_root, config)
    out = exporter.export(args.output)
    logger.info(f"Exported {len(exporter.design_runs)} design runs to {out}")


if __name__ == "__main__":
    main()

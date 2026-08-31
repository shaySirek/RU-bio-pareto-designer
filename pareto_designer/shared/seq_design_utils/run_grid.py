from enum import StrEnum
from pathlib import Path

from loguru import logger

from pareto_designer.algorithms.seq_design.sampling import PowerLawSUS
from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter
from pareto_designer.shared.seq_design_utils.fsm_builder import FSMBuilder
from pareto_designer.shared.seq_design_utils.pareto_utils import parse_sampler_alpha
from pareto_designer.shared.seq_design_utils.run_paths import metadata_path
from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.shared.seq_design_utils.seq_designer import SequenceDesigner


class GridMode(StrEnum):
    """How to execute each grid cell."""

    RUN = "run"
    SKIP_EXISTING = "skip_existing"
    FORCE = "force"
    DRY_RUN = "dry_run"


def run_design_grid(
    seq_files: list[Path],
    score_function_builder: ScoreFunctionBuilder,
    fsm_builder: FSMBuilder,
    *,
    k_values: list[int],
    sampler_alpha: list[str],
    reduce_fsm_by: list[float],
    mode: GridMode = GridMode.RUN,
    results_root: Path | None = None,
) -> dict[str, dict[str, ParetoExporter]]:
    dry_run = mode == GridMode.DRY_RUN
    skip_existing = mode == GridMode.SKIP_EXISTING

    seq_designer = SequenceDesigner().with_score_function_builder(
        score_function_builder
    )
    allowed_ratios = set(reduce_fsm_by)
    fsm_contexts = [
        ctx
        for ctx in fsm_builder.iter_contexts(dry_run=dry_run)
        if ctx.reduce_fsm_by in allowed_ratios
    ]
    batches: dict[str, dict[str, ParetoExporter]] = {}
    root = results_root or Path("designer_results")

    for seq_file in seq_files:
        exporters: dict[str, ParetoExporter] = {}
        seq_designer.with_target_sequence(seq_file)
        if results_root is not None:
            seq_designer.with_results_root(results_root)
        cost_params_str = seq_designer._build(dry_run=True).run_ctx.cost_params_str
        seq_id = seq_file.stem

        for fsm_ctx in fsm_contexts:
            seq_designer.with_fsm_context(fsm_ctx)
            for k in k_values:
                for exp_str in sampler_alpha:
                    alpha, log_pos = parse_sampler_alpha(exp_str)
                    sampler = PowerLawSUS(k, alpha, log_pos)
                    meta_path = metadata_path(
                        root,
                        seq_id,
                        cost_params_str,
                        fsm_ctx.motif_id,
                        fsm_ctx.fsm_id,
                        sampler,
                    )
                    if skip_existing and meta_path.exists():
                        logger.info(f"Skipping existing run: {meta_path.parent}")
                        if dry_run:
                            exporter = seq_designer.with_sampler(sampler).run(
                                dry_run=True
                            )
                            if exporter._results:
                                exporters[f"{fsm_ctx.fsm_id}__{sampler.params}"] = (
                                    exporter
                                )
                        continue
                    if dry_run and not meta_path.exists():
                        continue
                    exporter = seq_designer.with_sampler(sampler).run(dry_run=dry_run)
                    exporters[f"{fsm_ctx.fsm_id}__{sampler.params}"] = exporter
        batches[seq_id] = exporters
    return batches

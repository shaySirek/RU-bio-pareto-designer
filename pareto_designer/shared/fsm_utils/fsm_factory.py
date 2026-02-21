from loguru import logger

from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.models.motif import StrandForBindingScore
from pareto_designer.shared.fsm_utils.coloring import (
    partition_to_colors,
    PartitioningMethod,
)
from pareto_designer.algorithms.fsm import FSM, ColoredFSM


def get_binding_motif_fsm(
    matrix_id: str,
    strand: StrandForBindingScore = StrandForBindingScore.Forward,
    **kwargs,
) -> tuple[BindingMotif, FSM, dict[str, float]]:
    motif_ctx = BindingMotif(matrix_id, **kwargs)

    logger.info(f"Building {motif_ctx.length}-dimensional DB FSM...")
    db_fsm: FSM = FSM[str, str].de_bruijn_fsm(set(motif_ctx.alphabet), motif_ctx.length)

    logger.info("Calculating binding scores from PSSM...")
    binding_score_map = motif_ctx.get_binding_score_map(strand)

    return (motif_ctx, db_fsm, binding_score_map)


def get_binding_motif_fsm_all_strands(
    matrix_id: str, **kwargs
) -> tuple[BindingMotif, FSM, dict[StrandForBindingScore, dict[str, float]]]:
    motif_ctx = BindingMotif(matrix_id, **kwargs)

    logger.info(f"Building {motif_ctx.length}-dimensional DB FSM...")
    db_fsm: FSM = FSM[str, str].de_bruijn_fsm(set(motif_ctx.alphabet), motif_ctx.length)

    logger.info("Calculating binding scores from PSSM...")
    binding_score_maps = motif_ctx.get_binding_score_maps()

    return (motif_ctx, db_fsm, binding_score_maps)


def get_colored_fsm(
    db_fsm: FSM,
    binding_score_map: dict[str, float],
    n_colors: int,
    coloring_method: PartitioningMethod,
) -> ColoredFSM:
    logger.info(
        f"Coloring DB FSM by {coloring_method.value} clustering with k={n_colors}..."
    )
    colored_patterns, _ = partition_to_colors(
        binding_score_map, n_colors, coloring_method
    )
    colored_fsm: ColoredFSM = ColoredFSM[str, str, str].from_coloring(
        db_fsm,
        colored_patterns,
    )

    return colored_fsm

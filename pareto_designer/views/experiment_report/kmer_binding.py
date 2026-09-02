from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from pareto_designer.algorithms.spaces import score_space_from_fsm_id
from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.models.context import FSMContext
from pareto_designer.models.motif import StrandForBindingScore
from pareto_designer.shared.seq_design_utils.binding_metrics import (
    fill_kmer_binding_from_positional,
)
from pareto_designer.views.experiment_report.models import LoadedRun
from pareto_designer.views.experiment_report.paths import (
    motif_id_from_run_dir,
    resolve_path,
)


@lru_cache(maxsize=8)
def origin_map_for_motif(matrix_id: str) -> tuple[int, dict[str, float]]:
    motif = BindingMotif(matrix_id)
    origin_map = motif.get_binding_score_map(StrandForBindingScore.Double)
    return motif.length, origin_map


def _origin_from_contexts(
    fsm_contexts: Sequence[FSMContext] | None,
) -> dict[str, tuple[int, dict[str, float]]]:
    origin_by_motif: dict[str, tuple[int, dict[str, float]]] = {}
    if not fsm_contexts:
        return origin_by_motif
    for ctx in fsm_contexts:
        origin_by_motif.setdefault(
            ctx.motif_id, (ctx.motif_length, ctx.origin_binding_score_map)
        )
    return origin_by_motif


def fill_kmer_binding(
    runs: list[LoadedRun],
    fsm_contexts: Sequence[FSMContext] | None = None,
) -> None:
    origin_by_motif = _origin_from_contexts(fsm_contexts)
    for run in runs:
        motif_id = motif_id_from_run_dir(run.path)
        cached = origin_by_motif.get(motif_id)
        if cached is None:
            motif_length, origin_map = origin_map_for_motif(motif_id)
        else:
            motif_length, origin_map = cached
        fill_kmer_binding_from_positional(
            run.solutions,
            resolve_path(run.path.parent),
            origin_map,
            motif_length,
            score_space_from_fsm_id(run.params.fsm_id),
        )

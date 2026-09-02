from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

import numpy as np

from pareto_designer.algorithms.spaces import LinearSpace, ScoreSpace
from pareto_designer.models.context import FSMContext, ParetoResult
from pareto_designer.shared.binding_utils import get_binding, window_scores


@dataclass(frozen=True)
class KmerBindingScoreMse:
    mse: float
    err_std: float


def fsm_binding_score_mse(sse: float, db_fsm_size: int) -> float:
    if db_fsm_size <= 0 or not np.isfinite(sse):
        return float("nan")
    return float(sse) / float(db_fsm_size)


def kmer_binding_score_mse(
    reduced: np.ndarray,
    origin: np.ndarray,
    space: Type[ScoreSpace] = LinearSpace,
) -> KmerBindingScoreMse:
    n = min(np.asarray(reduced).size, np.asarray(origin).size)
    if n <= 0:
        return KmerBindingScoreMse(float("nan"), float("nan"))
    dist = np.asarray(
        space.distance(
            np.asarray(reduced[:n], dtype=float),
            np.asarray(origin[:n], dtype=float),
        ),
        dtype=float,
    )
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return KmerBindingScoreMse(float("nan"), float("nan"))
    return KmerBindingScoreMse(
        float(np.mean(dist)),
        float(np.std(dist, ddof=1)) if dist.size > 1 else float("nan"),
    )


def solution_kmer_binding_score_mse(seq: str, ctx: FSMContext) -> KmerBindingScoreMse:
    return kmer_binding_score_mse(
        get_binding(seq, ctx, use_origin=False),
        get_binding(seq, ctx, use_origin=True),
        ctx.binding_score_space,
    )


def fill_kmer_binding_from_positional(
    results: list[ParetoResult],
    run_dir: Path,
    origin_map: dict[str, float],
    motif_length: int,
    space: Type[ScoreSpace] = LinearSpace,
) -> None:
    for sol in results:
        npy_path = run_dir / sol.positional_objectives_file
        if not npy_path.exists():
            sol.kmer_binding_score_mse = float("nan")
            sol.kmer_binding_score_err_std = float("nan")
            continue
        metrics = kmer_binding_score_mse(
            np.load(npy_path)[:, 1],
            window_scores(sol.sequence, origin_map, motif_length),
            space,
        )
        sol.kmer_binding_score_mse = metrics.mse
        sol.kmer_binding_score_err_std = metrics.err_std


def run_kmer_binding_score_mse_summary(
    results: list[ParetoResult],
) -> KmerBindingScoreMse:
    values = np.array([r.kmer_binding_score_mse for r in results], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return KmerBindingScoreMse(float("nan"), float("nan"))
    return KmerBindingScoreMse(
        float(np.mean(finite)),
        float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
    )

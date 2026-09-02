from typing import Type

import numpy as np

from pareto_designer.algorithms.spaces import ScoreSpace
from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.models.context import FSMContext


def window_scores(
    seq: str, score_map: dict[str, float], motif_length: int
) -> np.ndarray:
    valid_len = len(seq) - motif_length + 1
    if valid_len <= 0:
        return np.array([], dtype=float)
    lookup = np.vectorize(lambda window: score_map.get(window, np.nan), otypes=[float])
    windows = np.array(
        [seq[i : i + motif_length] for i in range(valid_len)],
        dtype=f"U{motif_length}",
    )
    return lookup(windows)


def get_binding(seq: str, ctx: FSMContext, use_origin: bool = False) -> np.ndarray:
    seq_len = len(seq)
    data = np.full(seq_len, np.nan)
    score_map = ctx.origin_binding_score_map if use_origin else ctx.binding_score_map
    scores = window_scores(seq, score_map, ctx.motif_length)
    data[: scores.size] = scores
    return data


def get_total_binding(seq: str, ctx: FSMContext, use_origin: bool = False) -> float:
    scores = get_binding(seq, ctx, use_origin)
    finite = scores[np.isfinite(scores)]
    space: Type[ScoreSpace] = ctx.binding_score_space
    if finite.size == 0:
        return float(space.Identity)
    total = float(finite[0])
    for value in finite[1:]:
        total = space.add(total, float(value))
    return float(total)


def motif_hit_window_starts(seq: str, motif: BindingMotif, pvalue: float) -> set[int]:
    m_len = motif.length
    if len(seq) < m_len:
        return set()
    return {
        i
        for i in range(len(seq) - m_len + 1)
        if motif.is_significant_window(seq[i : i + m_len], pvalue)
    }


def motif_hit_binding_thresholds(ctx: FSMContext, n_hits: int = 3) -> list[float]:
    threshold = ctx.motif.hit_score_threshold(ctx.hit_pvalue)
    space = ctx.binding_score_space
    thresholds = [threshold]
    acc = threshold
    for _ in range(1, n_hits):
        acc = space.add(acc, threshold)
        thresholds.append(float(acc))
    return thresholds

import heapq
from typing import Iterable, Generator, Iterator, Any
from collections import defaultdict
from operator import itemgetter

import numpy as np

from pareto_designer.algorithms.seq_design.types import (
    T_SOLUTION,
    T_SOL_WITH_TRACK,
    T_LAZY_SOL_ITER_FACTORY,
    CompareFunc,
)


def default_compare(a: T_SOL_WITH_TRACK, b: T_SOL_WITH_TRACK) -> bool:
    x = a[0]
    y = b[0]
    return x[0] > y[0] or (x[0] == y[0] and x[1] < y[1])


def merge_sorted(
    sorted_scores_iters_factories: Iterable[T_LAZY_SOL_ITER_FACTORY],
    compare: CompareFunc = default_compare,
) -> Generator[T_SOL_WITH_TRACK, None, None]:

    class Node:
        __slots__ = ["val", "it"]

        def __init__(self, val: T_SOL_WITH_TRACK, it: Iterator[T_SOL_WITH_TRACK]):
            self.val = val
            self.it = it

        def __lt__(self, other):
            return compare(self.val, other.val)

    heap = []

    for factory in sorted_scores_iters_factories:
        iterator = iter(factory())
        try:
            val = next(iterator)
            heapq.heappush(heap, Node(val, iterator))
        except StopIteration:
            continue

    while heap:
        node = heapq.heappop(heap)
        yield node.val

        try:
            next_val = next(node.it)
            node.val = next_val
            heapq.heappush(heap, node)
        except StopIteration:
            node.it = None
            del node


def find_po(
    sorted_candidates: Iterable[T_SOL_WITH_TRACK],
    limit: int = 0,
    alpha: float = 2.0,
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    """
    Finds a representative sample of Pareto-optimal tuples using biased
    quantile-based pruning.

    This function processes a set of candidates already sorted by their
    primary objective. It first identifies the non-dominated set (Pareto
    frontier) and then prunes it to a maximum of `limit` solutions.

    The pruning utilizes a power-law distribution to select representatives,
    allowing for non-uniform sampling density across the frontier.

    Args:
        sorted_candidates: An iterable of (score, metadata) tuples,
            pre-sorted by the primary objective (index 0 of the score).
        limit: The maximum number of non-dominated solutions to retain
            (K). If 0, all non-dominated solutions are returned.
        alpha: The bias parameter for quantile selection.
            - alpha = 1.0: Uniform sampling (linear quantiles).
            - alpha > 1.0: Biased sampling favoring lower values of the
              primary objective (higher resolution at the 'start' of the
              frontier).
            - alpha < 1.0: Biased sampling favoring the secondary objective.

    Returns:
        A tuple containing:
            - po_scores: A list of non-dominated score tuples.
            - po_objs: A list of lists, where each sub-list contains all
              metadata objects (tracking info) associated with that score.

    Notes:
        The pruning mechanism maintains structural diversity by ensuring
        extreme points (anchors) are preserved, while the power-law bias
        concentrates computational budget on preferred trade-off regions.
    """

    candidates = list(sorted_candidates)
    if len(candidates) == 0:
        return [], []

    scores_array = np.fromiter(
        map(itemgetter(0), candidates), dtype=np.dtype((float, 2))
    )
    score_props = defaultdict(list)
    for score, prop in candidates:
        score_props[score].append(prop)

    b_values = scores_array[:, 1]
    running_min_b = np.minimum.accumulate(b_values)
    mask = np.concatenate(([True], b_values[1:] < running_min_b[:-1]))

    po_indices = np.where(mask)[0]
    final_indices = sample(po_indices, limit, alpha)

    po_scores: list[T_SOLUTION] = []
    po_objs: list[list[Any]] = []

    for idx in final_indices:
        target_score = tuple(scores_array[idx])
        po_scores.append(target_score)
        po_objs.append(score_props[target_score])

    return po_scores, po_objs


def sample(indices: np.ndarray, k: int, alpha: float):
    n = len(indices)

    # Unbounded or less than bound (k)
    if k == 0 or n <= k:
        return indices

    j = np.linspace(0, 1, k)
    sample_indices = np.round(np.power(j, alpha) * (n - 1)).astype(int)
    final_indices = indices[np.unique(sample_indices)]

    return final_indices


def find_po_from_sorted_iters(
    sorted_scores_iters_factories: Iterable[T_LAZY_SOL_ITER_FACTORY],
    limit: int = 0,
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    return find_po(merge_sorted(sorted_scores_iters_factories), limit)

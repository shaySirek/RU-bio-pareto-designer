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
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod, SUS


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
    sampler: SamplingMethod = SUS(0, 1.0),
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    """
    Identifies the Pareto-optimal frontier from a set of candidates and
    applies sampling.

    Args:
        sorted_candidates: An iterable of (score, metadata) tuples,
            pre-sorted by the primary objective (cost).
        sampler: The sampling precedure (Defaults to no sampling).

    Returns:
        A tuple containing a list of non-dominated score tuples and a
        list of associated metadata lists for each score.
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
    sampled_po_scores = sampler(scores_array, po_indices)

    po_scores: list[T_SOLUTION] = []
    po_objs: list[list[Any]] = []

    for score in sampled_po_scores:
        target_score = tuple(score)
        po_scores.append(target_score)
        po_objs.append(score_props[target_score])

    return po_scores, po_objs


def find_po_from_sorted_iters(
    sorted_scores_iters_factories: Iterable[T_LAZY_SOL_ITER_FACTORY],
    sampler: SamplingMethod = SUS(0, 1.0),  # no sampling
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    return find_po(merge_sorted(sorted_scores_iters_factories), sampler)

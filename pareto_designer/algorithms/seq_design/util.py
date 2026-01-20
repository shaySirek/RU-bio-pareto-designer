import heapq
from typing import Iterable, Generator, Iterator, Any

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
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    """
    Finds a representative sample of Pareto-optimal tuples using NumPy masking
    and vectorization for high performance and diversity.
    """
    po_scores: list[T_SOLUTION] = []
    po_objs: list[list[Any]] = []

    candidates = list(sorted_candidates)
    if not candidates:
        return po_scores, po_objs

    scores_array = np.array([c[0] for c in candidates])
    props = [c[1] for c in candidates]
    b_values = scores_array[:, 1]

    running_min_b = np.minimum.accumulate(b_values)
    mask = np.concatenate(([True], b_values[1:] < running_min_b[:-1]))

    po_indices = np.where(mask)[0]
    total_found = len(po_indices)

    num_to_keep = min(total_found, limit) if limit > 0 else total_found
    if num_to_keep > 1:
        sample_indices = np.round(np.linspace(0, total_found - 1, num_to_keep)).astype(
            int
        )
        final_indices = po_indices[sample_indices]
    else:
        final_indices = po_indices[:num_to_keep]

    for idx in final_indices:
        target_score = scores_array[idx]
        match_mask = np.all(scores_array == target_score, axis=1)
        matches = [props[i] for i in np.where(match_mask)[0]]

        po_scores.append(tuple(target_score))
        po_objs.append(matches)

    return po_scores, po_objs


def find_po_from_sorted_iters(
    sorted_scores_iters_factories: Iterable[T_LAZY_SOL_ITER_FACTORY],
    limit: int = 0,
) -> tuple[list[T_SOLUTION], list[list[Any]]]:
    return find_po(merge_sorted(sorted_scores_iters_factories), limit)

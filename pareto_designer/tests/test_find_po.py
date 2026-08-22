import numpy as np

from pareto_designer.algorithms.seq_design.util import find_po_from_arrays


def test_find_po_excludes_dominated():
    f = np.array([8.0, 10.0, 6.0, 9.0])
    b = np.array([5.0, 10.0, 1.0, 12.0])
    n = len(f)
    scores, _ = find_po_from_arrays(
        f, b, np.zeros(n, np.int32), np.zeros(n, np.int32), np.arange(n, dtype=np.int32)
    )
    got = set(zip(scores["f"].tolist(), scores["b"].tolist()))
    assert got == {(10.0, 10.0), (8.0, 5.0), (6.0, 1.0)}


def test_find_po_from_arrays_groups_duplicate_pointers():
    f = np.array([5.0, 5.0, 3.0])
    b = np.array([10.0, 10.0, 4.0])
    scores, ptrs = find_po_from_arrays(
        f,
        b,
        np.array([0, 1, 2], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 7, 0], dtype=np.int32),
    )
    assert len(scores) == 2
    assert scores[0]["f"] == 5.0 and scores[0]["b"] == 10.0
    assert scores[1]["f"] == 3.0 and scores[1]["b"] == 4.0
    assert len(ptrs[0]) == 2
    assert set(ptrs[0]["j"].tolist()) == {0, 7}

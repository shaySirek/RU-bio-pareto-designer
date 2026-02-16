from pareto_designer.algorithms.seq_design.util import find_po


def test_find_po():
    """Verify that dominated points are excluded."""
    candidates = [
        ((1.0, 10.0), "p1"),
        ((2.0, 11.0), "p2"),
        ((3.0, 5.0), "p3"),
        ((4.0, 5.0), "p4"),
        ((5.0, 2.0), "p5"),
    ]

    scores, _ = find_po(candidates)

    assert len(scores) == 3
    assert (1.0, 10.0) in scores
    assert (3.0, 5.0) in scores
    assert (5.0, 2.0) in scores


def test_find_po_duplicate_grouping():
    """Verify that multiple properties with the same score are grouped."""
    candidates = [
        ((1.0, 10.0), "p1_a"),
        ((1.0, 10.0), "p1_b"),
        ((2.0, 5.0), "p2"),
    ]

    scores, objs = find_po(candidates)

    idx_10 = scores.index((1.0, 10.0))
    assert len(objs[idx_10]) == 2
    assert "p1_a" in objs[idx_10]
    assert "p1_b" in objs[idx_10]

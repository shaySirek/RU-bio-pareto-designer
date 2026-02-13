from pareto_designer.algorithms.seq_design.util import find_po


def test_find_po_sampling_diversity():
    """Verify that we get a representative spread with uniform sampling (alpha=1.0)."""
    limit = 3
    # A perfect diagonal front: (0,10), (1,9), (2,8), (3,7), (4,6)
    candidates = [((float(i), float(10 - i)), f"p{i}") for i in range(5)]

    scores, _ = find_po(candidates, limit=limit, alpha=1.0)

    assert len(scores) == 3
    assert scores[0] == (0.0, 10.0)
    assert scores[1] == (2.0, 8.0)
    assert scores[2] == (4.0, 6.0)


def test_find_po_biased_sampling():
    """Verify that alpha > 1.0 concentrates points at the lower end."""
    limit = 3
    candidates = [((float(i), float(10 - i)), f"p{i}") for i in range(10)]

    scores, _ = find_po(candidates, limit=limit, alpha=2.0)

    assert len(scores) == 3
    assert scores[0][0] == 0.0
    assert scores[1][0] == 2.0  # Biased toward start
    assert scores[2][0] == 9.0


def test_find_po_masking_logic():
    """Verify that dominated points are excluded."""
    candidates = [
        ((1.0, 10.0), "p1"),
        ((2.0, 11.0), "p2"),
        ((3.0, 5.0), "p3"),
        ((4.0, 5.0), "p4"),
        ((5.0, 2.0), "p5"),
    ]

    scores, _ = find_po(candidates, limit=10, alpha=1.0)

    assert len(scores) == 3
    assert (1.0, 10.0) in scores
    assert (3.0, 5.0) in scores
    assert (5.0, 2.0) in scores


def test_find_po_duplicate_grouping():
    """Verify that multiple properties with the same score are grouped."""
    candidates = [((1.0, 10.0), "p1_a"), ((1.0, 10.0), "p1_b"), ((2.0, 5.0), "p2")]

    scores, objs = find_po(candidates, limit=10, alpha=1.0)

    idx_10 = scores.index((1.0, 10.0))
    assert len(objs[idx_10]) == 2
    assert "p1_a" in objs[idx_10]
    assert "p1_b" in objs[idx_10]


def test_find_po_edge_cases():
    """Test empty input and limits larger than available points."""
    s_empty, o_empty = find_po([], limit=5, alpha=1.0)
    assert s_empty == []
    assert o_empty == []

    candidates = [((1.0, 2.0), "p1"), ((2.0, 1.0), "p2")]
    scores, _ = find_po(candidates, limit=100, alpha=1.0)
    assert len(scores) == 2

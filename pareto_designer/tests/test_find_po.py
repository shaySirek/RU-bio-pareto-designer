from pareto_designer.algorithms.seq_design.util import find_po


def test_find_po_sampling_diversity():
    """Verify that we get a representative spread, not just the first N."""
    limit = 3
    # A perfect diagonal front: (0,10), (1,9), (2,8), (3,7), (4,6)
    # total_found = 5. Indices should be 0, 2, 4
    candidates = [((float(i), float(10 - i)), f"p{i}") for i in range(5)]

    scores, objs = find_po(candidates, limit=limit)

    assert len(scores) == 3
    assert scores[0] == (0.0, 10.0)  # The start
    assert scores[1] == (2.0, 8.0)  # The middle
    assert scores[2] == (4.0, 6.0)  # The end


def test_find_po_masking_logic():
    """Verify that dominated points are excluded via the running minimum mask."""
    candidates = [
        ((1, 10), "p1"),  # PO
        ((2, 11), "p2"),  # Dominated (11 > 10)
        ((3, 5), "p3"),  # PO
        ((4, 5), "p4"),  # Dominated (not strictly better than 5)
        ((5, 2), "p5"),  # PO
    ]

    scores, _ = find_po(candidates, limit=10)

    assert len(scores) == 3
    assert (1, 10) in scores
    assert (3, 5) in scores
    assert (5, 2) in scores
    assert (2, 11) not in scores
    assert (4, 5) not in scores


def test_find_po_duplicate_grouping():
    """Verify that multiple properties with the same score are grouped into one list."""
    candidates = [((1, 10), "p1_a"), ((1, 10), "p1_b"), ((2, 5), "p2")]

    scores, objs = find_po(candidates, limit=10)

    # Check that (1, 10) is one entry but has two properties
    idx_10 = scores.index((1, 10))
    assert len(objs[idx_10]) == 2
    assert "p1_a" in objs[idx_10]
    assert "p1_b" in objs[idx_10]


def test_find_po_edge_cases():
    """Test empty input and limits larger than available points."""
    # Empty
    assert find_po([], limit=5) == ([], [])

    # Limit larger than data
    candidates = [((1, 2), "p1"), ((2, 1), "p2")]
    scores, _ = find_po(candidates, limit=100)
    assert len(scores) == 2


def test_find_po_non_numeric_props():
    """Ensure Any types in properties don't break the NumPy logic."""
    candidates = [((1, 10), {"metadata": "info"}), ((2, 5), [1, 2, 3])]
    scores, objs = find_po(candidates, limit=2)
    assert isinstance(objs[0][0], dict)
    assert isinstance(objs[1][0], list)

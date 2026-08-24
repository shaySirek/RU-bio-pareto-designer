import pytest
import shutil
from collections import namedtuple
from typing import Generator

from pareto_designer.algorithms.seq_design.dp_matrix import (
    DP_Matrix,
    ITEM_SIZE,
    MAX_FILE_SIZE,
    get_flush_every,
)
from pareto_designer.shared.func_cost.exact_match_function import ExactMatchCostFunction

MockFSM = namedtuple("MockFSM", ["Sigma", "V", "v_init"])


@pytest.fixture
def target_seq() -> str:
    return "AAAAA"


@pytest.fixture
def exact_match_fn(target_seq: str) -> ExactMatchCostFunction:
    return ExactMatchCostFunction(target_sequence=target_seq)


@pytest.fixture
def test_env() -> Generator[DP_Matrix, None, None]:
    fsm = MockFSM(Sigma={"A", "C", "G", "T"}, V={"q0", "q1", "q2", "q3"}, v_init="q0")
    checkpoint_dir = ".test_dp_matrix_types"
    dp = DP_Matrix(fsm, n=5, flush_every=2, checkpoint_dir=checkpoint_dir)
    dp.start_row()
    dp.update(fsm.v_init, [(0.0, 0.0)], None)
    dp.end_row(0)
    yield dp

    if dp.checkpoint_path.exists():
        shutil.rmtree(dp.checkpoint_path.parent)


def test_get_flush_every_at_least_one():
    """A single DP row can exceed MAX_FILE_SIZE; flush_every must stay >= 1."""
    k = 100
    num_states = (MAX_FILE_SIZE // (k * ITEM_SIZE)) + 1
    fsm = MockFSM(Sigma={"A", "C", "G", "T"}, V=set(range(num_states)), v_init=0)
    assert get_flush_every(fsm, k) == 1


def test_reconstructed_sequence_costs(
    test_env: DP_Matrix, exact_match_fn: ExactMatchCostFunction
):
    """
    Verifies that get_costs correctly identifies position-wise costs
    for a sequence reconstructed from the DP Matrix.
    Path: 'GAAAG' vs Target: 'AAAAA'
    Expected costs: [1.0, 0.0, 0.0, 0.0, 1.0]
    """
    dp = test_env
    q0, q1 = "q0", "q1"
    test_seq = "GAAAG"

    current_f = 0.0
    for i in range(1, dp.n + 1):
        dp.start_row()
        char = test_seq[i - 1]
        u_prev = q0 if i == 1 else q1

        current_f += exact_match_fn(i - 1, u_prev, char)
        dp.update(q1, [(current_f, 0.0)], [[((u_prev, char), 0)]])
        dp.end_row(i)

    po_set = dp.reconstruct_po_set()
    reconstructed_seq, _ = list(po_set)[0]

    costs = exact_match_fn.get_costs(reconstructed_seq)

    assert reconstructed_seq == test_seq
    assert len(costs) == dp.n
    assert costs == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert sum(costs) == 2.0


def test_exact_match_po_integration(
    test_env: DP_Matrix, exact_match_fn: ExactMatchCostFunction
):
    """
    Tests DP updates across all rows where two distinct paths are Pareto-optimal.
    """
    dp = test_env
    q0, q1, q2 = "q0", "q1", "q2"

    for i in range(1, dp.n + 1):
        dp.start_row()
        idx = i - 1

        if i == 1:
            f_a = 0.0 + exact_match_fn(idx, q0, "A")
            dp.update(q1, [(f_a, 2.0)], [[((q0, "A"), 0)]])
            f_g = 0.0 + exact_match_fn(idx, q0, "G")
            dp.update(q2, [(f_g, 0.0)], [[((q0, "G"), 0)]])
        else:
            prev_q1 = dp.get(q1)[0]
            f_a = prev_q1["f"] + exact_match_fn(idx, q1, "A")
            b_a = prev_q1["b"] + 2.0
            dp.update(q1, [(f_a, b_a)], [[((q1, "A"), 0)]])

            prev_q2 = dp.get(q2)[0]
            f_g = prev_q2["f"] + exact_match_fn(idx, q2, "G")
            b_g = prev_q2["b"] + 0.0
            dp.update(q2, [(f_g, b_g)], [[((q2, "G"), 0)]])

        dp.end_row(i)

    po_set = dp.reconstruct_po_set()
    results = {res[0]: res[1] for res in po_set}

    assert len(results) == 2
    assert results["AAAAA"] == (0.0, 10.0)
    assert results["GGGGG"] == (-5.0, 0.0)


def test_full_reconstruction(test_env: DP_Matrix):
    """
    Fills the matrix up to n=5 and ensures reconstruction
    traverses the full depth of the matrix.
    """
    dp = test_env
    q0, q1 = "q0", "q1"

    for i in range(1, dp.n + 1):
        dp.start_row()
        u_prev = q0 if i == 1 else q1
        dp.update(q1, [(float(i), 0.0)], [[((u_prev, "A"), 0)]])
        dp.end_row(i)

    po_set = dp.reconstruct_po_set()

    assert len(po_set) == 1
    seq, score = list(po_set)[0]

    assert len(seq) == dp.n
    assert seq == "A" * dp.n
    assert score == (5.0, 0.0)


def test_lexsort_pointer_integrity(test_env: DP_Matrix):
    """
    Ensures that when solutions are reordered (like your lexsort logic),
    the DP_Matrix still retrieves the correct back-pointer for the correct score index.
    """
    dp = test_env
    q0, q1 = "q0", "q1"

    dp.start_row()
    po_scores = [(10.0, 10.0), (2.0, 2.0)]
    po_back_ptrs = [[((q0, "A"), 0)], [((q0, "G"), 0)]]
    dp.update(q1, po_scores, po_back_ptrs)
    dp.end_row(1)

    bps = dp.get_back_ptrs(1, dp._get_state_index(q1), 1)
    assert len(bps) == 1
    assert bps[0].sigma_idx == dp._sigma_to_idx["G"]

    scores = dp.get(q1)
    assert scores[1]["f"] == 2.0
    assert scores[1]["b"] == 2.0


def test_disk_chunk_caching(test_env: DP_Matrix):
    """Tests that the _cached_chunk_idx logic correctly switches between files."""
    dp = test_env
    q0, q1 = "q0", "q1"

    for i in range(1, 7):
        dp.start_row()
        dp.update(q1, [(float(i), 10.0 - i)], [[((q0, "A"), 0)]])
        dp.end_row(i)

    dp.get_back_ptrs(1, dp._get_state_index(q1), 0)
    assert dp._cached_chunk_idx == 0

    dp.get_back_ptrs(5, dp._state_to_idx[q1], 0)
    assert dp._cached_chunk_idx == 2

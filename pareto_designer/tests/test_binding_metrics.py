from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np

from pareto_designer.algorithms.spaces import (
    ExpSpace,
    LinearSpace,
    score_space_from_fsm_id,
)
from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.binding_metrics import (
    fill_kmer_binding_from_positional,
    fsm_binding_score_mse,
    kmer_binding_score_mse,
    run_kmer_binding_score_mse_summary,
    solution_kmer_binding_score_mse,
)


def _mock_ctx(motif_length: int = 2) -> MagicMock:
    ctx = MagicMock()
    ctx.motif_length = motif_length
    ctx.origin_binding_score_map = {"AC": 1.0, "CG": 2.0, "GT": 3.0}
    ctx.binding_score_map = {"AC": 1.5, "CG": 2.5, "GT": 2.0}
    ctx.binding_score_space = LinearSpace
    return ctx


def test_solution_kmer_binding_score_mse():
    mse = solution_kmer_binding_score_mse("ACGT", _mock_ctx())
    assert math.isclose(mse.mse, 0.5)
    assert math.isclose(mse.err_std, math.sqrt(0.1875))


def test_solution_kmer_binding_score_mse_short_sequence():
    mse = solution_kmer_binding_score_mse("A", _mock_ctx())
    assert math.isnan(mse.mse)
    assert math.isnan(mse.err_std)


def test_kmer_binding_score_mse_from_arrays():
    mse = kmer_binding_score_mse(
        np.array([1.5, 2.5, 2.0, np.nan]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert math.isclose(mse.mse, 0.5)
    assert math.isclose(mse.err_std, math.sqrt(0.1875))


def test_run_kmer_binding_score_mse_summary():
    results = [
        ParetoResult(
            cost=1.0,
            binding_score=1.0,
            origin_binding_score=1.1,
            id="001",
            url="001_details.html",
            txt_file="001.txt",
            fasta_file="001.fa",
            positional_objectives_file="001.npy",
            max_positional_cost=1.0,
            min_positional_binding=0.0,
            max_positional_binding=1.0,
            sequence="ACGT",
            n_cost_items=1,
            motif_hits=[],
            kmer_binding_score_mse=0.4,
            kmer_binding_score_err_std=0.1,
        ),
        ParetoResult(
            cost=2.0,
            binding_score=2.0,
            origin_binding_score=2.1,
            id="002",
            url="002_details.html",
            txt_file="002.txt",
            fasta_file="002.fa",
            positional_objectives_file="002.npy",
            max_positional_cost=1.0,
            min_positional_binding=0.0,
            max_positional_binding=1.0,
            sequence="ACGT",
            n_cost_items=1,
            motif_hits=[],
            kmer_binding_score_mse=0.6,
            kmer_binding_score_err_std=0.2,
        ),
    ]
    summary = run_kmer_binding_score_mse_summary(results)
    assert math.isclose(summary.mse, 0.5)
    assert math.isclose(summary.err_std, math.sqrt(0.02))


def test_fill_kmer_binding_from_positional(tmp_path):
    npy = tmp_path / "001.npy"
    np.save(npy, np.column_stack((np.zeros(4), np.array([1.5, 2.5, 2.0, np.nan]))))
    sol = ParetoResult(
        cost=1.0,
        binding_score=1.0,
        origin_binding_score=1.1,
        id="001",
        url="001_details.html",
        txt_file="001.txt",
        fasta_file="001.fa",
        positional_objectives_file="001.npy",
        max_positional_cost=1.0,
        min_positional_binding=0.0,
        max_positional_binding=1.0,
        sequence="ACGT",
        n_cost_items=1,
        motif_hits=[],
    )
    fill_kmer_binding_from_positional(
        [sol], tmp_path, {"AC": 1.0, "CG": 2.0, "GT": 3.0}, 2, LinearSpace
    )
    assert math.isclose(sol.kmer_binding_score_mse, 0.5)
    assert math.isclose(sol.kmer_binding_score_err_std, math.sqrt(0.1875))


def test_kmer_binding_score_mse_uses_exp_space():
    reduced = np.array([1.5, 2.5, 2.0])
    origin = np.array([1.0, 2.0, 3.0])
    mse = kmer_binding_score_mse(reduced, origin, ExpSpace)
    dist = np.asarray(ExpSpace.distance(reduced, origin))
    assert math.isclose(mse.mse, float(np.mean(dist)))
    assert math.isclose(mse.err_std, float(np.std(dist, ddof=1)))


def test_fsm_binding_score_mse():
    mse = fsm_binding_score_mse(0.14203769159523033, 16384)
    assert math.isclose(mse, 8.669292699904195e-06)
    assert math.isnan(fsm_binding_score_mse(0.1, 0))


def test_score_space_from_fsm_id():
    assert score_space_from_fsm_id("logexp_reduced_fsm_2048") is ExpSpace
    assert score_space_from_fsm_id("linear_db_fsm") is LinearSpace

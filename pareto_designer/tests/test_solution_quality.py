from __future__ import annotations

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.solution_quality import (
    SolutionRegion,
    classify_run_solutions,
    distribution_stats,
    has_nonsyn_substitution,
    region_borders,
    roi_distributions,
)


def _sol(
    sid: str,
    cost: float,
    binding: float,
    *,
    hits: list[tuple[int, int]] | None = None,
    n_nonsyn: int = 0,
    max_positional_cost: float = 1.0,
) -> ParetoResult:
    return ParetoResult(
        cost=cost,
        binding_score=binding,
        origin_binding_score=binding,
        id=sid,
        url=f"{sid}_details.html",
        txt_file=f"{sid}.txt",
        fasta_file=f"{sid}.fa",
        positional_objectives_file=f"{sid}.npy",
        max_positional_cost=max_positional_cost,
        min_positional_binding=0.0,
        max_positional_binding=1.0,
        sequence="ACGT",
        n_cost_items=1,
        n_nonsyn=n_nonsyn,
        motif_hits=hits or [],
    )


def test_classify_hits_roi_and_plateau():
    solutions = [
        _sol("001", 10.0, 20.0, hits=[(1, 7)]),
        _sol("002", 10.0, 19.0, hits=[(2, 8)]),
        _sol("003", 20.0, 15.0),
        _sol("004", 50.0, 10.0),
        _sol("005", 80.0, 9.5),
        _sol("006", 110.0, 9.4),
        _sol("007", 140.0, 9.3),
    ]
    counts, regions = classify_run_solutions(
        solutions, w=500.0, eps_binding=0.5, eps_rel=0.02, min_plateau_len=3
    )
    assert counts.n_with_hits == 2
    assert counts.n_nonsyn == 0
    assert counts.n_roi == 2
    assert counts.n_plateau == 3
    assert regions["001"] == SolutionRegion.HITS
    assert regions["003"] == SolutionRegion.ROI
    assert regions["005"] == SolutionRegion.PLATEAU
    assert (
        counts.n_with_hits + counts.n_nonsyn + counts.n_roi + counts.n_plateau
        == len(solutions)
    )


def test_nonsyn_excluded_from_roi():
    solutions = [
        _sol("001", 20.0, 15.0, n_nonsyn=1, max_positional_cost=501.0),
        _sol("002", 50.0, 10.0),
        _sol("003", 80.0, 9.5),
        _sol("004", 110.0, 9.4),
        _sol("005", 140.0, 9.3),
    ]
    counts, regions = classify_run_solutions(
        solutions, w=500.0, eps_binding=0.5, min_plateau_len=3
    )
    assert counts.n_nonsyn == 1
    assert regions["001"] == SolutionRegion.NONSYN
    assert regions["002"] == SolutionRegion.ROI
    assert regions["004"] == SolutionRegion.PLATEAU
    assert counts.n_roi == 1


def test_has_nonsyn_falls_back_to_max_positional_cost():
    sol = _sol("001", 10.0, 5.0, max_positional_cost=503.0)
    assert has_nonsyn_substitution(sol, w=500.0)
    assert not has_nonsyn_substitution(_sol("002", 10.0, 5.0), w=500.0)


def test_all_clean_frontier_is_roi_when_binding_keeps_dropping():
    solutions = [
        _sol("001", 10.0, 20.0),
        _sol("002", 20.0, 15.0),
        _sol("003", 30.0, 10.0),
        _sol("004", 40.0, 5.0),
    ]
    counts, _ = classify_run_solutions(
        solutions, w=500.0, eps_binding=0.5, min_plateau_len=3
    )
    assert counts.n_with_hits == 0
    assert counts.n_nonsyn == 0
    assert counts.n_roi == 4
    assert counts.n_plateau == 0
    assert counts.plateau_onset_cost is None


def test_single_clean_solution_counts_as_roi():
    solutions = [_sol("001", 10.0, 20.0, hits=[(1, 7)]), _sol("002", 20.0, 15.0)]
    counts, regions = classify_run_solutions(solutions, w=500.0)
    assert counts.n_with_hits == 1
    assert counts.n_roi == 1
    assert counts.n_plateau == 0
    assert regions["002"] == SolutionRegion.ROI


def test_region_borders():
    solutions = [
        _sol("001", 10.0, 20.0, hits=[(1, 7)]),
        _sol("002", 20.0, 15.0),
        _sol("003", 50.0, 9.5),
        _sol("004", 80.0, 9.5),
        _sol("005", 110.0, 9.4),
        _sol("006", 140.0, 9.3),
    ]
    borders = region_borders(solutions, w=500.0, eps_binding=0.5, min_plateau_len=3)
    assert borders.first_hit_free_cost == 20.0
    assert borders.plateau_onset_cost == 80.0


def test_late_binding_drop_does_not_delay_plateau_onset():
    solutions = [
        _sol("001", 10.0, 10.0),
        _sol("002", 20.0, 8.0),
        _sol("003", 40.0, 5.0),
        _sol("004", 80.0, 5.0),
        _sol("005", 120.0, 5.0),
        _sol("006", 160.0, 5.0),
        _sol("007", 200.0, 5.0),
        _sol("008", 240.0, 5.0),
        _sol("009", 280.0, 5.0),
        _sol("010", 320.0, 5.0),
        _sol("011", 2000.0, 4.5),
        _sol("012", 2030.0, 4.4),
        _sol("013", 2060.0, 4.3),
    ]
    counts, regions = classify_run_solutions(
        solutions,
        w=500.0,
        eps_binding=0.05,
        eps_rel=0.02,
        min_plateau_len=3,
        look_ahead=5,
    )
    assert counts.plateau_onset_cost == 80.0
    assert regions["004"] == SolutionRegion.PLATEAU
    assert regions["010"] == SolutionRegion.PLATEAU
    assert regions["011"] == SolutionRegion.PLATEAU


def test_roi_distribution_stats():
    solutions = [
        _sol("001", 10.0, 20.0, hits=[(1, 7)]),
        _sol("002", 20.0, 15.0),
        _sol("003", 50.0, 10.0),
        _sol("004", 80.0, 9.5),
        _sol("005", 110.0, 9.4),
        _sol("006", 140.0, 9.3),
    ]
    _, regions = classify_run_solutions(
        solutions, w=500.0, eps_binding=0.5, min_plateau_len=3
    )
    cost_roi, binding_roi = roi_distributions(solutions, regions)
    assert cost_roi.min == 20.0
    assert cost_roi.max == 50.0
    assert binding_roi.min == 10.0
    assert binding_roi.max == 15.0


def test_roi_distribution_empty_when_no_roi():
    solutions = [_sol("001", 10.0, 20.0, hits=[(1, 7)])]
    _, regions = classify_run_solutions(solutions, w=500.0)
    cost_roi, binding_roi = roi_distributions(solutions, regions)
    assert cost_roi.min != cost_roi.min
    assert binding_roi.max != binding_roi.max


def test_distribution_stats_empty():
    stats = distribution_stats([])
    assert stats.min != stats.min

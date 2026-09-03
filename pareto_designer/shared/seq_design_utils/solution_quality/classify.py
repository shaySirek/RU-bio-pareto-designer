from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pareto_designer.models.context import ParetoResult


class SolutionRegion(StrEnum):
    HITS = "hits"
    NONSYN = "nonsyn"
    ROI = "roi"
    PLATEAU = "plateau"


@dataclass(frozen=True)
class SolutionQualityCounts:
    n_with_hits: int
    n_nonsyn: int
    n_roi: int
    n_plateau: int
    plateau_onset_cost: float | None


@dataclass(frozen=True)
class RoiDistribution:
    min: float
    p25: float
    p50: float
    p75: float
    max: float
    mean: float
    std: float

    @staticmethod
    def empty() -> RoiDistribution:
        nan = float("nan")
        return RoiDistribution(nan, nan, nan, nan, nan, nan, nan)


@dataclass(frozen=True)
class RegionBorders:
    first_hit_free_cost: float | None
    plateau_onset_cost: float | None


def has_nonsyn_substitution(sol: ParetoResult, w: float | None = None) -> bool:
    if sol.n_nonsyn > 0:
        return True
    if w is not None and np.isfinite(w):
        return sol.max_positional_cost >= w
    return False


def distribution_stats(values: list[float]) -> RoiDistribution:
    if not values:
        return RoiDistribution.empty()
    arr = np.asarray(values, dtype=float)
    return RoiDistribution(
        min=float(np.min(arr)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
    )


def roi_distributions(
    solutions: list[ParetoResult],
    regions: dict[str, SolutionRegion],
) -> tuple[RoiDistribution, RoiDistribution]:
    roi_solutions = [
        sol for sol in solutions if regions.get(sol.id) == SolutionRegion.ROI
    ]
    return (
        distribution_stats([sol.cost for sol in roi_solutions]),
        distribution_stats([sol.binding_score for sol in roi_solutions]),
    )


def _plateau_epsilon(
    bindings: list[float], eps_binding: float, eps_rel: float
) -> float:
    if not bindings:
        return eps_binding
    b_range = max(bindings) - min(bindings)
    return max(eps_binding, eps_rel * b_range)


def _plateau_onset_index(
    costs: list[float],
    bindings: list[float],
    eps: float,
    min_plateau_len: int,
    *,
    look_ahead: int = 10,
    min_cost_span: float | None = None,
) -> int:
    n = len(bindings)
    if n <= 1:
        return n

    look_ahead = max(look_ahead, min_plateau_len)
    if min_cost_span is None:
        diffs = [
            costs[j + 1] - costs[j] for j in range(n - 1) if costs[j + 1] > costs[j]
        ]
        min_cost_span = float(np.median(diffs)) if diffs else 0.0

    slope_eps = eps / max(min_cost_span, 1e-9)

    for i in range(n - min_plateau_len + 1):
        end = min(i + look_ahead, n)
        short = bindings[i : i + min_plateau_len]
        if short[0] - min(short) >= eps:
            continue
        if i > 0 and costs[i] > costs[i - 1]:
            prev_slope = abs(
                (bindings[i] - bindings[i - 1]) / (costs[i] - costs[i - 1])
            )
            if prev_slope > slope_eps:
                continue
        binding_drop = bindings[i] - min(bindings[i:end])
        cost_span = costs[end - 1] - costs[i]
        if binding_drop < eps and cost_span >= min_cost_span:
            return i
    return n


def classify_run_solutions(
    solutions: list[ParetoResult],
    *,
    w: float | None = None,
    eps_binding: float = 0.01,
    eps_rel: float = 0.02,
    min_plateau_len: int = 3,
    look_ahead: int = 10,
) -> tuple[SolutionQualityCounts, dict[str, SolutionRegion]]:
    regions: dict[str, SolutionRegion] = {}
    n_with_hits = 0
    n_nonsyn = 0
    eligible: list[ParetoResult] = []

    for sol in solutions:
        if sol.n_motif_hits > 0:
            n_with_hits += 1
            regions[sol.id] = SolutionRegion.HITS
        elif has_nonsyn_substitution(sol, w):
            n_nonsyn += 1
            regions[sol.id] = SolutionRegion.NONSYN
        else:
            eligible.append(sol)

    eligible.sort(key=lambda r: r.cost)
    costs = [r.cost for r in eligible]
    bindings = [r.binding_score for r in eligible]
    eps = _plateau_epsilon(bindings, eps_binding, eps_rel)
    onset = _plateau_onset_index(
        costs,
        bindings,
        eps,
        min_plateau_len,
        look_ahead=look_ahead,
    )

    for idx, sol in enumerate(eligible):
        if idx < onset:
            regions[sol.id] = SolutionRegion.ROI
        else:
            regions[sol.id] = SolutionRegion.PLATEAU

    plateau_onset_cost = eligible[onset].cost if onset < len(eligible) else None
    return (
        SolutionQualityCounts(
            n_with_hits=n_with_hits,
            n_nonsyn=n_nonsyn,
            n_roi=onset,
            n_plateau=len(eligible) - onset,
            plateau_onset_cost=plateau_onset_cost,
        ),
        regions,
    )


def region_borders(
    solutions: list[ParetoResult],
    *,
    w: float | None = None,
    eps_binding: float = 0.01,
    eps_rel: float = 0.02,
    min_plateau_len: int = 3,
) -> RegionBorders:
    counts, _ = classify_run_solutions(
        solutions,
        w=w,
        eps_binding=eps_binding,
        eps_rel=eps_rel,
        min_plateau_len=min_plateau_len,
    )
    clean = sorted((r for r in solutions if r.n_motif_hits == 0), key=lambda r: r.cost)
    first_hit_free = clean[0].cost if clean else None
    return RegionBorders(
        first_hit_free_cost=first_hit_free,
        plateau_onset_cost=counts.plateau_onset_cost,
    )

import numpy as np

from pareto_designer.models.context import ParetoResult


def binding_score_sse(results: list[ParetoResult]) -> float:
    if not results:
        return float("nan")
    origin = np.array([r.origin_binding_score for r in results], dtype=float)
    approx = np.array([r.binding_score for r in results], dtype=float)
    return float(np.sum(np.square(approx - origin)))

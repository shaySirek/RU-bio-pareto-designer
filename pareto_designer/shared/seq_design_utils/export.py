from concurrent.futures import ThreadPoolExecutor, wait
from typing import Iterable

from loguru import logger
import numpy as np
from pathlib import Path

from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.parsing import write_sequence
from pareto_designer.shared.csv_writer import write_results_stream


def export_solutions(
    solutions: Iterable[tuple[str, T_SOLUTION]],
    score_function: ScoreFunction,
    path: Path,
):
    logger.info(f"Exporting {len(solutions)} Pareto-optimal solutions into {str(path)}")
    path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(export_solution, idx, sol, score_function, path)
            for idx, sol in enumerate(solutions)
        ]
        done, _ = wait(futures)
        for future in done:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task failed with error: {e}")


def export_solution(
    sol_idx: int,
    solution: tuple[str, T_SOLUTION],
    score_function: ScoreFunction,
    path: Path,
):
    sequence, (f_score, binding_score) = solution
    cost_from_alg = -f_score
    logger.info(
        f"Sequence no. {sol_idx + 1:03d}:"
        f"\n\tcost={cost_from_alg:.6f}"
        f"\n\tbinding_score={binding_score:.6f}"
    )
    costs = np.array(score_function.get_costs(sequence), dtype=float)
    total_cost = np.sum(costs)
    if not np.isclose(total_cost, cost_from_alg):
        logger.warning(
            f"Sequence no. {sol_idx + 1:03d}:"
            f" calculated cost is {total_cost:.6f},"
            f" whereas the cost returned by the algorithm is {cost_from_alg:.6f}"
            f"\n\tsolution: {sequence}"
            f"\n\ttarget:   {score_function.target_sequence}"
            f"\n\tcosts:    {', '.join(map(str, list(costs)))}"
        )

    sol_path = path / f"{sol_idx:03d}_sequence.txt"
    costs_path = path / f"{sol_idx:03d}_costs.csv"
    with sol_path.open("wt") as f:
        f.write(sequence)
    write_sequence(sol_path.with_suffix(".fa"), sequence)
    write_results_stream(_get_cost_generator(costs), costs_path)


def _get_cost_generator(arr: np.ndarray):
    positive_indices = np.where(arr > 0)[0]
    for idx in positive_indices:
        yield {"position": int(idx), "cost": float(arr[idx])}

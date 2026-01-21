from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from loguru import logger

from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.parsing import write_sequence
from pareto_designer.models.pareto_front import RunContext, ParetoResult
from pareto_designer.views.pareto_front.html_exporter import (
    render_solution_html,
    render_pareto_front,
)


def export_solutions(
    solutions: Iterable[tuple[str, T_SOLUTION]],
    ctx: RunContext,
    score_function: ScoreFunction,
):
    logger.info(
        f"Exporting {len(solutions)} Pareto-optimal solutions into {ctx.output_path}"
    )
    ctx.output_path.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(export_solution, idx, sol, ctx, score_function)
            for idx, sol in enumerate(sorted(solutions, key=lambda x: -x[1][0]))
        ]
        done, _ = wait(futures)

        results: list[ParetoResult] = []
        for future in done:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed with error: {e}")

    render_pareto_front(ctx, results)


def export_solution(
    sol_idx: int,
    solution: tuple[str, T_SOLUTION],
    ctx: RunContext,
    score_function: ScoreFunction,
) -> ParetoResult:
    sol_id = f"{sol_idx + 1:03d}"
    sequence, (f_score, binding_score) = solution
    functional_cost = -f_score
    logger.debug(
        f"Sequence no. {sol_id}:"
        f"\n\tcost={functional_cost:.6f}"
        f"\n\tbinding_score={binding_score:.6f}"
    )
    costs = np.array(score_function.get_costs(sequence), dtype=float)
    calculated_functional_cost = np.sum(costs)
    if not np.isclose(calculated_functional_cost, functional_cost):
        logger.warning(
            f"Sequence no. {sol_id}:"
            f" calculated cost is {calculated_functional_cost:.6f},"
            f" whereas the cost returned by the algorithm is {functional_cost:.6f}"
            f"\n\tsolution: {sequence}"
            f"\n\ttarget:   {score_function.target_sequence}"
            f"\n\tcosts:    {', '.join(map(str, list(costs)))}"
        )
    if np.isclose(functional_cost, 0, atol=1e-9):
        functional_cost = 0.0

    result = ParetoResult(
        cost=functional_cost,
        binding_score=binding_score,
        id=sol_id,
        url=f"{sol_id}_details.html",
        sequence=sequence,
        costs=costs,
    )
    render_solution_html(ctx, result)

    sol_base = ctx.output_path / f"{sol_id}_sequence"
    with sol_base.with_suffix(".txt").open("wt") as f:
        f.write(sequence)
    write_sequence(sol_base.with_suffix(".fa"), sequence)

    return result

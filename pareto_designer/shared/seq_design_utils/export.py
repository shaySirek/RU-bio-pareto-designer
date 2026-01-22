from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, wait

from pathlib import Path
import numpy as np
from loguru import logger

from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.bio_fetcher.fimo import get_number_of_hits
from pareto_designer.bio_fetcher.paths import MOTIF_DIR
from pareto_designer.models.motif import BindingMotif
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
    motif: BindingMotif,
):
    logger.info(
        f"Exporting {ctx.n_solutions} Pareto-optimal solutions into {ctx.output_path}"
    )
    ctx.output_path.mkdir(parents=True, exist_ok=True)

    motif_file = motif.dump("meme", MOTIF_DIR)
    results: list[ParetoResult] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                export_solution, idx, sol, ctx, score_function, motif, motif_file
            )
            for idx, sol in enumerate(sorted(solutions, key=lambda x: -x[1][0]))
        ]
        done, _ = wait(futures)

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
    motif: BindingMotif,
    motif_file: Path,
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

    sol_base = ctx.output_path / f"{sol_id}_sequence"
    sol_txt_file = sol_base.with_suffix(".txt")
    sol_fasta_file = sol_base.with_suffix(".fa")
    with sol_txt_file.open("wt") as f:
        f.write(sequence)
    write_sequence(sol_fasta_file, sequence, header=f"Solution {sol_id}")
    n_motif_hits = get_number_of_hits(sol_id, sol_fasta_file, motif, motif_file)

    result = ParetoResult(
        cost=functional_cost,
        binding_score=binding_score,
        id=sol_id,
        url=f"{sol_id}_details.html",
        txt_file=sol_txt_file.name,
        fasta_file=sol_fasta_file.name,
        sequence=sequence,
        costs=costs,
        n_motif_hits=n_motif_hits,
    )
    render_solution_html(ctx, result)

    return result

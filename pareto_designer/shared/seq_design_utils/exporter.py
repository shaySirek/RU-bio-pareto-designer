import json
from pathlib import Path
from typing import Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.bio_fetcher.fimo import get_number_of_hits
from pareto_designer.bio_fetcher.paths import MOTIF_DIR
from pareto_designer.models.context import ParetoResult, DesignContext
from pareto_designer.shared.parsing import write_sequence
from pareto_designer.views.pareto_front.html_exporter import (
    render_solution_html,
    render_pareto_front_html,
)
from pareto_designer.views.pareto_front.png_exporter import (
    render_heatmap_png,
    render_pareto_front_png,
)


class ParetoExporter:
    def __init__(self, design_ctx: DesignContext):
        self.design_ctx = design_ctx
        self.ctx = design_ctx.run_ctx
        self.score_function = design_ctx.score_function
        self.motif = design_ctx.fsm_ctx.motif
        self.results: list[ParetoResult] = []
        self.ctx.output_path.mkdir(parents=True, exist_ok=True)

    def process_all(self, solutions: Iterable[tuple[str, T_SOLUTION]]):
        motif_file = self.motif.dump("meme", MOTIF_DIR)
        sorted_sols = sorted(solutions, key=lambda x: -x[1][0])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._process_single, idx, sol, motif_file)
                for idx, sol in enumerate(sorted_sols)
            ]
            done, _ = wait(futures)
            self.results = [f.result() for f in done if not f.exception()]

        self.results.sort(key=lambda x: x.id)

    def _process_single(
        self, sol_idx: int, solution: tuple[str, T_SOLUTION], motif_file: Path
    ) -> ParetoResult:
        sol_id = f"{sol_idx + 1:03d}"
        sequence, (f_score, binding_score) = solution
        functional_cost = max(0.0, -f_score)
        costs = np.array(self.score_function.get_costs(sequence), dtype=float)

        sol_base = self.ctx.output_path / f"{sol_id}_sequence"
        sol_fasta_file = sol_base.with_suffix(".fa")

        write_sequence(sol_fasta_file, sequence, header=f"Solution {sol_id}")
        n_hits = get_number_of_hits(sol_id, sol_fasta_file, self.motif, motif_file)

        return ParetoResult(
            cost=float(functional_cost),
            binding_score=float(binding_score),
            id=sol_id,
            url=f"{sol_id}_details.html",
            txt_file=f"{sol_id}_sequence.txt",
            fasta_file=f"{sol_id}_sequence.fa",
            sequence=sequence,
            costs=costs,
            n_motif_hits=n_hits,
        )

    def save(self):
        export_data = {
            "metadata": {
                "runtime": self.ctx.runtime,
                "n_solutions": len(self.results),
                "target_id": self.ctx.target_sequence_id,
            },
            "results": [],
        }

        for res in self.results:
            (self.ctx.output_path / res.txt_file).write_text(res.sequence)
            export_data["results"].append(
                {
                    "id": res.id,
                    "cost": res.cost,
                    "binding_score": res.binding_score,
                    "sequence": res.sequence,
                    "costs": res.costs.tolist(),
                    "n_motif_hits": res.n_motif_hits,
                }
            )

        with (self.ctx.output_path / "results_metadata.json").open("w") as f:
            json.dump(export_data, f, indent=4)

    def load(self):
        path = self.ctx.output_path / "results_metadata.json"
        if not path.exists():
            return

        with path.open("r") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        self.ctx.runtime = meta.get("runtime", "-")

        self.results = []
        for d in data.get("results", []):
            res = ParetoResult(
                cost=float(d["cost"]),
                binding_score=float(d["binding_score"]),
                id=d["id"],
                url=f"{d['id']}_details.html",
                txt_file=f"{d['id']}_sequence.txt",
                fasta_file=f"{d['id']}_sequence.fa",
                sequence=d["sequence"],
                costs=np.array(d["costs"]),
                n_motif_hits=d["n_motif_hits"],
            )
            self.results.append(res)

        self.results.sort(key=lambda x: x.id)
        self.ctx.n_solutions = len(self.results)

    def _get_codon_context(self, res: ParetoResult, pos: int) -> Optional[dict]:
        for start, end in self.ctx.orfs:
            if start <= pos <= end:
                rel_pos = pos - start
                codon_start = start + ((rel_pos // 3) * 3) - 1
                return {
                    "sequence": res.sequence[codon_start : codon_start + 3],
                    "pos_in_codon": (rel_pos % 3) + 1,
                }
        return None

    def render(self):
        if not self.results:
            return

        render_pareto_front_png(self.ctx, self.results)
        render_pareto_front_html(self.ctx, self.results)

        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            for res in self.results:
                substitutions = [
                    (i + 1, cost, self._get_codon_context(res, i + 1))
                    for i, cost in enumerate(res.costs)
                    if cost > 0
                ]

                seq_with_meta = []
                for i, char in enumerate(res.sequence):
                    pos = i + 1
                    is_orf = any(s <= pos <= e for s, e in self.ctx.orfs)
                    seq_with_meta.append((char, is_orf))

                tasks.append(
                    executor.submit(
                        render_solution_html,
                        self.ctx,
                        res,
                        substitutions,
                        seq_with_meta,
                    )
                )
                tasks.append(
                    executor.submit(
                        render_heatmap_png, self.ctx, res, self.score_function.maximum
                    )
                )
            wait(tasks)

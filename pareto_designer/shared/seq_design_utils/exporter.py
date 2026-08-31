import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from pareto_designer.algorithms.seq_design.types import T_SOLUTION
from pareto_designer.bio_fetcher.fimo import get_motif_hits
from pareto_designer.bio_fetcher.paths import MOTIF_DIR
from pareto_designer.models.context import (
    ParetoResult,
    DesignContext,
    RunContext,
    FSMContext,
)
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.binding_utils import get_binding, get_total_binding
from pareto_designer.shared.parsing import write_sequence
from pareto_designer.shared.seq_design_utils.binding_metrics import binding_score_sse
from pareto_designer.views.pareto_frontier.html_exporter import (
    render_solution_html,
    render_pareto_frontier_html,
)
from pareto_designer.views.pareto_frontier.png_exporter import (
    render_heatmap_png,
    render_pareto_frontier_png,
    render_heatmap_legend,
    render_scatter_binding_scores,
)


class ParetoExporter:
    def __init__(self, ctx: DesignContext):
        self.ctx = ctx
        self._results: list[ParetoResult] = []
        self._frontier: np.ndarray = None
        self.__motif_file: Path = None
        self.output_path.mkdir(parents=True, exist_ok=True)

    @property
    def _run_ctx(self) -> RunContext:
        return self.ctx.run_ctx

    @property
    def output_path(self) -> Path:
        return self.ctx.run_ctx.output_path

    @property
    def _index_file(self) -> Path:
        return self.output_path / "results_metadata.json"

    @property
    def _score_function(self) -> ScoreFunction:
        return self.ctx.score_function

    @property
    def _fsm_ctx(self) -> FSMContext:
        return self.ctx.fsm_ctx

    @property
    def _motif_file(self) -> Path:
        if self.__motif_file is None:
            self.__motif_file = self._fsm_ctx.motif.dump("meme", MOTIF_DIR)
        return self.__motif_file

    def save(self, solutions: Iterable[tuple[str, T_SOLUTION]]):
        sorted_sols = sorted(solutions, key=lambda x: -x[1][0])
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._process_single, idx, sol)
                for idx, sol in enumerate(sorted_sols)
            ]
            done, _ = wait(futures)
            self._results = [f.result() for f in done if not f.exception()]

        self._results.sort(key=lambda x: x.id)
        self._dump()

    def _process_single(
        self, sol_idx: int, solution: tuple[str, T_SOLUTION]
    ) -> ParetoResult:
        sol_id = f"{sol_idx + 1:03d}"
        sol_fasta_file = self.output_path / f"{sol_id}_sequence.fa"
        sol_txt_file = self.output_path / f"{sol_id}_sequence.txt"
        sol_positional_objectives_file = (
            self.output_path / f"{sol_id}_sequence_positional_objectives.npy"
        )

        sequence, (f_score, binding_score) = solution
        write_sequence(sol_fasta_file, sequence, header=f"Solution {sol_id}")
        sol_txt_file.write_text(sequence)

        functional_cost = max(0.0, -f_score)
        costs = np.array(self._score_function.get_costs(sequence), dtype=float)
        n_cost_items = int(np.count_nonzero(costs > 0))
        binding = get_binding(sequence, self._fsm_ctx)
        positional_objectives = np.column_stack((costs, binding))
        np.save(sol_positional_objectives_file, positional_objectives)

        origin_binding_score = get_total_binding(
            sequence, self._fsm_ctx, use_origin=True
        )
        motif_hits = get_motif_hits(
            sol_id,
            sol_fasta_file,
            self._fsm_ctx.motif,
            self._motif_file,
            pval=self._fsm_ctx.hit_pvalue,
        )

        return ParetoResult(
            cost=float(functional_cost),
            binding_score=float(binding_score),
            origin_binding_score=float(origin_binding_score),
            id=sol_id,
            url=f"{sol_id}_details.html",
            txt_file=sol_txt_file.name,
            fasta_file=sol_fasta_file.name,
            positional_objectives_file=sol_positional_objectives_file.name,
            max_positional_cost=np.max(costs),
            min_positional_binding=np.nanmin(binding),
            max_positional_binding=np.nanmax(binding),
            sequence=sequence,
            n_cost_items=n_cost_items,
            motif_hits=motif_hits,
        )

    def _dump(self):
        data = {
            "metadata": {
                "runtime": self._run_ctx.runtime,
                "runtime_seconds": self._run_ctx.runtime_seconds,
                "n_solutions": len(self._results),
                "target_id": self._run_ctx.target_sequence_id,
                "fsm_binding_score_err": self._fsm_ctx.fsm_binding_score_err,
            },
            "results": [asdict(res) for res in self._results],
        }
        with self._index_file.open("w") as f:
            json.dump(data, f, indent=4)

    def load(self):
        if not self._index_file.exists():
            return
        with self._index_file.open("r") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        self._run_ctx.runtime = meta.get("runtime", "-")
        self._run_ctx.runtime_seconds = float(meta.get("runtime_seconds", 0.0))
        if "fsm_binding_score_err" in meta:
            self._fsm_ctx.fsm_binding_score_err = meta["fsm_binding_score_err"]
        self._results = [ParetoResult(**d) for d in data.get("results", [])]
        self._results.sort(key=lambda x: x.id)
        self._run_ctx.n_solutions = len(self._results)

    @property
    def frontier(self) -> np.ndarray:
        if self._frontier is None:
            self._frontier = np.array(
                [[r.cost, r.binding_score, r.n_motif_hits] for r in self._results],
                dtype=float,
            )
        return self._frontier

    @property
    def max_cost(self) -> float:
        if not self._results:
            return 0.0
        return np.max(self.frontier[:, 0])

    @property
    def max_positional_cost(self) -> float:
        if not self._results:
            return 0.0
        return max(r.max_positional_cost for r in self._results)

    @property
    def min_binding(self) -> float:
        if not self._results:
            return 0.0
        return np.min(self.frontier[:, 1])

    @property
    def max_binding(self) -> float:
        if not self._results:
            return 0.0
        return np.max(self.frontier[:, 1])

    @property
    def min_positional_binding(self) -> float:
        if not self._results:
            return 0.0
        return min(r.min_positional_binding for r in self._results)

    @property
    def max_positional_binding(self) -> float:
        if not self._results:
            return 0.0
        return max(r.max_positional_binding for r in self._results)

    @property
    def binding_score_sse(self) -> float:
        return binding_score_sse(self._results)

    @property
    def fsm_binding_score_err(self) -> float:
        return self._fsm_ctx.fsm_binding_score_err

    def render(
        self,
        max_cost: float,
        binding_range: tuple[float, float],
        max_positional_cost: float,
        positional_binding_range: tuple[float, float],
        hit_thresholds: list[float] | None = None,
    ):
        if not self._results:
            return

        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                executor.submit(
                    render_pareto_frontier_png,
                    self._run_ctx,
                    self._results,
                    max_cost,
                    binding_range,
                    hit_thresholds,
                ),
                executor.submit(
                    render_pareto_frontier_html, self._run_ctx, self._results
                ),
                executor.submit(
                    render_scatter_binding_scores, self._run_ctx, self._results
                ),
            ]
            tasks.extend(
                [
                    executor.submit(
                        self._render_solution,
                        res,
                        max_positional_cost,
                        positional_binding_range,
                    )
                    for res in self._results
                ]
            )

            wait(tasks)

    def render_target_sequence(
        self,
        max_positional_cost: float,
        positional_binding_range: tuple[float, float],
    ):
        seq_id = "target_sequence"
        sequence = self.ctx.target_sequence
        seq_fasta_file = self.output_path.parent / f"{seq_id}.fa"
        write_sequence(seq_fasta_file, sequence, header="Target Sequence")

        binding = get_binding(sequence, self._fsm_ctx)
        motif_hits = get_motif_hits(
            seq_id,
            seq_fasta_file,
            self._fsm_ctx.motif,
            self._motif_file,
            pval=self._fsm_ctx.hit_pvalue,
        )

        render_heatmap_png(
            self._run_ctx,
            seq_id,
            None,
            binding,
            motif_hits,
            max_positional_cost,
            positional_binding_range,
        )
        render_heatmap_legend(
            self._run_ctx,
            max_positional_cost,
            positional_binding_range,
        )

    def _render_solution(
        self,
        res: ParetoResult,
        max_positional_cost: float,
        positional_binding_range: tuple[float, float],
    ):
        positional_objectives: np.ndarray = np.load(
            self.output_path / res.positional_objectives_file
        )
        costs = positional_objectives[:, 0]
        binding = positional_objectives[:, 1]
        cost_items = [
            (i + 1, cost, self._get_codon_context(i + 1))
            for i, cost in enumerate(costs.tolist())
            if cost > 0
        ]
        seq_with_meta = self._get_seq_with_is_coding(res)
        render_solution_html(
            self._run_ctx,
            res,
            cost_items,
            seq_with_meta,
        )
        render_heatmap_png(
            self._run_ctx,
            res.id,
            costs,
            binding,
            res.motif_hits,
            max_positional_cost,
            positional_binding_range,
        )

    def _get_codon_context(self, pos: int) -> dict | None:
        for start, end in self._run_ctx.orfs:
            if start <= pos <= end:
                rel_pos = pos - start
                codon_start = start + ((rel_pos // 3) * 3) - 1
                return {
                    "start": codon_start,
                }
        return None

    def _get_seq_with_is_coding(self, res: ParetoResult) -> list[tuple[str, bool]]:
        seq_with_meta = []
        for i, char in enumerate(res.sequence):
            pos = i + 1
            is_orf = any(s <= pos <= e for s, e in self._run_ctx.orfs)
            seq_with_meta.append((char, is_orf))

        return seq_with_meta

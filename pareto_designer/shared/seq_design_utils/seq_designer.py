from loguru import logger
from pathlib import Path

from pareto_designer.algorithms.seq_design.algorithm import ParetoOptimalDesign
from pareto_designer.shared.func_cost.base_function import ScoreFunction
from pareto_designer.shared.prof import run_with_timing, format_duration
from pareto_designer.shared.seq_design_utils.score_function_builder import (
    ScoreFunctionBuilder,
)
from pareto_designer.shared.seq_design_utils.fsm_builder import FSMBuilder
from pareto_designer.algorithms.seq_design.sampling import SamplingMethod
from pareto_designer.models.context import RunContext, FSMContext, DesignContext
from pareto_designer.shared.seq_design_utils.exporter import ParetoExporter


class SequenceDesigner:
    def __init__(self):
        self._sequence_id: str = None
        self._score_function_builder: ScoreFunctionBuilder = None
        self._score_function: ScoreFunction = None
        self._fsm_builder: FSMBuilder = None
        self._fsm_ctx: FSMContext = None
        self._sampler: SamplingMethod = None

    def with_score_function_builder(
        self, score_function_builder: ScoreFunctionBuilder
    ) -> "SequenceDesigner":
        self._score_function_builder = score_function_builder
        self._score_function = None
        return self

    def with_target_sequence(self, sequence_file: Path) -> "SequenceDesigner":
        if self._sequence_id is None or self._sequence_id != sequence_file.stem:
            self._sequence_id = sequence_file.stem
            self._score_function_builder.with_target_sequence(sequence_file)
            self._score_function = None
        return self

    def with_fsm_builder(self, fsm_builder: FSMBuilder) -> "SequenceDesigner":
        self._fsm_builder = fsm_builder
        self._fsm_ctx = None
        return self

    def with_fsm_context(self, fsm_ctx: FSMContext) -> "SequenceDesigner":
        self._fsm_ctx = fsm_ctx
        return self

    def with_sampler(self, sampler: SamplingMethod) -> "SequenceDesigner":
        self._sampler = sampler
        return self

    def _build(self, dry_run: bool = False) -> DesignContext:
        if not self._score_function:
            self._score_function = self._score_function_builder.build()
        if not self._fsm_ctx:
            self._fsm_ctx = self._fsm_builder.build(dry_run)
        run_ctx = RunContext(
            target_sequence_id=self._sequence_id,
            target_sequence=self._score_function.target_sequence,
            orfs=self._score_function.orfs,
            cost_params=self._score_function.params,
            motif_id=self._fsm_ctx.motif_id,
            fsm_id=self._fsm_ctx.fsm_id,
            sampler=self._sampler,
            fsm_size=self._fsm_ctx.size,
        )
        return DesignContext(self._score_function, self._fsm_ctx, run_ctx)

    def run(self, dry_run: bool = False) -> ParetoExporter:
        ctx = self._build(dry_run)
        exporter = ParetoExporter(ctx)

        if dry_run:
            logger.info(f"Dry run: loading results for {self._sequence_id}")
            exporter.load()
        else:
            designer = ParetoOptimalDesign(ctx)
            logger.info(
                f"Running algorithm on target sequence {self._sequence_id}"
                f" and binding motif of {self._fsm_ctx.motif_id}"
                f" with sampler {ctx.run_ctx.sampler}"
                f" [n={ctx.sequence_length}, |V|={self._fsm_ctx.size}, K={ctx.run_ctx.sampler.k}]..."
            )
            solutions, duration = run_with_timing(designer.find_pareto_optimal)
            runtime = format_duration(int(duration))
            n_solutions = len(solutions)
            logger.info(f"Found {n_solutions} Pareto-optimal sequences in {runtime}")
            ctx.run_ctx.runtime = runtime
            ctx.run_ctx.n_solutions = n_solutions
            exporter.save(solutions)

        return exporter

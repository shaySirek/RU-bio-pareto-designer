from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pareto_designer.shared.seq_design_utils.run_grid import GridMode, run_design_grid


def _mock_designer(mock_designer_cls):
    mock_designer = MagicMock()
    mock_exporter = MagicMock()
    mock_exporter._results = []
    mock_designer.with_score_function_builder.return_value = mock_designer
    mock_designer.with_target_sequence.return_value = mock_designer
    mock_designer.with_fsm_context.return_value = mock_designer
    mock_designer.with_sampler.return_value = mock_designer
    mock_designer.run.return_value = mock_exporter
    mock_designer_cls.return_value = mock_designer
    return mock_designer


@patch("pareto_designer.shared.seq_design_utils.run_grid.SequenceDesigner")
def test_run_design_grid_builds_fsm_contexts(mock_designer_cls):
    mock_designer = _mock_designer(mock_designer_cls)
    fsm_ctx = MagicMock()
    fsm_ctx.reduce_fsm_by = 0.875
    fsm_ctx.fsm_id = "logexp_reduced_fsm_2048"
    fsm_ctx.motif_id = "MA0267.1"
    fsm_builder = MagicMock()
    fsm_builder.iter_contexts.return_value = [fsm_ctx]

    seq_file = Path("seq.txt")
    batches = run_design_grid(
        [seq_file],
        MagicMock(),
        fsm_builder,
        k_values=[100],
        sampler_alpha=["1.0"],
        reduce_fsm_by=[0.875],
        mode=GridMode.RUN,
    )
    assert seq_file.stem in batches
    fsm_builder.iter_contexts.assert_called_once_with(dry_run=False)
    mock_designer.run.assert_called_with(dry_run=False)


@patch("pareto_designer.shared.seq_design_utils.run_grid.SequenceDesigner")
def test_run_design_grid_reuses_fsm_contexts(mock_designer_cls):
    mock_designer = _mock_designer(mock_designer_cls)
    keep = MagicMock()
    keep.reduce_fsm_by = 0.875
    keep.fsm_id = "logexp_reduced_fsm_2048"
    skip = MagicMock()
    skip.reduce_fsm_by = 0.75
    skip.fsm_id = "logexp_reduced_fsm_4096"
    fsm_builder = MagicMock()

    run_design_grid(
        [Path("seq.txt")],
        MagicMock(),
        fsm_builder,
        k_values=[100],
        sampler_alpha=["1.0"],
        reduce_fsm_by=[0.875],
        mode=GridMode.RUN,
        fsm_contexts=[keep, skip],
    )
    fsm_builder.iter_contexts.assert_not_called()
    mock_designer.with_fsm_context.assert_called_once_with(keep)


def test_is_design_run_dir(tmp_path: Path):
    from pareto_designer.shared.seq_design_utils.pareto_utils import _is_design_run_dir

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not _is_design_run_dir(run_dir)
    (run_dir / "results_metadata.json").write_text("{}", encoding="utf-8")
    assert _is_design_run_dir(run_dir)

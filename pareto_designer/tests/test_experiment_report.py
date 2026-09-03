from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pareto_designer.models.context import ParetoResult
from pareto_designer.views.experiment_report.config import (
    ConfigError,
    alpha_comparison_groups,
    effective_grid,
    expected_runs,
    load_experiment_config,
    seq_files,
)
from pareto_designer.views.experiment_report.metrics import build_design_run_summaries
from pareto_designer.views.experiment_report.models import (
    LoadedRun,
    RunParams,
    SamplerParams,
)
from pareto_designer.views.experiment_report.paths import (
    fsm_id_for_ratio,
    fsm_size_from_id,
    parse_run_dir,
)
from pareto_designer.views.experiment_report.sweeps import (
    alpha_regime,
    sweep_membership,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "pareto_experiment_ma0267.yaml"
)


def _make_result(cost: float, binding: float, origin: float, sid: str = "001"):
    return ParetoResult(
        cost=cost,
        binding_score=binding,
        origin_binding_score=origin,
        id=sid,
        url=f"{sid}_details.html",
        txt_file=f"{sid}.txt",
        fasta_file=f"{sid}.fa",
        positional_objectives_file=f"{sid}.npy",
        max_positional_cost=1.0,
        min_positional_binding=0.0,
        max_positional_binding=1.0,
        sequence="ACGT",
        n_cost_items=1,
        motif_hits=[],
        kmer_binding_score_mse=0.01,
        kmer_binding_score_err_std=0.005,
    )


def _make_run(
    seq_id: str,
    k: int,
    alpha: float,
    log_pos: bool,
    reduce_fsm_by: float,
    fsm_size: int,
    solutions: list[ParetoResult],
) -> LoadedRun:
    fsm_id = "logexp_db_fsm" if reduce_fsm_by == 0 else f"logexp_reduced_fsm_{fsm_size}"
    params = RunParams(
        seq_id=seq_id,
        fsm_id=fsm_id,
        fsm_size=fsm_size,
        reduce_fsm_by=reduce_fsm_by,
        sampler=SamplerParams(k=k, alpha=alpha, log_pos=log_pos),
    )
    return LoadedRun(
        params=params,
        metadata={
            "n_solutions": len(solutions),
            "runtime": "0:00:01",
            "fsm_binding_score_err": 0.1,
            "db_fsm_size": 8192,
        },
        solutions=solutions,
        path=Path("dummy/results_metadata.json"),
    )


def test_load_valid_config():
    config = load_experiment_config(CONFIG_PATH)
    assert config.name == "pareto_parameter_sweep_ma0267"
    n_cells = sum(
        len(grid.k_values) * len(grid.sampler_alpha) * len(grid.reduce_fsm_by)
        for grid in (
            effective_grid(config, name) for name in ("alpha", "k", "fsm_size")
        )
    )
    assert n_cells == 15
    assert seq_files(config)


def test_alpha_comparison_groups():
    config = load_experiment_config(CONFIG_PATH)
    groups = dict(alpha_comparison_groups(config))
    assert groups["const_low"] == ("0.0", "0.5", "1.0")
    assert groups["const_high"] == ("2.0", "3.0")
    assert groups["log_pos0.5_vs_const"] == ("0.5_log_pos", "0.5", "1.0")
    assert groups["log_pos1_vs_const"] == ("1.0_log_pos", "1.0", "2.0")
    assert groups["log_pos2_vs_const"] == ("2.0_log_pos", "2.0", "3.0")
    alpha_grid = effective_grid(config, "alpha")
    assert len(alpha_grid.sampler_alpha) == 8
    assert "3.0" in alpha_grid.sampler_alpha
    assert "0.5_log_pos" in alpha_grid.sampler_alpha


def test_reject_unknown_top_level_key(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nfixed: {}\nextra: 1\nsweeps: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment_config(bad)


def test_reject_invalid_sampler_alpha(tmp_path: Path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "0.0, 0.5, 1.0", "bad_alpha, 0.5, 1.0"
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid sampler_alpha"):
        load_experiment_config(bad)


def test_fsm_id_round_trip():
    db_size = 16384
    assert fsm_size_from_id("logexp_db_fsm", db_fsm_size=db_size) == (db_size, 0.0)
    assert fsm_id_for_ratio("logexp", 0.0, db_size) == ("logexp_db_fsm", db_size)
    fsm_id, n_states = fsm_id_for_ratio("logexp", 0.875, db_size)
    assert fsm_id == "logexp_reduced_fsm_2048"
    assert fsm_size_from_id(fsm_id, db_fsm_size=db_size)[0] == n_states


def test_parse_run_dir(tmp_path: Path):
    run_dir = (
        tmp_path
        / "seq_a"
        / "transition_0.50"
        / "MA0267.1"
        / "logexp_reduced_fsm_2048"
        / "PowerLawSUS"
        / "k_100__alpha_1.0_log_pos"
    )
    run_dir.mkdir(parents=True)
    params = parse_run_dir(run_dir)
    assert params.seq_id == "seq_a"
    assert params.sampler.k == 100
    assert params.sampler.alpha == 1.0
    assert params.sampler.log_pos is True


def test_alpha_regime():
    params = RunParams(
        seq_id="s",
        fsm_id="logexp_reduced_fsm_2048",
        fsm_size=2048,
        reduce_fsm_by=0.875,
        sampler=SamplerParams(k=100, alpha=1.0, log_pos=True),
    )
    assert alpha_regime(params) == "log_pos"


def test_sweep_membership_with_config():
    config = load_experiment_config(CONFIG_PATH)
    params = RunParams(
        seq_id="s",
        fsm_id="logexp_reduced_fsm_2048",
        fsm_size=2048,
        reduce_fsm_by=0.875,
        sampler=SamplerParams(k=100, alpha=1.0, log_pos=False),
    )
    assert "alpha" in sweep_membership(params, config)


def test_build_design_run_summary_dedupes():
    sols = [_make_result(1.0, 2.0, 2.1), _make_result(2.0, 1.0, 1.1, "002")]
    run = _make_run("seq1", 100, 1.0, True, 0.875, 2048, sols)
    config = load_experiment_config(CONFIG_PATH)
    summaries = build_design_run_summaries([run], config)
    assert len(summaries) == 1
    assert "alpha" in summaries[0].sweeps
    assert math.isclose(summaries[0].fsm_binding_score_err, 0.1 / 8192)


def test_sort_design_runs():
    from pareto_designer.views.experiment_report.metrics import (
        design_run_summary,
        sort_design_runs,
    )

    runs = [
        design_run_summary(
            _make_run("b", 50, 1.0, True, 0.875, 2048, [_make_result(1, 2, 2)]),
            ["k"],
        ),
        design_run_summary(
            _make_run("a", 100, 1.0, True, 0.875, 4096, [_make_result(1, 2, 2)]),
            ["k"],
        ),
        design_run_summary(
            _make_run("a", 150, 1.0, True, 0.875, 2048, [_make_result(1, 2, 2)]),
            ["k"],
        ),
    ]
    ordered = sort_design_runs(runs)
    assert [r.seq_id for r in ordered] == ["a", "a", "b"]
    assert ordered[0].fsm_size == 4096
    assert ordered[1].k == 150


def _kmer_run(tmp_path: Path) -> tuple[LoadedRun, ParetoResult, Path]:
    import numpy as np

    run_dir = (
        tmp_path
        / "seq"
        / "cost"
        / "MA0267.1"
        / "logexp_reduced_fsm_2048"
        / "PowerLawSUS"
        / "k_100__alpha_1.0"
    )
    run_dir.mkdir(parents=True)
    np.save(
        run_dir / "001.npy",
        np.column_stack((np.zeros(4), np.array([1.5, 2.5, 2.0, np.nan]))),
    )
    meta_path = run_dir / "results_metadata.json"
    meta_path.write_text("{}", encoding="utf-8")
    sol = _make_result(1.0, 2.0, 2.1)
    sol.positional_objectives_file = "001.npy"
    run = _make_run("seq", 100, 1.0, False, 0.875, 2048, [sol])
    run.path = meta_path
    return run, sol, meta_path


def test_fill_kmer_binding(tmp_path: Path):
    import numpy as np

    from pareto_designer.algorithms.spaces import ExpSpace
    from pareto_designer.shared.seq_design_utils.binding_metrics import (
        kmer_binding_score_mse,
    )
    from pareto_designer.views.experiment_report.kmer_binding import fill_kmer_binding

    run, sol, meta_path = _kmer_run(tmp_path)
    with patch(
        "pareto_designer.views.experiment_report.kmer_binding.origin_map_for_motif",
        return_value=(2, {"AC": 1.0, "CG": 2.0, "GT": 3.0}),
    ):
        fill_kmer_binding([run])
    expected = kmer_binding_score_mse(
        np.array([1.5, 2.5, 2.0, np.nan]),
        np.array([1.0, 2.0, 3.0]),
        ExpSpace,
    )
    assert math.isclose(sol.kmer_binding_score_mse, expected.mse)
    assert "kmer_binding_score_mse" not in meta_path.read_text(encoding="utf-8")


def test_fill_kmer_binding_uses_fsm_context(tmp_path: Path):
    from pareto_designer.views.experiment_report.kmer_binding import fill_kmer_binding

    run, sol, _ = _kmer_run(tmp_path)
    ctx = MagicMock()
    ctx.motif_id = "MA0267.1"
    ctx.motif_length = 2
    ctx.origin_binding_score_map = {"AC": 1.0, "CG": 2.0, "GT": 3.0}
    with patch(
        "pareto_designer.views.experiment_report.kmer_binding.origin_map_for_motif"
    ) as mock_origin:
        fill_kmer_binding([run], fsm_contexts=[ctx])
    mock_origin.assert_not_called()
    assert math.isfinite(sol.kmer_binding_score_mse)


def test_expected_runs_uses_passed_fsm_contexts():
    config = load_experiment_config(CONFIG_PATH)
    contexts = []
    for ratio, fsm_id, size in (
        (0.0, "logexp_db_fsm", 16384),
        (0.75, "logexp_reduced_fsm_4096", 4096),
        (0.875, "logexp_reduced_fsm_2048", 2048),
        (0.9375, "logexp_reduced_fsm_1024", 1024),
    ):
        ctx = MagicMock()
        ctx.reduce_fsm_by = ratio
        ctx.fsm_id = fsm_id
        ctx.size = size
        contexts.append(ctx)
    with patch(
        "pareto_designer.views.experiment_report.config.db_fsm_state_count_for_motif"
    ) as mock_db:
        runs = expected_runs(config, fsm_contexts=contexts)
    mock_db.assert_not_called()
    assert {r.params.fsm_id for r in runs if r.sweep == "alpha"} == {
        "logexp_reduced_fsm_2048"
    }
    assert "logexp_db_fsm" in {r.params.fsm_id for r in runs if r.sweep == "fsm_size"}
    assert all("transition_0.50" in str(r.metadata_path) for r in runs)


def test_expected_runs_derives_fsm_ids_without_builder():
    config = load_experiment_config(CONFIG_PATH)
    with patch(
        "pareto_designer.views.experiment_report.config.db_fsm_state_count_for_motif",
        return_value=16384,
    ):
        runs = expected_runs(config)
    assert len(runs) == 15 * len(seq_files(config))
    assert {r.params.fsm_id for r in runs if r.sweep == "fsm_size"} == {
        "logexp_db_fsm",
        "logexp_reduced_fsm_4096",
        "logexp_reduced_fsm_2048",
        "logexp_reduced_fsm_1024",
    }


def test_alpha_roi_boxplot_renders_png(tmp_path):
    from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
        alpha_roi_boxplot_filename,
        render_alpha_roi_boxplot,
    )

    roi_by_alpha = {
        "1.0": [(10.0, 5.0), (20.0, 4.0), (30.0, 3.5)],
        "2.0": [(12.0, 4.8), (22.0, 3.9)],
    }
    out = tmp_path / alpha_roi_boxplot_filename(100)
    path = render_alpha_roi_boxplot("seq1", roi_by_alpha, out)
    assert path == out
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pareto_designer.models.context import ParetoResult
from pareto_designer.shared.seq_design_utils.run_paths import format_cost_params_str
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


def _make_result(
    cost: float,
    binding: float,
    origin: float,
    sid: str = "001",
    *,
    hits: list[tuple[int, int]] | None = None,
    n_nonsyn: int = 0,
):
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
        motif_hits=hits or [],
        n_nonsyn=n_nonsyn,
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


def _n_grid_cells(config) -> int:
    return sum(
        len(grid.k_values) * len(grid.sampler_alpha) * len(grid.reduce_fsm_by)
        for grid in (
            effective_grid(config, name) for name in ("alpha", "k", "fsm_size")
        )
    )


def test_load_valid_config():
    config = load_experiment_config(CONFIG_PATH)
    assert config.name == "pareto_parameter_sweep_ma0267"
    for name in ("alpha", "k", "fsm_size"):
        grid = effective_grid(config, name)
        assert grid.k_values
        assert grid.sampler_alpha
        assert grid.reduce_fsm_by
    assert _n_grid_cells(config) > 0
    assert seq_files(config)


def test_alpha_comparison_groups():
    config = load_experiment_config(CONFIG_PATH)
    groups = dict(alpha_comparison_groups(config))
    raw_groups = config.sweeps["alpha"]["comparison_groups"]
    allowed = set(effective_grid(config, "alpha").sampler_alpha)
    assert groups
    assert set(groups) == set(raw_groups)
    for name, alphas in groups.items():
        assert name.strip()
        assert alphas
        assert alphas == tuple(str(item) for item in raw_groups[name])
        assert set(alphas) <= allowed


def test_reject_unknown_top_level_key(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nfixed: {}\nextra: 1\nsweeps: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment_config(bad)


def test_reject_invalid_sampler_alpha(tmp_path: Path):
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["sweeps"]["alpha"]["vary"]["sampler_alpha"] = ["bad_alpha"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid sampler_alpha"):
        load_experiment_config(bad)


def test_fsm_id_round_trip():
    db_size = 16384
    assert fsm_size_from_id("logexp_db_fsm", db_fsm_size=db_size) == (db_size, 0.0)
    assert fsm_id_for_ratio("logexp", 0.0, db_size) == ("logexp_db_fsm", db_size)
    fsm_id, n_states = fsm_id_for_ratio("logexp", 0.875, db_size)
    assert fsm_id == "logexp_reduced_fsm_2048"
    assert fsm_size_from_id(fsm_id, db_fsm_size=db_size)[0] == n_states


def test_format_cost_params_str():
    assert (
        format_cost_params_str({"alpha": 0.5, "beta": 1.0, "w": 500.0})
        == "alpha0.5_beta1.0_w500.0"
    )


def test_parse_run_dir(tmp_path: Path):
    run_dir = (
        tmp_path
        / "seq_a"
        / "alpha0.5_beta1.0_w500.0"
        / "MA0267.1"
        / "logexp_reduced_fsm_2048"
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
    assert ordered[0].k == 100
    assert ordered[0].fsm_size == 4096
    assert ordered[1].k == 150


def _kmer_run(tmp_path: Path) -> tuple[LoadedRun, ParetoResult, Path]:
    import numpy as np

    run_dir = (
        tmp_path
        / "seq"
        / "alpha0.5_beta1.0_w500.0"
        / "MA0267.1"
        / "logexp_reduced_fsm_2048"
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
    assert all("alpha0.5_beta1.0_w500.0" in str(r.metadata_path) for r in runs)


def test_expected_runs_derives_fsm_ids_without_builder():
    config = load_experiment_config(CONFIG_PATH)
    with patch(
        "pareto_designer.views.experiment_report.config.db_fsm_state_count_for_motif",
        return_value=16384,
    ):
        runs = expected_runs(config)
    assert len(runs) == _n_grid_cells(config) * len(seq_files(config))
    assert {r.params.fsm_id for r in runs if r.sweep == "fsm_size"} == {
        "logexp_db_fsm",
        "logexp_reduced_fsm_4096",
        "logexp_reduced_fsm_2048",
        "logexp_reduced_fsm_1024",
    }


def test_alpha_group_cost_hist(tmp_path):
    from pareto_designer.shared.seq_design_utils.pareto_utils import (
        sweep_alpha_cost_hist_filename,
    )
    from pareto_designer.views.pareto_frontier.png_exporter import (
        cost_histogram_lines,
        render_cost_hist_lines,
    )

    costs = {"a": [1.0, 2.0, 2.0], "b": [2.0, 3.0]}
    hist = cost_histogram_lines({**costs, "empty": []})
    assert hist is not None
    grid, dens = hist
    assert "empty" not in dens and len(grid) >= 64 and dens["a"].min() >= 0
    out = tmp_path / sweep_alpha_cost_hist_filename(100, "g")
    assert render_cost_hist_lines(costs, out) == out


def test_k_sweep_frontier_filenames():
    from types import SimpleNamespace

    from pareto_designer.shared.seq_design_utils.pareto_utils import (
        sweep_pareto_frontiers_filename,
    )
    from pareto_designer.views.pareto_frontier.png_exporter import FrontierPlotStyle

    grid = SimpleNamespace(sampler_alpha=["4.0"], k_values=[100])
    assert (
        sweep_pareto_frontiers_filename("k", grid)
        == "sweep_K_alpha_4.0_pareto_frontiers.png"
    )
    assert (
        sweep_pareto_frontiers_filename("k", grid, plot_style=FrontierPlotStyle.POINTS)
        == "sweep_K_alpha_4.0_pareto_frontiers.png"
    )
    assert (
        sweep_pareto_frontiers_filename("k", grid, plot_style=FrontierPlotStyle.LINES)
        == "sweep_K_alpha_4.0_pareto_frontiers_lines.png"
    )
    assert (
        sweep_pareto_frontiers_filename(
            "k", grid, plot_style=FrontierPlotStyle.LINES_ANNO
        )
        == "sweep_K_alpha_4.0_pareto_frontiers_lines_anno.png"
    )


def test_alpha_group_roi_boxwhisker(tmp_path):
    from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
        alpha_roi_boxplot_filename,
        render_roi_boxplot,
    )

    out = tmp_path / alpha_roi_boxplot_filename(100, "g")
    assert out.name == "sweep_alpha_K100_g_roi_boxwhisker.png"
    assert (
        render_roi_boxplot(
            {"1.0": [(10.0, 5.0)], "2.0": [(12.0, 4.0)]},
            out,
            xlabel="alpha",
        )
        == out
    )


def test_alpha_group_roi_violin(tmp_path):
    from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
        _roi_series_legend_label,
        alpha_roi_violin_filename,
        render_roi_violin,
    )

    out = tmp_path / alpha_roi_violin_filename(100, "g")
    assert out.name == "sweep_alpha_K100_g_roi_violin.png"
    assert _roi_series_legend_label("1.0", "α") == "α=1.0"
    assert (
        render_roi_violin(
            {
                "1.0": [(10.0, 5.0), (11.0, 4.5), (12.0, 4.0)],
                "2.0": [(12.0, 4.0), (13.0, 3.5), (14.0, 3.0)],
            },
            out,
            series_prefix="α",
        )
        == out
    )


def test_k_sweep_roi_boxwhisker(tmp_path):
    from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
        k_roi_boxplot_filename,
        render_roi_boxplot,
    )

    out = tmp_path / k_roi_boxplot_filename("4.0")
    assert out.name == "sweep_K_alpha_4.0_roi_boxwhisker.png"
    assert (
        render_roi_boxplot(
            {"50": [(10.0, 5.0)], "100": [(12.0, 4.0)]},
            out,
            label_order=("50", "100", "150"),
            xlabel="K",
        )
        == out
    )


def test_k_sweep_roi_violin(tmp_path):
    from pareto_designer.shared.seq_design_utils.solution_quality.plots import (
        _roi_series_legend_label,
        k_roi_violin_filename,
        render_roi_violin,
    )

    out = tmp_path / k_roi_violin_filename("4.0")
    assert out.name == "sweep_K_alpha_4.0_roi_violin.png"
    assert _roi_series_legend_label("50", "K") == "K=50"
    assert (
        render_roi_violin(
            {
                "50": [(10.0, 5.0), (11.0, 4.8), (12.0, 4.5)],
                "100": [(12.0, 4.0), (13.0, 3.8), (14.0, 3.5)],
            },
            out,
            label_order=("50", "100", "150"),
            series_prefix="K",
        )
        == out
    )

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import yaml

from pareto_designer.algorithms.spaces import ScoreSpaceOption
from pareto_designer.algorithms.seq_design.sampling import PowerLawSUS
from pareto_designer.models.context import FSMContext
from pareto_designer.shared.seq_design_utils.pareto_utils import parse_sampler_alpha
from pareto_designer.shared.seq_design_utils.run_paths import (
    format_cost_params_str,
    metadata_path,
)
from pareto_designer.views.experiment_report.models import (
    ExpectedRun,
    ExperimentConfig,
    RunParams,
    SamplerParams,
    SweepGrid,
)
from pareto_designer.views.experiment_report.paths import (
    db_fsm_state_count_for_motif,
    fsm_id_for_ratio,
)

SAMPLER_ALPHA_RE = re.compile(r"^\d+(\.\d+)?(_log_pos)?$")

TOP_LEVEL_KEYS = frozenset({"name", "fixed", "sweeps"})
FIXED_KEYS = frozenset(
    {
        "target_sequences",
        "matrix_id",
        "codon_usage",
        "binding_score_space",
        "hit_pval",
        "cost_params",
        "results_root",
    }
)
SWEEP_NAMES = frozenset({"alpha", "k", "fsm_size"})
ALPHA_FIXED_KEYS = frozenset({"k", "reduce_fsm_by"})
K_FIXED_KEYS = frozenset({"reduce_fsm_by", "sampler_alpha"})
FSM_FIXED_KEYS = frozenset({"k", "sampler_alpha"})
COST_PARAM_KEYS = frozenset({"alpha", "beta", "w"})


class ConfigError(ValueError):
    pass


def _reject_unknown(data: dict, allowed: frozenset[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{context}: unknown keys {sorted(unknown)}")


def _require_type(value, expected_type, name: str):
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{name} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _validate_sampler_alpha(values: list) -> list[str]:
    _require_type(values, list, "sampler_alpha")
    if not values:
        raise ConfigError("sampler_alpha must be a non-empty list")
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        s = str(item)
        if not SAMPLER_ALPHA_RE.match(s):
            raise ConfigError(f"invalid sampler_alpha entry: {s!r}")
        if s in seen:
            raise ConfigError(f"duplicate sampler_alpha entry: {s!r}")
        seen.add(s)
        parse_sampler_alpha(s)
        out.append(s)
    return out


def _validate_reduce_fsm_by(values: list) -> list[float]:
    _require_type(values, list, "reduce_fsm_by")
    if not values:
        raise ConfigError("reduce_fsm_by must be a non-empty list")
    out: list[float] = []
    for item in values:
        ratio = float(item)
        if ratio < 0 or ratio >= 1:
            raise ConfigError(f"reduce_fsm_by must be in [0, 1), got {ratio}")
        out.append(ratio)
    return out


def _validate_k_values(values: list) -> list[int]:
    _require_type(values, list, "k")
    if not values:
        raise ConfigError("k must be a non-empty list")
    out: list[int] = []
    for item in values:
        k = int(item)
        if k <= 0:
            raise ConfigError(f"k must be > 0, got {k}")
        out.append(k)
    return out


def load_experiment_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("YAML root must be a mapping")
    _reject_unknown(raw, TOP_LEVEL_KEYS, "top level")

    name = _require_type(raw["name"], str, "name")
    if not name.strip():
        raise ConfigError("name must be non-empty")

    fixed = _require_type(raw["fixed"], dict, "fixed")
    _reject_unknown(fixed, FIXED_KEYS, "fixed")

    target_sequences = Path(str(fixed["target_sequences"]))
    if not target_sequences.exists():
        raise ConfigError(f"target_sequences path does not exist: {target_sequences}")

    matrix_id = _require_type(fixed["matrix_id"], str, "matrix_id")
    if not matrix_id.strip():
        raise ConfigError("matrix_id must be non-empty")

    codon_usage = Path(str(fixed["codon_usage"]))
    if not codon_usage.exists():
        raise ConfigError(f"codon_usage path does not exist: {codon_usage}")

    binding_score_space = str(fixed["binding_score_space"])
    try:
        ScoreSpaceOption(binding_score_space)
    except ValueError as exc:
        valid = [x.value for x in ScoreSpaceOption]
        raise ConfigError(
            f"binding_score_space must be one of {valid}, got {binding_score_space!r}"
        ) from exc

    hit_pval = float(fixed.get("hit_pval", 0.002))
    if not (0 < hit_pval <= 1):
        raise ConfigError(f"hit_pval must be in (0, 1], got {hit_pval}")

    cost_params = _require_type(fixed["cost_params"], dict, "cost_params")
    _reject_unknown(cost_params, COST_PARAM_KEYS, "cost_params")
    for key in COST_PARAM_KEYS:
        if key not in cost_params:
            raise ConfigError(f"cost_params missing required key: {key}")
        val = float(cost_params[key])
        if val <= 0:
            raise ConfigError(f"cost_params.{key} must be > 0, got {val}")
        cost_params[key] = val

    results_root = Path(str(fixed.get("results_root", "designer_results")))

    sweeps = _require_type(raw["sweeps"], dict, "sweeps")
    _reject_unknown(sweeps, SWEEP_NAMES, "sweeps")
    for sweep_name in SWEEP_NAMES:
        if sweep_name not in sweeps:
            raise ConfigError(f"missing required sweep: {sweep_name}")
        _validate_sweep_entry(sweep_name, sweeps[sweep_name])

    normalized_fixed = {
        **fixed,
        "target_sequences": str(target_sequences),
        "codon_usage": str(codon_usage),
        "hit_pval": hit_pval,
        "cost_params": cost_params,
        "results_root": str(results_root),
    }

    return ExperimentConfig(
        name=name,
        fixed=normalized_fixed,
        sweeps=sweeps,
        config_path=path.resolve(),
    )


def _validate_alpha_comparison_groups(
    groups: dict,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    _require_type(groups, dict, "sweeps.alpha.comparison_groups")
    if not groups:
        raise ConfigError("sweeps.alpha.comparison_groups must be non-empty")
    result: list[tuple[str, tuple[str, ...]]] = []
    for name, alphas in groups.items():
        group_name = _require_type(name, str, "comparison group name")
        if not group_name.strip():
            raise ConfigError("comparison group name must be non-empty")
        validated = _validate_sampler_alpha(alphas)
        result.append((group_name, tuple(validated)))
    return tuple(result)


def _validate_sweep_entry(name: str, entry: dict) -> None:
    allowed_keys = frozenset({"fixed", "vary"})
    if name == "alpha":
        allowed_keys = frozenset({"fixed", "vary", "comparison_groups"})
    _require_type(entry, dict, f"sweeps.{name}")
    _reject_unknown(entry, allowed_keys, f"sweeps.{name}")
    fixed = _require_type(entry["fixed"], dict, f"sweeps.{name}.fixed")
    vary = _require_type(entry["vary"], dict, f"sweeps.{name}.vary")
    if len(vary) != 1:
        raise ConfigError(f"sweeps.{name}.vary must contain exactly one key")

    if name == "alpha":
        _reject_unknown(fixed, ALPHA_FIXED_KEYS, f"sweeps.{name}.fixed")
        _reject_unknown(vary, frozenset({"sampler_alpha"}), f"sweeps.{name}.vary")
        k = int(fixed["k"])
        if k <= 0:
            raise ConfigError(f"sweeps.alpha.fixed.k must be > 0, got {k}")
        ratio = float(fixed["reduce_fsm_by"])
        if ratio < 0 or ratio >= 1:
            raise ConfigError("sweeps.alpha.fixed.reduce_fsm_by must be in [0, 1)")
        _validate_sampler_alpha(vary["sampler_alpha"])
        if "comparison_groups" not in entry:
            raise ConfigError("sweeps.alpha.comparison_groups is required")
        _validate_alpha_comparison_groups(entry["comparison_groups"])
    elif name == "k":
        _reject_unknown(fixed, K_FIXED_KEYS, f"sweeps.{name}.fixed")
        _reject_unknown(vary, frozenset({"k"}), f"sweeps.{name}.vary")
        _validate_sampler_alpha([fixed["sampler_alpha"]])
        ratio = float(fixed["reduce_fsm_by"])
        if ratio < 0 or ratio >= 1:
            raise ConfigError("sweeps.k.fixed.reduce_fsm_by must be in [0, 1)")
        _validate_k_values(vary["k"])
    elif name == "fsm_size":
        _reject_unknown(fixed, FSM_FIXED_KEYS, f"sweeps.{name}.fixed")
        _reject_unknown(vary, frozenset({"reduce_fsm_by"}), f"sweeps.{name}.vary")
        k = int(fixed["k"])
        if k <= 0:
            raise ConfigError(f"sweeps.fsm_size.fixed.k must be > 0, got {k}")
        _validate_sampler_alpha([fixed["sampler_alpha"]])
        _validate_reduce_fsm_by(vary["reduce_fsm_by"])


ExperimentConfig.from_yaml = classmethod(
    lambda cls, path: load_experiment_config(Path(path))  # type: ignore[method-assign]
)


def effective_grid(config: ExperimentConfig, sweep_name: str) -> SweepGrid:
    entry = config.sweeps[sweep_name]
    fixed = entry["fixed"]
    vary = entry["vary"]
    if sweep_name == "alpha":
        return SweepGrid(
            k_values=[int(fixed["k"])],
            sampler_alpha=_validate_sampler_alpha(vary["sampler_alpha"]),
            reduce_fsm_by=[float(fixed["reduce_fsm_by"])],
        )
    if sweep_name == "k":
        return SweepGrid(
            k_values=_validate_k_values(vary["k"]),
            sampler_alpha=[str(fixed["sampler_alpha"])],
            reduce_fsm_by=[float(fixed["reduce_fsm_by"])],
        )
    return SweepGrid(
        k_values=[int(fixed["k"])],
        sampler_alpha=[str(fixed["sampler_alpha"])],
        reduce_fsm_by=_validate_reduce_fsm_by(vary["reduce_fsm_by"]),
    )


ExperimentConfig.effective_grid = effective_grid  # type: ignore[method-assign]


def alpha_comparison_groups(
    config: ExperimentConfig,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    entry = config.sweeps["alpha"]
    if "comparison_groups" not in entry:
        raise ConfigError("sweeps.alpha.comparison_groups is required")
    return _validate_alpha_comparison_groups(entry["comparison_groups"])


ExperimentConfig.alpha_comparison_groups = alpha_comparison_groups  # type: ignore[method-assign]


def seq_files(config: ExperimentConfig) -> list[Path]:
    target = Path(config.fixed["target_sequences"])
    if target.is_file():
        return [target.resolve()]
    return sorted(target.glob("*.txt"))


ExperimentConfig.seq_files = seq_files  # type: ignore[method-assign]


def report_output_path(config: ExperimentConfig) -> Path:
    root = Path(config.fixed["results_root"])
    return root / "pareto_experiment_report.xlsx"


def nonsyn_w(config: ExperimentConfig | None) -> float | None:
    if config is None:
        return None
    w = config.fixed.get("cost_params", {}).get("w")
    return float(w) if w is not None else None


ExperimentConfig.report_output_path = report_output_path  # type: ignore[method-assign]


def _cost_params_str(config: ExperimentConfig) -> str:
    cost_params = config.fixed["cost_params"]
    return format_cost_params_str(
        {
            "Transition": float(cost_params["alpha"]),
            "Transversion": float(cost_params["beta"]),
            "Non-synonymous codon": float(cost_params["w"]),
        }
    )


def _fsm_info_by_ratio(
    config: ExperimentConfig,
    fsm_contexts: Sequence[FSMContext] | None = None,
) -> dict[float, tuple[str, int]]:
    by_ratio: dict[float, tuple[str, int]] = {}
    if fsm_contexts:
        for ctx in fsm_contexts:
            by_ratio[ctx.reduce_fsm_by] = (ctx.fsm_id, ctx.size)

    needed: set[float] = set()
    for sweep_name in SWEEP_NAMES:
        needed.update(effective_grid(config, sweep_name).reduce_fsm_by)
    missing = [ratio for ratio in needed if ratio not in by_ratio]
    if not missing:
        return by_ratio

    space = str(config.fixed["binding_score_space"])
    db_size = db_fsm_state_count_for_motif(config.fixed["matrix_id"])
    for ratio in missing:
        by_ratio[ratio] = fsm_id_for_ratio(space, ratio, db_size)
    return by_ratio


def _run_params(
    seq_id: str,
    k: int,
    alpha_str: str,
    fsm_id: str,
    fsm_size: int,
    reduce_fsm_by: float,
) -> RunParams:
    alpha, log_pos = parse_sampler_alpha(alpha_str)
    return RunParams(
        seq_id=seq_id,
        fsm_id=fsm_id,
        fsm_size=fsm_size,
        reduce_fsm_by=reduce_fsm_by,
        sampler=SamplerParams(k=k, alpha=alpha, log_pos=log_pos),
    )


def expected_runs(
    config: ExperimentConfig,
    fsm_contexts: Sequence[FSMContext] | None = None,
) -> list[ExpectedRun]:
    fsm_by_ratio = _fsm_info_by_ratio(config, fsm_contexts)
    seq_paths = seq_files(config)
    cost_params_str = _cost_params_str(config)
    root = Path(config.fixed["results_root"])
    matrix_id = config.fixed["matrix_id"]
    runs: list[ExpectedRun] = []

    for sweep_name in ("alpha", "k", "fsm_size"):
        grid = effective_grid(config, sweep_name)
        for seq_file in seq_paths:
            seq_id = seq_file.stem
            for k, alpha_str, ratio in product(
                grid.k_values, grid.sampler_alpha, grid.reduce_fsm_by
            ):
                fsm_id, fsm_size = fsm_by_ratio[ratio]
                params = _run_params(seq_id, k, alpha_str, fsm_id, fsm_size, ratio)
                sampler = PowerLawSUS(
                    params.sampler.k,
                    params.sampler.alpha,
                    params.sampler.log_pos,
                )
                runs.append(
                    ExpectedRun(
                        seq_id=seq_id,
                        sweep=sweep_name,
                        params=params,
                        metadata_path=metadata_path(
                            root,
                            seq_id,
                            cost_params_str,
                            matrix_id,
                            fsm_id,
                            sampler,
                        ),
                    )
                )
    return runs


ExperimentConfig.expected_runs = expected_runs  # type: ignore[method-assign]

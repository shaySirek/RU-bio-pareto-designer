from __future__ import annotations

import re
from functools import lru_cache
from math import ceil
from pathlib import Path

from pareto_designer.bio_fetcher.motif import JASPAR_DB
from pareto_designer.views.experiment_report.models import RunParams, SamplerParams

SAMPLER_DIR_RE = re.compile(
    r"^k_(?P<k>\d+)__alpha_(?P<alpha>\d+(?:\.\d+)?)(?P<log_pos>_log_pos)?$"
)


def resolve_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive:
        path_str = str(resolved)
        if not path_str.startswith("\\\\?\\"):
            return Path("\\\\?\\" + path_str)
    return resolved


def metadata_path_for_run(run_dir: Path) -> Path:
    return run_dir / "results_metadata.json"


def db_fsm_state_count(alphabet_size: int, motif_length: int) -> int:
    return alphabet_size**motif_length


@lru_cache(maxsize=32)
def db_fsm_state_count_for_motif(matrix_id: str) -> int:
    motif = JASPAR_DB().get_motif(matrix_id)
    return db_fsm_state_count(len(motif.alphabet), motif.length)


def fsm_size_from_id(fsm_id: str, *, db_fsm_size: int) -> tuple[int, float]:
    if fsm_id.endswith("_db_fsm"):
        return db_fsm_size, 0.0
    match = re.search(r"_reduced_fsm_(\d+)$", fsm_id)
    if match:
        size = int(match.group(1))
        return size, 1.0 - size / db_fsm_size
    return 0, 0.0


def fsm_id_for_ratio(space: str, ratio: float, db_fsm_size: int) -> tuple[str, int]:
    if not ratio:
        return f"{space}_db_fsm", db_fsm_size
    n_states = ceil((1 - ratio) * db_fsm_size)
    return f"{space}_reduced_fsm_{n_states}", n_states


def parse_run_dir(run_dir: Path) -> RunParams:
    parts = run_dir.resolve().parts
    try:
        sampler_idx = parts.index("PowerLawSUS")
    except ValueError as exc:
        raise ValueError(f"not a design run directory: {run_dir}") from exc

    sampler_dir = parts[sampler_idx + 1]
    match = SAMPLER_DIR_RE.match(sampler_dir)
    if not match:
        raise ValueError(f"cannot parse sampler directory: {sampler_dir}")

    k = int(match.group("k"))
    alpha = float(match.group("alpha"))
    log_pos = match.group("log_pos") is not None
    fsm_id = parts[sampler_idx - 1]
    motif_id = parts[sampler_idx - 2]
    seq_id = parts[sampler_idx - 4]

    db_size = db_fsm_state_count_for_motif(motif_id)
    fsm_size, reduce_fsm_by = fsm_size_from_id(fsm_id, db_fsm_size=db_size)
    return RunParams(
        seq_id=seq_id,
        fsm_id=fsm_id,
        fsm_size=fsm_size,
        reduce_fsm_by=reduce_fsm_by,
        sampler=SamplerParams(k=k, alpha=alpha, log_pos=log_pos),
    )


def motif_id_from_run_dir(path: Path) -> str:
    parts = Path(path).resolve().parts
    try:
        sampler_idx = parts.index("PowerLawSUS")
    except ValueError as exc:
        raise ValueError(f"cannot parse motif id from {path}") from exc
    return parts[sampler_idx - 2]


def parse_metadata_path(metadata_path: Path) -> RunParams:
    return parse_run_dir(metadata_path.parent)

from typing import NamedTuple
from pathlib import Path

import numpy as np


class RunContext(NamedTuple):
    target_sequence: str
    orfs: list[tuple[int, int]]
    cost_params: dict[str, float]
    motif_id: str
    fsm_size: int
    solutions_limit: int
    runtime: str
    output_path: Path


class ParetoResult(NamedTuple):
    cost: float
    binding_score: float
    id: str
    url: str
    sequence: str
    costs: np.ndarray

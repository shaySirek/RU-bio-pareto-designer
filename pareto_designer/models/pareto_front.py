from typing import NamedTuple
from pathlib import Path

import numpy as np


class RunContext(NamedTuple):
    target_sequence_id: str
    target_sequence: str
    orfs: list[tuple[int, int]]
    cost_params: dict[str, float]
    motif_id: str
    fsm_id: str
    fsm_size: int
    solutions_limit: int
    n_solutions: int
    runtime: str

    @property
    def output_path(self) -> Path:
        return (
            Path("designer_results")
            / self.target_sequence_id
            / self.motif_id
            / self.fsm_id
            / f"solutions_limit_{self.solutions_limit}"
        )


class ParetoResult(NamedTuple):
    cost: float
    binding_score: float
    id: str
    url: str
    txt_file: str
    fasta_file: str
    sequence: str
    costs: np.ndarray
    n_motif_hits: int

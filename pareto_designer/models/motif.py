from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional
from pathlib import Path

from Bio import motifs
from Bio.motifs.matrix import PositionWeightMatrix, PositionSpecificScoringMatrix


class StrandForBindingScore(Enum):
    Forward = "forward"
    Backward = "backward"
    Double = "double"

    def get_description(self) -> str:
        if self == StrandForBindingScore.Double:
            return "double-stranded (max)"
        else:
            return f"{self.value} strand"

    def get_label(self, eff: float) -> str:
        return {
            StrandForBindingScore.Forward: "B   ",
            StrandForBindingScore.Backward: "B$_{RC}$ ",
            StrandForBindingScore.Double: "B$_{ds}$ ",
        }[self] + f" efficiency={eff:.3f}"

    def get_color(self) -> str:
        return {
            StrandForBindingScore.Forward: "blue",
            StrandForBindingScore.Backward: "red",
            StrandForBindingScore.Double: "black",
        }[self]


@dataclass
class BindingMotif:
    matrix_id: str  # identifier

    pseudocounts: Union[dict[str, int], int] = 1
    background: Union[dict[str, float], float] = 0.5

    # mutation options
    reverse_complement: bool = False
    counts_override: Optional[dict[tuple[str, int], int]] = None
    consensus_override: Optional[dict[int, str]] = None

    # derived and calculated fields
    motif: motifs.Motif = field(init=False)
    alphabet: list[str] = field(init=False)
    length: int = field(init=False)
    n_sequences: int = field(init=False)
    pwm: PositionWeightMatrix = field(init=False)
    pssm: PositionSpecificScoringMatrix = field(init=False)
    pssm_backward: PositionSpecificScoringMatrix = field(
        init=False
    )  # used for calculating score
    ic_vector: list[float] = field(init=False)
    var_vector: list[float] = field(init=False)
    mean: float = field(init=False)
    std: float = field(init=False)

    def write(self, fmt: str, f) -> None:
        fmt = fmt.lower()
        if fmt == "meme":
            alphabet = self.motif.alphabet
            background = dict(self.motif.background)
            f.write("MEME version 5.0.0\n\n")
            f.write(f"ALPHABET= {alphabet}\n\n")
            f.write("Background letter frequencies\n")
            f.write(
                " ".join(f"{base} {background[base]:.6f}" for base in alphabet) + "\n\n"
            )
            f.write("strands: + -\n\n")
            f.write(f"MOTIF {self.matrix_id}\n\n")
            f.write(
                f"letter-probability matrix: alength= {len(alphabet)} w= {self.length} nsites= {self.n_sequences} E= 0\n"
            )
            for i in range(self.length):
                f.write(
                    " ".join(f"{self.pwm[base][i]:.6f}" for base in alphabet) + "\n"
                )
        else:
            f.write(motifs.write([self.motif], fmt))

    def dump(self, fmt: str, base_dir: Path) -> Path:
        motif_dir = base_dir / self.matrix_id
        motif_dir.mkdir(parents=True, exist_ok=True)
        motif_file = motif_dir / f"motif.{fmt}"
        with open(motif_file, "w") as f:
            self.write(fmt, f)
        return motif_file

    def __str__(self) -> str:
        return f"{self.matrix_id} (m={self.length})"

    @property
    def total_ic(self) -> float:
        return self.mean

    @property
    def avg_ic(self) -> float:
        return self.mean / self.length

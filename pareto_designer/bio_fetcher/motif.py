from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional
from itertools import product
from operator import itemgetter
import math
from pathlib import Path

from loguru import logger
from pyjaspar import jaspardb
from Bio import motifs
from Bio.motifs.matrix import PositionWeightMatrix, PositionSpecificScoringMatrix
import numpy as np


class JASPAR_DB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, release: str = "JASPAR2024"):
        self._jdb_obj = jaspardb(release=release)
        self._motifs = None

    def list(self, **kwargs):
        if self._motifs is None:
            self._motifs = self._jdb_obj.fetch_motifs(**kwargs)
        return self._motifs

    def get_motif(self, matrix_id: str) -> motifs.Motif:
        return self._jdb_obj.fetch_motif_by_id(matrix_id)


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

    def __post_init__(self):
        self.db = JASPAR_DB()

        logger.info(f"Fetching motif {self.matrix_id}...")
        self.motif = self.db.get_motif(self.matrix_id)
        self.alphabet = list(self.motif.alphabet)
        self.length = self.motif.length
        self.n_sequences = sum(self.motif.counts[letter][0] for letter in self.alphabet)

        self._opt_modify_motif()
        self._opt_reverse_complement()
        self.motif.pseudocounts = self.pseudocounts
        self.motif.background = self.background
        self.pwm = self.motif.pwm
        self.pssm = self.motif.pssm
        self.pssm_backward = self.motif.reverse_complement().pssm

        self._init_vectors()
        self.mean = self.pssm.mean(self.motif.background)
        self.std = self.pssm.std(self.motif.background)

    def _opt_modify_motif(self):
        if self.consensus_override is not None:
            logger.info("Using consensus to override...")
            self.matrix_id += "_consensus"
            for pos, consensus_letter in self.consensus_override.items():
                logger.info(
                    f"\n\tmotif.counts[{consensus_letter}][{pos}] = {self.motif.counts[consensus_letter][pos]} -> {self.n_sequences}"
                )
                self.motif.counts[consensus_letter][pos] = self.n_sequences
                self.matrix_id += f"__{pos}{consensus_letter}"
                for letter in set(self.alphabet).difference({consensus_letter}):
                    logger.info(
                        f"\n\tmotif.counts[{letter}][{pos}] = {self.motif.counts[letter][pos]} -> 0"
                    )
                    self.motif.counts[letter][pos] = 0

        elif self.counts_override is not None:
            logger.info("Using counts to override...")
            for (letter, pos), new_count in self.counts_override.items():
                logger.info(
                    f"\n\tmotif.counts[{letter}][{pos}] = {self.motif.counts[letter][pos]} -> {new_count}"
                )
                self.motif.counts[letter][pos] = new_count
                self.matrix_id += f"__{letter}{pos}_{new_count}"

    def _opt_reverse_complement(self):
        if self.reverse_complement:
            logger.info("Using reverse complement...")
            self.motif = self.motif.reverse_complement()
            self.matrix_id += ".rc"

    def _init_vectors(self):
        self.ic_vector = []
        self.var_vector = []
        for i in range(self.length):
            ic = 0.0
            for letter in self.alphabet:
                logodds = self.pssm[letter, i]
                if math.isnan(logodds) or (math.isinf(logodds) and logodds < 0):
                    continue
                b = self.motif.background[letter]
                p = b * math.pow(2, logodds)
                ic += p * logodds
            scores = np.array([self.pssm[letter, i] for letter in self.alphabet])
            var = np.var(scores)
            self.ic_vector.append(round(ic, 3))
            self.var_vector.append(round(var, 3))

    def get_ic_at(self, i: int) -> float:
        return self.ic_vector[i]

    def forward_score(self, pattern: str) -> float:
        return self.pssm.calculate(pattern)

    def backward_score(self, pattern: str) -> float:
        return self.pssm_backward.calculate(pattern)

    @staticmethod
    def double_stranded_score(forward_score: float, backward_score: float) -> float:
        return max(forward_score, backward_score)

    def score(self, pattern: str, strand_for_score: StrandForBindingScore) -> float:
        if strand_for_score == StrandForBindingScore.Forward:
            return self.forward_score(pattern)
        elif strand_for_score == StrandForBindingScore.Backward:
            return self.backward_score(pattern)
        elif strand_for_score == StrandForBindingScore.Double:
            forward_score = self.forward_score(pattern)
            backward_score = self.backward_score(pattern)
            return self.double_stranded_score(forward_score, backward_score)
        else:
            raise ValueError(
                f"Invalid value for strand for binding score: {strand_for_score}"
            )

    def get_binding_score_map(
        self, strand_for_score: StrandForBindingScore = StrandForBindingScore.Double
    ) -> dict[str, float]:
        logger.info(
            f"Using {strand_for_score.get_description()} to calculate binding scores"
        )
        pattern_score: dict[str, float] = {}
        for kmer in product(self.alphabet, repeat=self.length):
            pattern = "".join(kmer)
            score = self.score(pattern, strand_for_score)
            pattern_score[pattern] = score
        return pattern_score

    def get_binding_score_maps(self) -> dict[StrandForBindingScore, dict[str, float]]:
        forward_pattern_score: dict[str, float] = {}
        backward_pattern_score: dict[str, float] = {}
        double_stranded_pattern_score: dict[str, float] = {}
        for kmer in product(self.alphabet, repeat=self.length):
            pattern = "".join(kmer)
            forward_score = self.forward_score(pattern)
            backward_score = self.backward_score(pattern)
            forward_pattern_score[pattern] = forward_score
            backward_pattern_score[pattern] = backward_score
            double_stranded_pattern_score[pattern] = self.double_stranded_score(
                forward_score, backward_score
            )

        return {
            StrandForBindingScore.Forward: forward_pattern_score,
            StrandForBindingScore.Backward: backward_pattern_score,
            StrandForBindingScore.Double: double_stranded_pattern_score,
        }

    def find_significant_patterns(self, pvalue: float) -> list[str]:
        distribution = self.pssm.distribution()
        threshold = distribution.threshold_fpr(pvalue)

        significant_patterns: list[tuple[str, float]] = []

        max_suffix_scores = [0.0] * (self.length + 1)
        for i in range(self.length - 1, -1, -1):
            max_at_i = max(self.pssm[base][i] for base in self.alphabet)
            max_suffix_scores[i] = max_suffix_scores[i + 1] + max_at_i

        stack = [("", 0.0, 0)]
        while stack:
            current_kmer, current_score, pos = stack.pop()

            if pos == self.length:
                if current_score >= threshold:
                    significant_patterns.append((current_kmer, current_score))
                continue

            if current_score + max_suffix_scores[pos] < threshold:
                continue

            for base in self.alphabet:
                stack.append(
                    (current_kmer + base, current_score + self.pssm[base][pos], pos + 1)
                )

        significant_patterns = sorted(
            significant_patterns, key=itemgetter(1), reverse=True
        )
        logger.info(
            f"Found {len(significant_patterns)} unwanted patterns (p-value={pvalue})"
            f"\n\tHighest score:  {significant_patterns[0]}"
            f"\n\tLowest score:   {significant_patterns[-1]}"
        )
        return [p[0] for p in significant_patterns]

    def write(self, fmt: str, f) -> None:
        fmt = fmt.lower()
        if fmt == "meme":
            alphabet = self.motif.alphabet
            f.write("MEME version 5.0.0\n\n")
            f.write(f"ALPHABET= {alphabet}\n\n")
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

    @property
    def total_ic(self) -> float:
        return self.mean

    @property
    def avg_ic(self) -> float:
        return self.mean / self.length

    def __str__(self) -> str:
        return f"{self.matrix_id} (m={self.length})"

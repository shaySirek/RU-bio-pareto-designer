from dataclasses import dataclass
from itertools import product
from operator import itemgetter
import math

from loguru import logger
from pyjaspar import jaspardb
from Bio import motifs
import numpy as np

from pareto_designer.models.motif import (
    BindingMotif as MotifBase,
    StrandForBindingScore,
)

PSSM_DISTRIBUTION_PRECISION = 3_000


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


@dataclass
class BindingMotif(MotifBase):
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
        self._score_distribution = None

    def _get_score_distribution(self):
        if self._score_distribution is None:
            self._score_distribution = self.pssm.distribution(
                precision=PSSM_DISTRIBUTION_PRECISION,
                background=dict(self.motif.background),
            )
        return self._score_distribution

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

    def score_fpr(self, score: float) -> float:
        dist = self._get_score_distribution()
        idx = dist._index_diff(score, dist.min_score)
        idx = max(0, min(dist.n_points - 1, idx))
        return float(sum(dist.bg_density[idx:]))

    def is_significant_window(self, pattern: str, pvalue: float) -> bool:
        return (
            self.score_fpr(self.forward_score(pattern)) <= pvalue
            or self.score_fpr(self.backward_score(pattern)) <= pvalue
        )

    def hit_score_threshold(self, pvalue: float) -> float:
        """FIMO-aligned score cutoff: single-strand p-value threshold.

        Uses the same background as the MEME export and FIMO (--bgfile --motif--).
        A window is a hit when max(forward, backward) >= this value.
        """
        return float(self._get_score_distribution().threshold_fpr(pvalue))

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
        threshold = self.hit_score_threshold(pvalue)

        significant_patterns: list[tuple[str, float]] = []
        for kmer in product(self.alphabet, repeat=self.length):
            pattern = "".join(kmer)
            if not self.is_significant_window(pattern, pvalue):
                continue
            score = self.score(pattern, StrandForBindingScore.Double)
            significant_patterns.append((pattern, score))

        significant_patterns.sort(key=itemgetter(1), reverse=True)
        if significant_patterns:
            logger.info(
                f"Found {len(significant_patterns)} unwanted patterns (p-value={pvalue})"
                f"\n\tHighest score:  {significant_patterns[0]}"
                f"\n\tLowest score:   {significant_patterns[-1]}"
            )
        else:
            logger.info(
                f"Found 0 unwanted patterns (p-value={pvalue}, threshold={threshold:.3f})"
            )
        return [p[0] for p in significant_patterns]

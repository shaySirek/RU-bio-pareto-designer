import numpy as np
from Bio.Data import CodonTable
from loguru import logger
from pathlib import Path

from pareto_designer.shared.parsing import read_sequence


class CostUtils:
    _TRANSITIONS: set[tuple[str, str]] = {
        ("A", "G"),
        ("G", "A"),
        ("C", "T"),
        ("T", "C"),
    }

    STOP_CODON_MARKER: str = "*"
    START_CODON_MARKER: str = "*"
    ORF_START_POS: int = -3

    def __init__(self, table_id: int = 1):
        self.table_id = table_id
        biopython_table = CodonTable.ambiguous_dna_by_id[self.table_id]
        self._genetic_code = biopython_table.forward_table
        self._start_codons = set(biopython_table.start_codons)
        self._stop_codons = set(biopython_table.stop_codons)

    def encodes_same_amino_acid(self, proposed_codon: str, current_codon: str) -> bool:
        return self._genetic_code.get(proposed_codon) == self._genetic_code.get(
            current_codon
        )

    def either_is_stop_codon(self, current_codon: str, proposed_codon: str) -> bool:
        current_aa = self._genetic_code.get(current_codon, self.STOP_CODON_MARKER)
        proposed_aa = self._genetic_code.get(proposed_codon, self.STOP_CODON_MARKER)
        return (
            current_aa == self.STOP_CODON_MARKER
            or proposed_aa == self.STOP_CODON_MARKER
        )

    @staticmethod
    def is_orf_start(codon_pos: int) -> bool:
        return codon_pos == CostUtils.ORF_START_POS

    @staticmethod
    def is_transition(nucleotide1: str, nucleotide2: str) -> bool:
        return (nucleotide1, nucleotide2) in CostUtils._TRANSITIONS

    @staticmethod
    def hamming_dist(target_codon: str, proposed_codon: str) -> int:
        return sum(x != y for x, y in zip(target_codon, proposed_codon))

    def calculate_codon_costs(self, codon_usage: dict[str, float]) -> dict[str, float]:
        if not codon_usage:
            return {}

        aa_to_max_f: dict[str, float] = {}
        for codon, freq in codon_usage.items():
            aa = self._genetic_code.get(codon, self.STOP_CODON_MARKER)
            if aa:
                aa_to_max_f[aa] = max(aa_to_max_f.get(aa, 0.0), freq)

        codons = list(codon_usage.keys())
        f_vals = np.fromiter(map(codon_usage.get, codons), dtype=float)
        m_vals = np.fromiter(
            map(
                lambda c: aa_to_max_f.get(
                    self._genetic_code.get(c, self.STOP_CODON_MARKER)
                ),
                codons,
            ),
            dtype=float,
        )

        w = f_vals / m_vals
        costs = -np.log(w)
        costs[np.isclose(costs, 0.0, atol=1e-9)] = 0.0
        costs = np.round(costs, 6).tolist()

        codon_costs = dict(zip(codons, costs))

        return codon_costs

    def get_coding_positions(self, sequence: str) -> tuple[str, list[int]]:
        start_codon_index = sequence.find(self.START_CODON_MARKER)
        if start_codon_index == -1:
            return sequence, [0] * len(sequence)

        sequence = sequence[:start_codon_index] + sequence[start_codon_index + 1 :]
        if self.START_CODON_MARKER in sequence:
            raise ValueError("Only a single coding region is supported")

        detected_codon = sequence[start_codon_index : start_codon_index + 3]
        if detected_codon not in self._start_codons:
            raise ValueError(
                "Invalid marker for start codon:"
                f" Expected one of {sorted(self._start_codons)} after {self.START_CODON_MARKER},"
                f" got {detected_codon}."
            )

        n = len(sequence)
        logger.info(f"Processing sequence of length {n}")

        codon_positions: list[int] = [0] * n
        codon_positions[start_codon_index : start_codon_index + 3] = [
            1,
            2,
            CostUtils.ORF_START_POS,
        ]

        for i in range(start_codon_index + 3, n - 2, 3):
            codon_positions[i : i + 3] = [1, 2, 3]
            codon = sequence[i : i + 3]
            if codon in self._stop_codons:
                logger.info(
                    f"Found ORF in positions {start_codon_index + 1}-{i + 3} (stop codon: {codon})"
                )
                return sequence, codon_positions

        raise ValueError(
            "Invalid coding region:"
            f" no stop codon found for start codon at index {start_codon_index}."
        )


def fasta_to_txt_with_marked_cds(
    fasta_file: Path, bases_before_cds: int, bases_after_cds: int
):
    sequence = read_sequence(fasta_file)
    sequence = (
        sequence[:bases_before_cds]
        + CostUtils.START_CODON_MARKER
        + sequence[bases_after_cds:]
    )
    with fasta_file.with_suffix(".txt").open("wt") as f:
        f.write(sequence)

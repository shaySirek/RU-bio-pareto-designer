from pathlib import Path
from loguru import logger

from pareto_designer.shared.consts import START_CODON_MARKER, START_CODON, STOP_CODONS
from pareto_designer.shared.parsing import read_sequence


def fasta_to_txt_with_marked_cds(fasta_file: Path, bases_before_cds: int):
    sequence = read_sequence(fasta_file)
    sequence = (
        sequence[:bases_before_cds] + START_CODON_MARKER + sequence[bases_before_cds:]
    )
    with fasta_file.with_suffix(".txt").open("wt") as f:
        f.write(sequence)


def get_coding_positions(sequence: str) -> tuple[str, list[int]]:
    start_codon_index = sequence.find(START_CODON_MARKER)
    if start_codon_index == -1:
        return sequence, [0] * len(sequence)  # no coding region

    sequence = sequence[:start_codon_index] + sequence[start_codon_index + 1 :]
    if START_CODON_MARKER in sequence:
        raise ValueError("Only a single coding region is supported")

    if sequence[start_codon_index : start_codon_index + 3] != START_CODON:
        raise ValueError(
            "Invalid marker for start codon:"
            f" {START_CODON} is excepted after {START_CODON_MARKER},"
            f" got {sequence[start_codon_index: start_codon_index + 3]}."
        )

    n = len(sequence)
    codon_positions: list[int] = [0] * n
    codon_positions[start_codon_index : start_codon_index + 3] = [1, 2, -3]

    for i in range(start_codon_index + 3, n - 2, 3):
        codon_positions[i : i + 3] = [1, 2, 3]
        if sequence[i : i + 3] in STOP_CODONS:
            logger.info(f"ORF found in positions {start_codon_index + 1}-{i + 3}")
            return sequence, codon_positions

    raise ValueError(
        "Invalid coding region:"
        f" no stop codon found for start codon at index {start_codon_index}."
    )

from pathlib import Path


def read_sequence(fasta_file: Path) -> str:
    sequence = []
    with fasta_file.open("r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        sequence.append(line)

    return "".join(sequence)


def write_sequence(
    fasta_file: Path,
    sequence: str,
    header: str = "",
    width: int = 60,
):
    with fasta_file.open("w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), width):
            f.write(sequence[i : i + width] + "\n")


def read_codon_usage(codon_usage_file: Path, convert_to_dna=True) -> dict[str, float]:
    codon_usage: dict[str, float] = {}

    with codon_usage_file.open("r") as f:
        raw_lines = f.readlines()

    for line_num, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid format in codon usage file at line {line_num}: '{line}'. "
                "Expected format: CODON FREQUENCY (e.g., ACG 0.02)"
            )

        codon = parts[0].upper()
        if convert_to_dna:
            codon = codon.replace("U", "T")

        if len(codon) != 3 or any(base not in "ATCGU" for base in codon):
            raise ValueError(
                f"Invalid codon '{codon}' at line {line_num}. Must be 3 letters A/T/C/G (or U before conversion)."
            )

        freq = float(parts[1])
        codon_usage[codon] = freq

    return codon_usage

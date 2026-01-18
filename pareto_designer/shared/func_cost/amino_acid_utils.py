STOP_CODON = "*"
codon_to_amino_acid: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",  # Phenylalanine (F)
    "TTA": "L",
    "TTG": "L",  # Leucine (L)
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",  # Leucine (L)
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",  # Isoleucine (I)
    "ATG": "M",  # Methionine (M) (Start codon)
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",  # Valine (V)
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",  # Serine (S)
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",  # Proline (P)
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",  # Threonine (T)
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",  # Alanine (A)
    "TAT": "Y",
    "TAC": "Y",  # Tyrosine (Y)
    "CAT": "H",
    "CAC": "H",  # Histidine (H)
    "CAA": "Q",
    "CAG": "Q",  # Glutamine (Q)
    "AAT": "N",
    "AAC": "N",  # Asparagine (N)
    "AAA": "K",
    "AAG": "K",  # Lysine (K)
    "GAT": "D",
    "GAC": "D",  # Aspartic acid (D)
    "GAA": "E",
    "GAG": "E",  # Glutamic acid (E)
    "TGT": "C",
    "TGC": "C",  # Cysteine (C)
    "TGG": "W",  # Tryptophan (W)
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",  # Arginine (R)
    "AGT": "S",
    "AGC": "S",  # Serine (S)
    "AGA": "R",
    "AGG": "R",  # Arginine (R)
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",  # Glycine (G)
    "TGA": STOP_CODON,
    "TAA": STOP_CODON,
    "TAG": STOP_CODON,
}


class AminoAcidConfig:

    @staticmethod
    def encodes_same_amino_acid(proposed_codon: str, current_codon: str) -> bool:
        """
        Checks if two codons encode the same amino acid.

        Args:
            proposed_codon (str): The codon to be tested.
            current_codon (str): The current codon.

        Returns:
            bool: True if both codons encode the same amino acid, False otherwise.
        """
        return codon_to_amino_acid.get(proposed_codon) == codon_to_amino_acid.get(
            current_codon
        )

    @staticmethod
    def either_is_stop_codon(current_codon: str, proposed_codon: str) -> bool:
        """
        Checks if a given codon is a stop codon.

        Args:
            current_codon (str): The original codon.
            proposed_codon (str): The codon to be tested.

        Returns:
            bool: True if the codon is a stop codon, False otherwise.
        """
        return (
            codon_to_amino_acid.get(current_codon) == STOP_CODON
            or codon_to_amino_acid.get(proposed_codon) == STOP_CODON
        )

    @staticmethod
    def is_start_codon(codon_position: int) -> bool:
        """
        Checks if a given codon is a stop codon.

        Args:
            codon_position (number): The codon to be tested, should be always -3

        Returns:
            bool: True if the codon is a start codon, False otherwise.
        """
        return codon_position == -3

    @staticmethod
    def is_transition(nucleotide1: str, nucleotide2: str) -> bool:
        """
        Checks if the substitution between two nucleotides is a transition mutation.

        Args:
            nucleotide1 (str): The original nucleotide (A, C, G, or T).
            nucleotide2 (str): The proposed nucleotide (A, C, G, or T).

        Returns:
            bool: True if the substitution is a transition mutation, False otherwise.
        """

        return (nucleotide1, nucleotide2) in {
            ("A", "G"),
            ("G", "A"),
            ("C", "T"),
            ("T", "C"),
        }

    @staticmethod
    def edit_dist(target_codon: str, proposed_codon: str) -> int:
        """
        Calculate the number of nucleotide differences between two codons.

        Args:
            target_codon (str): The original codon (3 nucleotides).
            proposed_codon (str): The new codon to compare with.

        Returns:
            int: The number of positions where the nucleotides differ.
        """
        # Use zip to pair each nucleotide from target and proposed codon
        # Compare each pair (a, b) and count +1 if they are different
        # The sum(...) collects the total count of mismatches
        return sum(1 for a, b in zip(target_codon, proposed_codon) if a != b)

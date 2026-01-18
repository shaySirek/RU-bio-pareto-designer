from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Region:
    """https://rest.ensembl.org/documentation/info/overlap_region"""

    species: str = ""
    feature_type: str = ""
    biotype: str = ""
    seq_region_name: str = ""
    id: str = ""
    source: str = ""
    logic_name: str = ""
    assembly_name: str = ""
    external_name: str = ""
    canonical_transcript: str = ""
    description: str = ""
    version: int = 0
    gene_id: str = ""
    phase: str = ""
    Parent: str = ""
    protein_id: str = ""
    start: int = 0
    end: int = 0
    strand: int = 0

    @property
    def region_desc(self) -> str:
        return f"{self.seq_region_name}:{self.start}..{self.end}:{self.strand}"

    @property
    def _id(self):
        return self.gene_id or self.protein_id.split("_")[0]

    def get_fasta_path(self, base: Path) -> Path:
        fasta_path = (
            base
            / self.species
            / self.seq_region_name
            / f"{self._id}_{self.strand}_{self.start}_{self.end}.fa"
        )
        return fasta_path

    def get_sub_region(self, start: int, end: int) -> "Region":
        sub_region_kwargs = asdict(self)
        sub_region_kwargs.update(start=start, end=end)
        return Region(**sub_region_kwargs)

    def __str__(self) -> str:
        return f"{self.region_desc} [gene={self._id},species={self.species}]"

    def to_dict(self) -> dict:
        return asdict(self)

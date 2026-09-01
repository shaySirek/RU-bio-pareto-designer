import subprocess
import shutil

from pathlib import Path
from loguru import logger
import pandas as pd

from pareto_designer.models.motif import BindingMotif
from pareto_designer.models.region import Region
from pareto_designer.bio_fetcher.paths import FIMO_DIR


def find_hits_in_region(
    region: Region,
    seq_fasta_file: Path,
    motif: BindingMotif,
    motif_file: Path,
    pval: float,
) -> tuple[Path, pd.DataFrame]:
    outdir = FIMO_DIR / motif.matrix_id / region.species / region._id
    hits = run_fimo(
        motif_file,
        seq_fasta_file,
        pval,
        outdir,
    )
    if hits.empty:
        logger.info(
            f"region {region} has no hits with motif {motif.matrix_id} (p-value={pval})"
        )
    else:
        logger.info(
            f"region {region} has {len(hits)} hits with motif {motif.matrix_id} (p-value={pval})"
        )

    return outdir, hits


def get_motif_hits(
    seq_id: str,
    seq_fasta_file: Path,
    motif: BindingMotif,
    motif_file: Path,
    pval: float = 2e-3,
) -> list[tuple[int, int]]:
    outdir = FIMO_DIR / motif.matrix_id / seq_id
    hits = run_fimo(
        motif_file,
        seq_fasta_file,
        pval,
        outdir,
    )
    motif_hits = [(int(hit["start"]), int(hit["stop"])) for _, hit in hits.iterrows()]
    shutil.rmtree(outdir)

    return motif_hits


def fimo_window_starts(hits: pd.DataFrame) -> set[int]:
    if hits.empty:
        return set()
    return {int(row["start"]) - 1 for _, row in hits.iterrows()}


def run_fimo(
    motif_file: Path,
    fasta_file: Path,
    pval: float,
    outdir: Path,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(outdir)

    cmd = [
        "fimo",
        "--thresh",
        str(pval),
        "--bgfile",
        "--motif--",
        "--o",
        str(outdir),
        str(motif_file),
        str(fasta_file),
    ]
    subprocess.run(cmd, check=True)

    df = pd.DataFrame()
    tsv = outdir / "fimo.tsv"
    if tsv.exists():
        try:
            df = pd.read_csv(tsv, sep="\t", comment="#")
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

    return df

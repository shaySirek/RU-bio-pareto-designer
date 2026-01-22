from typing import Optional
import subprocess
import shutil

from pathlib import Path
from loguru import logger
import pandas as pd

from pareto_designer.models.motif import BindingMotif
from pareto_designer.models.region import Region
from pareto_designer.bio_fetcher.paths import FIMO_DIR


def find_hits(
    region: Region,
    seq_fasta_file: Path,
    motif: BindingMotif,
    motif_file: Path,
    fimo_pval: float,
) -> tuple[Path, Optional[pd.DataFrame]]:
    outdir = FIMO_DIR / motif.matrix_id / region.species / region._id
    hits = _run_fimo(
        motif_file,
        seq_fasta_file,
        fimo_pval,
        outdir,
    )
    if hits is None:
        logger.info(f"region {region} has no hits with motif {motif.matrix_id}")
    else:
        logger.info(
            f"region {region} has {len(hits)} hits with motif {motif.matrix_id}"
        )

    return outdir, hits


def _run_fimo(
    motif_file: Path,
    fasta_file: Path,
    pval: float,
    outdir: Path,
) -> Optional[pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(outdir)

    cmd = [
        "fimo",
        "--thresh",
        str(pval),
        "--o",
        str(outdir),
        str(motif_file),
        str(fasta_file),
    ]
    subprocess.run(cmd, check=True)

    tsv = outdir / "fimo.tsv"
    if not tsv.exists():
        return None

    df = pd.read_csv(tsv, sep="\t", comment="#")
    if "start" not in df.columns:
        return None
    df = df.dropna(subset=["start"])
    return df

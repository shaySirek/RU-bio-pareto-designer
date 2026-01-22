from typing import Generator

from pathlib import Path
from loguru import logger
import pandas as pd

from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.bio_fetcher.ensembl import fetch_species_genes, fetch_cdss_by_id
from pareto_designer.bio_fetcher.fimo import find_hits
from pareto_designer.models.region import Region
from pareto_designer.bio_fetcher.paths import (
    MOTIF_DIR,
    STATS_DIR,
    STATS_FILENAME,
    MOTIF_PATTERNS_FILENAME,
    MOTIF_HITS_FILENAME,
)
from pareto_designer.shared.csv_writer import write_results_stream
from pareto_designer.shared.cds_util import fasta_to_txt_with_marked_cds


def _get_genes_motif_hits_stats(
    species: str,
    motif: BindingMotif,
    fimo_pval: float,
) -> Generator[dict, None, None]:
    motif_file = motif.dump("meme", MOTIF_DIR)
    motif_related_obj = {
        "motif_id": motif.matrix_id,
        "motif_hit_pvalue": fimo_pval,
    }
    for region, fasta_file in fetch_species_genes(species):
        hits = find_hits(region, fasta_file, motif, motif_file, fimo_pval)
        if hits:
            yield {
                **region.to_dict(),
                **motif_related_obj,
                "number_of_motif_hits": len(hits),
            }


def get_genes_motif_hits_stats(species: str, motif: BindingMotif, fimo_pval: float):
    write_results_stream(
        _get_genes_motif_hits_stats(species, motif, fimo_pval),
        STATS_DIR / motif.matrix_id / species / STATS_FILENAME,
    )


def find_hits_in_genes(
    species: str,
    gene_id: str,
    bases_before_cds: int,
    bases_after_cds: int,
    motif: BindingMotif,
    fimo_pval: float,
):
    motif_file = motif.dump("meme", MOTIF_DIR)
    for region, fasta_file in fetch_cdss_by_id(
        species,
        gene_id,
        bases_before=bases_before_cds,
        bases_after=bases_after_cds,
    ):
        fasta_to_txt_with_marked_cds(fasta_file, bases_before_cds)
        outdir, hits = find_hits(region, fasta_file, motif, motif_file, fimo_pval)
        if hits is not None:
            region_hits_to_bed_file(region, motif, hits, outdir / MOTIF_HITS_FILENAME)


def region_hits_to_bed_file(
    region: Region,
    motif: BindingMotif,
    hits: pd.DataFrame,
    outfile: Path,
):
    bed_data = []
    for _, hit in hits.iterrows():
        fs, fe = int(hit["start"]), int(hit["stop"])
        if region.strand == 1:
            gs = region.start + fs - 1
            ge = region.start + fe - 1
        else:
            gs = region.end - fe + 1
            ge = region.end - fs + 1
        bed_data.append(
            {
                "chrom": region.seq_region_name,
                "start": gs - 1,
                "end": ge,
                "name": hit["matched_sequence"],
                "score": hit["score"],
                "strand": hit["strand"],
            }
        )

    bed_df = pd.DataFrame(bed_data)
    if not bed_df.empty:
        s_min, s_max = bed_df["score"].min(), bed_df["score"].max()
        if s_max != s_min:
            bed_df["score"] = (
                (bed_df["score"] - s_min) / (s_max - s_min) * 1000
            ).astype(int)
        else:
            bed_df["score"] = 1000

    track_line = f'track name="{motif.matrix_id} hits" itemRgb="On" color=255,0,0\n'
    with outfile.open("w") as f:
        f.write(track_line)
        bed_df.to_csv(f, sep="\t", index=False, header=False)

    logger.info(f"Generated BED file: {str(outfile)} ({len(bed_df)} hits)")


def find_significant_patterns(
    motif: BindingMotif,
    hit_pvalue: float,
):
    unwanted_patterns = motif.find_significant_patterns(hit_pvalue)
    unwanted_patterns_file = MOTIF_DIR / motif.matrix_id / MOTIF_PATTERNS_FILENAME
    unwanted_patterns_file.parent.mkdir(parents=True, exist_ok=True)
    with unwanted_patterns_file.open("wt") as f:
        f.write("\n".join(unwanted_patterns) + "\n")
    logger.info(
        f"{len(unwanted_patterns)} patterns were written into {str(unwanted_patterns_file)}"
    )

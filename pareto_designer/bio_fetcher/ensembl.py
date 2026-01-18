from typing import Generator
import time
from pathlib import Path
from http import HTTPStatus

import requests
from loguru import logger

from pareto_designer.bio_fetcher.structs import Region
from pareto_designer.bio_fetcher.paths import SEQ_DIR

ENSEMBL_SERVER = "https://rest.ensembl.org"
JSON_HEADERS = {"Content-Type": "application/json"}
FASTA_HEADERS = {"Content-Type": "text/x-fasta"}
API_MAX_RETRIES = 5
CHUNK_SIZE = 1_000_000


def _ensembl_get_json(url: str) -> dict:
    for _ in range(API_MAX_RETRIES):
        r = requests.get(url, headers=JSON_HEADERS)
        if r.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            time.sleep(1)
            continue
        if r.status_code == HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f"Bad request: {url}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed Ensembl request: {url}")


def _ensembl_fetch_fasta_to_file(url: str, out_fasta: Path) -> bool:
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    if out_fasta.exists():
        return False
    for _ in range(API_MAX_RETRIES):
        with requests.get(url, headers=FASTA_HEADERS, stream=True) as r:
            if r.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                time.sleep(1)
                continue
            if r.status_code == HTTPStatus.BAD_REQUEST:
                raise RuntimeError(f"Bad request: {url}")
            r.raise_for_status()
            with open(out_fasta, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            return True
    raise RuntimeError(f"Failed FASTA fetch: {url}")


def get_species_chromosomes(species: str) -> list[tuple[str, int]]:
    url = f"{ENSEMBL_SERVER}/info/assembly/{species}"
    data = _ensembl_get_json(url)
    return [
        (r["name"], r["length"])
        for r in data["top_level_region"]
        if r["coord_system"] in {"chromosome", "scaffold"}
    ]


def fetch_species_genes(
    species: str,
    outdir: Path = SEQ_DIR,
) -> Generator[tuple[Region, Path], None, None]:
    regions = get_species_chromosomes(species)
    for chrom, chrom_len in regions:
        logger.info(f"Got chromosome {chrom} of length {chrom_len}")
        for region_start in range(1, chrom_len + 1, CHUNK_SIZE):
            region_end = min(region_start + CHUNK_SIZE - 1, chrom_len)
            yield from fetch_genes_within_region(
                species, chrom, region_start, region_end, outdir
            )


def fetch_genes_within_region(
    species: str,
    chrom: str,
    region_start: int,
    region_end: int,
    outdir: Path = SEQ_DIR,
    bases_before: int = 0,
    bases_after: int = 0,
) -> Generator[tuple[Region, Path], None, None]:
    location = f"{chrom}:{region_start}-{region_end}"
    url = f"{ENSEMBL_SERVER}/overlap/region/{species}/{location}?feature=gene;biotype=protein_coding"
    genes = _ensembl_get_json(url)
    logger.info(f"Got {len(genes)} genes in {location} of {species}")
    yield from _download_regions(species, genes, outdir, bases_before, bases_after)


def fetch_cdss_by_id(
    species: str,
    gene_id: str,
    outdir: Path = SEQ_DIR,
    bases_before: int = 0,
    bases_after: int = 0,
) -> Generator[tuple[Region, Path], None, None]:
    url = f"{ENSEMBL_SERVER}/overlap/id/{gene_id}?feature=cds;biotype=protein_coding"
    cdss = _ensembl_get_json(url)
    logger.info(f"Got {len(cdss)} CDS(s) by id {gene_id} [species={species}]")
    yield from _download_regions(species, cdss, outdir, bases_before, bases_after)


def _download_regions(
    species: str,
    regions: list[dict],
    outdir: Path,
    bases_before: int,
    bases_after: int,
) -> Generator[tuple[Region, Path], None, None]:
    for r in regions:
        region = Region(species=species, **r)
        if bases_before:
            logger.info(
                f"Adding {bases_before} bases before start of region {region.start}"
            )
            region.start -= bases_before
        if bases_after:
            logger.info(f"Adding {bases_after} bases after end of region {region.end}")
            region.end += bases_after
        logger.info(f"Downloading region {region}")
        fasta_file = region.get_fasta_path(outdir)
        _download_region(region, fasta_file)
        yield region, fasta_file


def _download_region(region: Region, fasta_file: Path):
    seq_url = f"{ENSEMBL_SERVER}/sequence/region/{region.species}/{region.region_desc}"
    written = _ensembl_fetch_fasta_to_file(seq_url, fasta_file)
    if written:
        logger.info(f"Wrote region {region} into {fasta_file}")

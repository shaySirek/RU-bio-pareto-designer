from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from pathlib import Path

import pandas as pd
from loguru import logger

from pareto_designer.bio_fetcher.structs import Region
from pareto_designer.bio_fetcher.ensembl import _download_region


def combine_fimo_hits(
    main_region: Region,
    hits: pd.DataFrame,
    max_region_length: int,
    outdir: Path,
) -> Iterator[tuple[Region, Path]]:
    outdir.mkdir(parents=True, exist_ok=True)
    genomic_hits = _convert_strand_hits_to_genomic(main_region, hits)
    combined_hits = _combine_hits_to_regions(genomic_hits, max_region_length)

    def _download(gene_hit_idxs: tuple[int, int]) -> tuple[Region, Path]:
        g_start, g_end = gene_hit_idxs
        region = main_region.get_sub_region(g_start, g_end)
        logger.info(f"Downloading region {region}")
        fasta_file = region.get_fasta_path(outdir)
        _download_region(region, fasta_file)
        return region, fasta_file

    with ThreadPoolExecutor(max_workers=4) as executor:
        return executor.map(_download, combined_hits)


def _convert_strand_hits_to_genomic(
    region: Region, hits: pd.DataFrame
) -> list[tuple[int, int]]:
    genomic_hits = []

    strand = {1: "+", -1: "-"}.get(region.strand)
    strand_hits = hits[hits["strand"] == strand]
    logger.info(f"Using {len(strand_hits)} hits of strand {strand}")

    for _, hit in strand_hits.iterrows():
        fs, fe = int(hit["start"]), int(hit["stop"])
        if region.strand == 1:
            gs = region.start + fs - 1
            ge = region.start + fe - 1
        else:
            gs = region.end - fe + 1
            ge = region.end - fs + 1
        genomic_hits.append((min(gs, ge), max(gs, ge)))

    genomic_hits.sort()
    return genomic_hits


def _combine_hits_to_regions(
    genomic_hits: list[tuple[int, int]], max_region_length: int
) -> list[tuple[int, int]]:
    reg_start, reg_end = None, None
    regions = []
    for hs, he in genomic_hits:
        if reg_start is None:
            reg_start, reg_end = hs, he
            continue
        if he - reg_start + 1 > max_region_length:
            regions.append((reg_start, reg_end))
            reg_start, reg_end = hs, he
        else:
            reg_end = max(reg_end, he)
    if reg_start is not None:
        regions.append((reg_start, reg_end))

    return regions

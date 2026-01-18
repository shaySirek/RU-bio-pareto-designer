#!/usr/bin/env python3

import argparse

from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.bio_fetcher.pipeline import (
    find_hits_in_genes,
    find_significant_patterns,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species", type=str, required=True, help="Species name (Ensembl)"
    )
    parser.add_argument("--gene-id", type=str, required=True, help="Gene id (Ensembl)")
    parser.add_argument(
        "--bases-before",
        "-B",
        type=int,
        default=500,
        help="Number of bases before cds",
    )
    parser.add_argument(
        "--bases-after",
        "-A",
        type=int,
        default=500,
        help="Number of bases after cds",
    )
    parser.add_argument("--motif-id", type=str, required=True, help="Motif ID (JASPAR)")
    parser.add_argument(
        "--hit-pval",
        type=float,
        default=2e-3,
        help="Motif hit p-value threshold (FIMO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    species: str = args.species
    gene_id: str = args.gene_id
    bases_before: int = args.bases_before
    bases_after: int = args.bases_after
    motif = BindingMotif(args.motif_id)
    hit_pvalue: float = args.hit_pval

    find_hits_in_genes(species, gene_id, bases_before, bases_after, motif, hit_pvalue)
    find_significant_patterns(motif, hit_pvalue)

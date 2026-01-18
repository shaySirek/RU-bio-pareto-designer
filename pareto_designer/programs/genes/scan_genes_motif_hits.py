#!/usr/bin/env python3

import argparse

from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.bio_fetcher.pipeline import get_genes_motif_hits_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species", type=str, required=True, help="Species name (Ensembl)"
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
    motif = BindingMotif(args.motif_id)
    fimo_pval: float = args.hit_pval

    get_genes_motif_hits_stats(species, motif, fimo_pval)

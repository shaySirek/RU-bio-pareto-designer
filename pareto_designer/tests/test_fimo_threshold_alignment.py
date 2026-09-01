from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import pytest

from pareto_designer.bio_fetcher.fimo import fimo_window_starts, run_fimo
from pareto_designer.bio_fetcher.motif import BindingMotif
from pareto_designer.bio_fetcher.paths import MOTIF_DIR
from pareto_designer.shared.binding_utils import motif_hit_window_starts

MATRIX_ID = "MA0267.1"
HIT_PVAL = 0.002
RANDOM_SEEDS = (0, 1, 2)
RANDOM_SEQ_LEN = 5000


pytestmark = pytest.mark.skipif(
    shutil.which("fimo") is None, reason="fimo not installed"
)


def _compare_on_sequence(
    seq_id: str,
    seq: str,
    motif: BindingMotif,
    motif_file: Path,
    pval: float,
) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / f"{seq_id}.fa"
        fasta.write_text(f">{seq_id}\n{seq}\n")
        fimo_hits = run_fimo(motif_file, fasta, pval, tmp_path / "fimo_out")

    fimo_starts = fimo_window_starts(fimo_hits)
    our_starts = motif_hit_window_starts(seq, motif, pval)

    return {
        "fimo_hits": len(fimo_starts),
        "our_hits": len(our_starts),
        "agreement": len(fimo_starts & our_starts),
        "fimo_not_ours": len(fimo_starts - our_starts),
        "ours_not_fimo": len(our_starts - fimo_starts),
    }


def test_motif_hit_threshold_matches_fimo_on_random_sequences() -> None:
    motif = BindingMotif(MATRIX_ID)
    motif_file = motif.dump("meme", MOTIF_DIR)

    totals = {
        "fimo_hits": 0,
        "our_hits": 0,
        "agreement": 0,
        "fimo_not_ours": 0,
        "ours_not_fimo": 0,
    }
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        seq = "".join(rng.choice("ACGT") for _ in range(RANDOM_SEQ_LEN))
        stats = _compare_on_sequence(f"random_{seed}", seq, motif, motif_file, HIT_PVAL)
        for key in totals:
            totals[key] += stats[key]

    assert totals["fimo_not_ours"] == 0, totals
    assert totals["ours_not_fimo"] == 0, totals
    if totals["fimo_hits"]:
        assert totals["agreement"] / totals["fimo_hits"] == 1.0
    if totals["our_hits"]:
        assert totals["agreement"] / totals["our_hits"] == 1.0

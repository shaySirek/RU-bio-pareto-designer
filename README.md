# Multiple Objective DNA Sequence Design

DNA sequence design using multiple objective optimization to jointly minimize functional interference and unintended binding.

## Install

```bash
poetry install
```

FIMO is required to annotate motif hits on designed sequences.

## Data

Place local inputs under `bio_data/`:

- `bio_data/zea_mays_genes/` — target sequences (`*.txt`)
- `bio_data/codon_usage/saccharomyces_cerevisiae.txt` — codon frequencies (`CODON FREQUENCY` per line)

The examples below use maize gene `Zm00001eb052570_-1_265197378_265198704` and JASPAR motif `MA0267.1`. Other genes can go in the same folder; pass a directory of `*.txt` files to `-s` to run a batch.

## Experiments

`--reduce-fsm-by 0` keeps the full DB FSM. `0.75`, `0.875`, and `0.9375` are 4-fold, 8-fold, and 16-fold reduction.

### Alpha sweep (K=100, 8-fold FSM)

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 100 \
  --sampler-alpha 0.0 1.0 1.0_log_pos 2.0_log_pos \
  --reduce-fsm-by 0.875
```

### K sweep (α=1.0 and 1.0_log_pos, 8-fold FSM)

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 50 100 150 \
  --sampler-alpha 1.0 1.0_log_pos \
  --reduce-fsm-by 0.875
```

### FSM reduction sweep

After choosing a sampler from the sweeps above (example: `-k 100 --sampler-alpha 1.0`):

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 100 \
  --sampler-alpha 1.0 \
  --reduce-fsm-by 0 0.75 0.875 0.9375
```

Add `--dry-run` to re-render outputs without running the optimizer.

## Outputs

Each run is written under:

```
designer_results/<gene>/<cost_params>/<motif>/<fsm_id>/PowerLawSUS/<sampler_params>/
```

Comparison files for one CLI invocation are written at the common parent of those run directories:

- `pareto_frontiers.png`, `motif_cost_dists.png`
- `pareto_comparison.json` — pairwise coverage and normalized hypervolume
- `pareto_comparison.csv` — per-run `k`, `alpha`, `log_pos`, `fsm_size`, `reduce_fsm_by`, binding-score SSE, FSM reduction error, and 2-objective hypervolume (cost and binding, both minimized)

All sequences from the same `design-seq` run are also collected in `designer_results/pareto_comparison.csv` (includes a `seq_id` column).

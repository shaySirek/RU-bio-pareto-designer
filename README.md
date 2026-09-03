# Multiple-objective DNA sequence design

This guide documents how to install, run, and reproduce **pareto-designer**, an implementation of a method for synthetic DNA sequence design that reconciles functional requirements with avoidance of unintended protein DNA binding. Instead of treating motif avoidance as a binary constraint, the method uses position-specific scoring matrices (PSSMs) to quantify binding affinity and formalizes sequence redesign as a multi-objective optimization task. A dynamic programming algorithm jointly minimizes unintended binding and functional interference, tracking continuous PSSM scores across overlapping windows with a de Bruijn graph-based finite state machine (FSM). To maintain efficiency, we utilize a state reduction algorithm that merges FSM states with minimal impact on accuracy, and a pruning strategy to manage the growth of the Pareto-optimal set in long sequences.

The reference case study targets expression of maize genes in yeast. Motifs come from [JASPAR](https://jaspar.elixir.no/); target genes from [Ensembl](https://www.ensembl.org/); designed sequences are checked for spurious motif hits with [FIMO](https://meme-suite.org/meme/doc/fimo.html). The bundled experiment ([configs/pareto_experiment_ma0267.yaml](configs/pareto_experiment_ma0267.yaml)) sweeps sampler settings (α and K, as separate sweeps) and FSM reduction on three maize genes against motif [MA0267.1](https://jaspar.elixir.no/matrix/MA0267.1/).

## Prerequisites

- **Python 3.11+** (tested on 3.12)
- **[Poetry](https://python-poetry.org/docs/#installation)**: dependency and CLI management
- **[FIMO](https://meme-suite.org/meme/doc/fimo.html)** (MEME Suite): must be on `PATH`; used when exporting solutions to annotate motif hits (tested with 5.5.8)
- **Network**: first run for a new JASPAR motif ID fetches the matrix via `pyjaspar` (cached under `bio_data/motifs/` afterward)

Run all commands from the repository root.

## Install

```bash
poetry install
fimo --version   # verify FIMO is on PATH
```

## Quick start

Smoke-test on a single gene:

```bash
poetry run design-seq -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt -m MA0267.1 --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt -k 50 --sampler-alpha 1.0 --reduce-fsm-by 0.875
```

Outputs appear under `designer_results/` (see [Outputs](#outputs)).

## Verify install

```bash
poetry run pytest pareto_designer/tests/
```

Tests that require FIMO are skipped automatically when the binary is missing.

## Reproduce reference experiment

The canonical experiment is [configs/pareto_experiment_ma0267.yaml](configs/pareto_experiment_ma0267.yaml): three maize genes, motif [MA0267.1](https://jaspar.elixir.no/matrix/MA0267.1/), and three parameter sweeps over sampler α, sampler K, and FSM size. Bundled inputs are already in `bio_data/`.

1. Install dependencies (see above).
2. Run sweeps. Existing runs are skipped when `results_metadata.json` is present; use `--force` to re-run. Add `--export` to write the Excel report after the sweeps:

```bash
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml --export
```

To try one sweep first (recommended before the full grid):

```bash
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml --sweep alpha --export
```

3. Open `designer_results/pareto_experiment_report.xlsx`.

Re-render plots or the report from existing results without re-running the optimizer:

```bash
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml --dry-run --export
```

Add `--skip-render-per-solution` to skip per-solution HTML and heatmaps (and the target-sequence FIMO/heatmap pass) for a faster plot-only re-render of Pareto PNGs and comparison files.

The full sweep runs all combinations in the YAML (42 design runs across 3 genes: 7 α × 3 genes, 3 K × 3 genes, 4 FSM sizes × 3 genes). Expect long runtimes for the complete grid.

## Data

Reference inputs live under `bio_data/`. Run outputs go to `designer_results/` (gitignored).

### Target sequences

`bio_data/zea_mays_genes/` holds maize target sequences as `*.txt`. Each file is raw DNA with `*` marking the CDS start.

File names follow `{gene_id}_{strand}_{start}_{end}`: Ensembl gene ID, strand (`1` forward, `-1` reverse), then 1-based start and end coordinates of the fetched genomic window (including any upstream/downstream padding from `gene-fetch`).

Included genes:

- `Zm00001eb052570_-1_265197378_265198704`
- `Zm00001eb186060_1_154283147_154284746`
- `Zm00001eb319980_-1_150436154_150437498`

Additional `*.txt` files can go in the same folder; pass a directory to `-s` to run a batch.

### Motifs

`bio_data/motifs/<matrix_id>/` caches MEME format motif files (e.g. `motif.meme`). Running `gene-fetch` also writes `significant_patterns.txt` there; the bundled design runs only require `motif.meme`.

### Codon usage

Yeast codon frequencies for the maize-in-yeast case study:

- `bio_data/codon_usage/saccharomyces_cerevisiae.txt`: codon frequencies (`CODON FREQUENCY` per line; U or T)
- `bio_data/codon_usage/saccharomyces_cerevisiae.costs.csv`: derived codon costs (`Codon,Cost`), written when a design run builds the cost function

### Fetch new target sequences

`gene-fetch` downloads a CDS window from Ensembl, writes `.fa` and `.txt` under `bio_data/sequences/`, and annotates motif hits with FIMO:

```bash
poetry run gene-fetch --species zea_mays --gene-id Zm00001eb052570 --motif-id MA0267.1
```

Copy the resulting `.txt` into `bio_data/zea_mays_genes/` (defaults: 500 bp upstream and downstream of the CDS). Requires network access to Ensembl.

## Usage

| Command | Use for |
|---------|---------|
| `run-experiment-sweeps` | Structured sampler (`α` / `K`) and FSM sweeps from YAML; `--export` writes the Excel report |
| `design-seq` | Ad-hoc single invocation (custom CLI flags) |
| `gene-fetch` | Fetch a gene region from Ensembl and annotate motif hits |

Multi-line shell examples below use bash line continuations (`\`). On PowerShell, join into one line or use backtick (`` ` ``) instead of `\`.

### Structured sweeps

```bash
# Run all sweeps (skips existing results_metadata.json)
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml

# Single sweep, force re-run, export report after
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml --sweep alpha --force --export
```

#### Config

See [configs/pareto_experiment_ma0267.yaml](configs/pareto_experiment_ma0267.yaml) for the reference experiment:

- `fixed`: shared inputs (target sequences, motif, codon usage, `binding_score_space`, `hit_pval`, `cost_params` {α, β, w}, results root)
- `sweeps.alpha` / `sweeps.k`: sampler settings; each fixes the other sampler dimension and FSM reduction, then varies `sampler_alpha` or `K`
- `sweeps.alpha.comparison_groups`: named subsets of `sampler_alpha` values for separate alpha-sweep Pareto comparison PNGs (e.g. `const` vs `log_pos`)
- `sweeps.fsm_size`: FSM reduction; varies `reduce_fsm_by`
- Unknown keys are rejected (strict validation)

#### Excel report

Written to `{results_root}/pareto_experiment_report.xlsx` with six sheets: **Overview**, **Summary**, three sweep sheets (**Sweep alpha**, **Sweep K**, **Sweep FSM size**), and **Solutions**.

**Summary** lists all design runs (sorted by `seq_id`, `fsm_size` descending, `K` descending).

Each sweep sheet lists that sweep's design runs (same sort order).

Binding approximation error (reduced vs origin FSM scores), using the design score space (`logexp`: squared error of `exp(score)`):

Per `k`-mer window (within one solution):
- **kmer_binding_score_mse**: mean score-space squared error over motif-length windows in the solution
- **kmer_binding_score_err_std**: standard deviation of those per-window errors

Per run (across Pareto-optimal solutions):
- **kmer_binding_score_mse_mean** / **kmer_binding_score_mse_solution_std**: mean and std of per-solution `kmer_binding_score_mse`

Per FSM (all de Bruijn `k`-mers / states):
- **fsm_binding_score_err**: the same mean score-space squared error, averaged uniformly over every origin `k`-mer. Designed-sequence kmer MSE sits on this scale; remaining differences come from which `k`-mers appear in the solution.

Each design run also writes `kmer_binding_score_mse_histogram.png`: a histogram of per-solution `k`-mer binding score MSE, with the solution mean as a solid line and FSM MSE as a dashed line.

### Ad-hoc runs (`design-seq`)

Examples below explore one parameter dimension at a time on a single gene. Parameter values differ slightly from the YAML experiment; use `run-experiment-sweeps` to match the reference config exactly.

`--reduce-fsm-by 0` keeps the origin de Bruijn FSM. `0.75`, `0.875`, and `0.9375` are 4-fold, 8-fold, and 16-fold reduction. Add `--dry-run` to re-render outputs without running the optimizer.

**Alpha sweep** (K=100, 8-fold FSM):

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 100 \
  --sampler-alpha 0.0 1.0 1.0_log_pos 2.0_log_pos \
  --reduce-fsm-by 0.875
```

**K sweep** (α=1.0 and 1.0_log_pos, 8-fold FSM):

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 50 100 150 \
  --sampler-alpha 1.0 1.0_log_pos \
  --reduce-fsm-by 0.875
```

**FSM reduction sweep** (after choosing a sampler, e.g. `-k 100 --sampler-alpha 1.0`):

```bash
poetry run design-seq \
  -s bio_data/zea_mays_genes/Zm00001eb052570_-1_265197378_265198704.txt \
  -m MA0267.1 \
  --codon-usage bio_data/codon_usage/saccharomyces_cerevisiae.txt \
  -k 100 \
  --sampler-alpha 1.0 \
  --reduce-fsm-by 0 0.75 0.875 0.9375
```

## Outputs

Each run is written under:

```
designer_results/<gene>/<cost_params>/<motif>/<fsm_id>/<sampler_params>/
```

Comparison files for one CLI invocation are written at the common parent of those run directories (never inside an individual run folder):

- `pareto_frontiers.png` from `design-seq`, or sweep-specific names from `run-experiment-sweeps`:
  - `sweep_alpha_K100_const_pareto_frontiers.png`, `sweep_alpha_K100_log_pos_pareto_frontiers.png`
  - `sweep_K_alpha_1.0_log_pos_pareto_frontiers.png`
  - `sweep_fsm_K100_alpha_1.0_log_pos_pareto_frontiers.png`
- `pareto_comparison.csv`: per-run `K`, `alpha`, `log_pos`, `fsm_size`, `reduce_fsm_by`, `k`-mer binding score MSE aggregates, and FSM reduction error

Pareto PNG plots use **lines** with horizontal dashed gray lines at cumulative binding thresholds for 1–3 motif hits. Reduced-FSM runs show a dashed origin (de Bruijn FSM) binding line on per-run `pareto_frontier.png` and on FSM size sweep comparison plots (same color as the solid reduced-FSM line).

All sequences from the same `design-seq` run are also collected in `designer_results/pareto_comparison.csv` (includes a `seq_id` column).

# Multiple Objective DNA Sequence Design

DNA sequence design using multiple objective optimization to jointly minimize functional interference and unintended binding.

## Install

```bash
poetry install
```

FIMO is required to annotate motif hits on designed sequences.

## Data

Reference inputs are tracked under `bio_data/`. Run outputs go to `designer_results/` (gitignored).

- `bio_data/zea_mays_genes/` — maize target sequences (`*.txt`). Each file is raw DNA with `*` marking the CDS start. Included genes:
  - `Zm00001eb052570_-1_265197378_265198704`
  - `Zm00001eb186060_1_154283147_154284746`
  - `Zm00001eb319980_-1_150436154_150437498`
- `bio_data/codon_usage/saccharomyces_cerevisiae.txt` — yeast codon frequencies (`CODON FREQUENCY` per line; U or T)
- `bio_data/codon_usage/saccharomyces_cerevisiae.costs.csv` — derived codon costs (`Codon,Cost`), written when a design run builds the cost function

The examples below use maize gene `Zm00001eb052570_-1_265197378_265198704` and JASPAR motif [MA0267.1](https://jaspar.elixir.no/matrix/MA0267.1/). The YAML experiment uses the whole `zea_mays_genes/` directory. Additional `*.txt` files can go in the same folder; pass a directory to `-s` to run a batch.

## Experiments

Structured parameter sweeps use a YAML config and dedicated CLI commands. Ad-hoc runs still use `design-seq`.

### Parameter sweep workflow

```bash
# Run all sweeps (skips existing results_metadata.json)
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml

# Single sweep, force re-run, export report after
poetry run run-experiment-sweeps -c configs/pareto_experiment_ma0267.yaml --sweep alpha --force --export

poetry run export-designer-results -c configs/pareto_experiment_ma0267.yaml
```

### When to use which command

| Command | Use for |
|---------|---------|
| `run-experiment-sweeps` | Structured alpha / K / FSM sweeps from YAML |
| `export-designer-results` | Build Excel report from completed JSON results |
| `design-seq` | Ad-hoc single invocation (custom CLI flags) |

### YAML config

See [configs/pareto_experiment_ma0267.yaml](configs/pareto_experiment_ma0267.yaml) for the reference experiment:

- `fixed` — shared inputs: target sequences, motif, codon usage, cost params, results root
- `sweeps.alpha` / `sweeps.k` / `sweeps.fsm_size` — each sweep fixes all but one dimension (`k`, `sampler_alpha`, or `reduce_fsm_by`)
- Unknown keys are rejected (strict validation)

### Excel report (`pareto_experiment_report.xlsx`)

Six sheets: **Overview**, **Summary**, three sweep sheets (**Sweep alpha**, **Sweep K**, **Sweep FSM size**), and **Solutions**.

**Summary** lists all design runs (sorted by `seq_id`, `fsm_size` descending, `k` descending) with no correlations or charts.

Each sweep sheet lists that sweep's design runs (sorted by `seq_id`, `fsm_size` descending, `k` descending), CORREL formulas on the right, and bar charts in one row below.

Column **binding_score_mse** is the mean squared proxy binding error per Pareto solution; **binding_score_rmse** is its square root. Bar charts cluster one bar per sequence at each swept value. On the FSM size sheet, correlations compare **fsm_size** and **fsm_err** against Hypervolume, binding_sse, and binding_mse.

Report default path: `{results_root}/pareto_experiment_report.xlsx`.

### Ad-hoc runs (`design-seq`)

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

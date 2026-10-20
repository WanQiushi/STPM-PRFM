# Physics-guided bridge thermal-response forecasting

Code and figures accompanying **Physics-guided knowledge fusion for frequency-aware forecasting of bridge thermal responses**.

[中文说明](README.zh-CN.md) · [Experiment map](docs/EXPERIMENTS.md) · [Data contract](docs/DATA.md) · [Figure index](docs/FIGURES.md)

Machine paths, account identifiers and saved process IDs have been removed from the public configuration. Use the [runtime configuration guide](docs/RUNTIME.md) and `tools/run_experiment.py` to select local paths. D1 normalization values are private inputs and are not distributed.

This project couples a Shape-aware Temperature Prediction Module (STPM) with a Physics-inspired Response Fusion Module (PRFM) for 24-hour, multi-location response forecasting.

## Contents

```text
experiments/    Experiment-specific training, evaluation, configurations and dependencies
plotting/       Portable rendering scripts for the manuscript's five data figures
figures/        The nine figures referenced by the current manuscript
results/        Current manuscript tables and supporting result summaries
tools/         Figure-rendering entry point
docs/          Experiment, data, environment, and source documentation
local_assets/  Local-only frozen originals, plot data and source-path records (Git-ignored)
outputs/       Generated outputs (Git-ignored)
```

The experiment directories preserve the different protocols used in the paper. In particular, the independently retrained D1 main comparison and the fixed-response ablation reference are separate experiments.

## Main results

| Experiment | Response RMSE, mean ± sample SD |
|---|---:|
| D1, full 109-day test period, five independently trained PRFM pairs | 16.851 ± 0.470 mm |
| D2, full 60-day test period | 20.972 ± 0.475 mm |
| D1, one-month supervision budget, common 39-window test cohort | 17.736 ± 1.057 mm |
| D1 PRFM with measured future temperature, 60 test days | 5.8882 ± 0.1857 mm |

The one-month result reduces RMSE by 14.66% relative to LSTM in that budgeted evaluation. D1 main-comparison PRFMs are independently initialized, pretrained, and adapted, using a common fixed out-of-fold temperature cache. The reported standard deviations use five seed-level scores, with `ddof=1`, without prediction ensembling.

## Re-render the data figures

Install an appropriate PyTorch build for the intended machine, then the project dependencies:

```bash
python -m pip install -r requirements.txt
python tools/render_figures.py --figure all \
  --data-dir local_assets/figure_data --output outputs/figures
```

The local preparation includes the required figure caches. They are Git-ignored and are **not included in a clone**. Figures 1–4 are supplied as final vector PDFs and PNG previews; Figures 5–9 have executable rendering scripts. Rendering writes to `outputs/figures`, preserving the manuscript exports.

## Training

The exact experiment entry points and required inputs are listed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). For example, after supplying the prepared D1 observations and fixed OOF caches:

```bash
python tools/run_experiment.py --runtime config/runtime.local.json experiments/d1_prfm/train_prfm.py \
  --seed 2024 --data data/d1/minute_complete.npz \
  --output outputs/d1_prfm/seed2024
```

Training code retains the original stage objectives, window definitions and checkpoint selection. Configure the public path tokens and required private inputs before running an experiment. `config/runtime.json` supplies neutral defaults; local overrides belong in a Git-ignored `*.local.json` file. This source release is not a single interchangeable training pipeline.

## Data and software

D1 observation arrays and trained checkpoints are not bundled for GitHub upload. The local figure caches and editable drawing sources are excluded by `.gitignore`. The complete **D2 synthetic dataset** is included at `experiments/d2_pipeline/data/minute_synthetic.npz`, together with its construction code, fixed response coefficients and split metadata. Its [dataset README](experiments/d2_pipeline/data/README.md) describes the fields and loading example. D1 datasets, training/evaluation arrays and plotting caches are excluded. See [docs/DATA.md](docs/DATA.md) for the schemas and [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the recorded environments.

Upstream licenses are retained with their source copies. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new license has been assigned to the authors' code by this packaging operation.

## Preparation status

The five data-figure exports were executed successfully during source preparation. The public cleanup introduces a common runtime entry and removes machine information and private D1 statistics. It preserves numerical algorithm constants, the D2 arrays and manuscript result values. Training was not rerun, and no repository was uploaded to GitHub during preparation.

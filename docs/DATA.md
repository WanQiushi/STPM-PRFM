# Data contracts and local assets

## D1: monitored bearing displacement

D1 contains 1,051,200 minute samples spanning 730 days. Columns are temperature, Pier 1, Pier 2, Pier 3, and Pier 4. The response labels refer to four downstream bearings. The publication-facing code uses anonymous names; original identifiers and source-path mappings are retained only in the local frozen copies.

| Field | Shape | Meaning |
|---|---|---|
| `values` | `[1051200, 5]` | Completed history values; temperature in °C and displacement in mm |
| `valid_observed` | `[1051200, 5]` | Retained measured-label mask |
| `input_available_at` | `[1051200, 5]` | Minute index at which the value becomes available to a causal forecast |
| `timestamps` | `[1051200]`, when provided | Minute timestamps |

The chronological split contains 511 training days, 110 validation days and 109 test days. The training input screens differ by stage; use the saved stage-specific origin lists. The small-sample experiment uses its own common 39-window test cohort. The measured-future-temperature experiment uses 60 eligible test days.

Reading example:

```python
import numpy as np
with np.load("data/d1/minute_complete.npz", allow_pickle=False) as data:
    values = data["values"]
    observed = data["valid_observed"]
    available_at = data["input_available_at"]
temperature = values[:, :1]
response = values[:, 1:]
```

D1 source CSVs, original sensor dictionaries and the full observation arrays are not included in the upload set. `experiments/d1_preprocessing/` preserves the construction stages with anonymous column aliases. Raw CSVs supplied to those publication-facing scripts must use the corresponding aliases.

## D1 complete-period evaluation cache

The cache builder in `experiments/d1_full_period/prepare.py` constructs 219 daily histories. Required arrays include:

- `origins.npy`: 110 validation origins followed by 109 test origins.
- `history_physical.npy`: `[219, 43200, 5]`.
- `history_mask.npy`: historical measurement indicators.
- `target_physical.npy` and `target_mask.npy`: `[219, 1440, 5]`.
- `temperature_values.npy` and `temperature_observed.npy`.
- `manifest.json`: split and reconstruction metadata.

Reported response scores pool valid measured nodes; interpolated history values are not newly observed evaluation labels.

## D2: controlled thermal deflection

The complete frozen D2 dataset is included at `experiments/d2_pipeline/data/minute_synthetic.npz` (82,345,344 bytes). This is the existing dataset used by the paper's D2 observation figure, copied without regenerating or modifying its arrays. See the [dataset README](../experiments/d2_pipeline/data/README.md) for all fields and a loading example. The upload contains this D2 NPZ only; D1 observation arrays and all training/evaluation/plot caches remain excluded.

The generator is `experiments/d2_pipeline/scripts/generate_dataset.py`. Its inputs are `generation_config.json` and `thermal_response.json`. The latter contains the fixed seven-location finite-element response coefficients.

The generator uses seasonal/daily harmonics, stochastic weather, two thermal states and independent observation noise. Four rounded harmonic parameters were derived from D1 training-period temperature statistics; no D1 measured time series is contained in the synthetic data or bundled with this release. The complete parameter provenance remains in `generation_config.json`.

The generated minute NPZ contains 576,000 rows and eight columns: one temperature driver and seven deflections, at 1/8, 1/4, 3/8, 1/2, 5/8, 3/4 and 7/8 of the main span. Fields include `values`, `clean_values`, `thermal_states_c`, `girder_contribution_mm`, `cable_contribution_mm`, `valid_observed`, `input_available_at`, `channels`, `units`, and `timestamps`. Training/validation/test lengths are 280/60/60 days. Response evaluation uses 96 common 15-minute nodes per day.

## Figure inputs

The local folder `local_assets/figure_data/` contains:

| Folder | Files | Used by |
|---|---|---|
| `d1/` | `full109_plot_data.npz`, `component_plot_data.npz` | Figures 5 and 6 |
| `d2/` | `plot_data_full60_seed2024.npz` | Figure 8 |
| `d2/` | `plot_data_fig6_seed2024.npz`, `selected_day_recomputed.json`, `extracted_metrics.json` | Figure 9 and selection provenance |
| `d2_dataset/` | `minute_synthetic.npz` | Figure 7 |

These caches are excluded from Git. Retaining them locally supports exact re-rendering without retraining. Figure 6 uses the fixed 14 November illustration; Figure 9 uses the archived day-391 illustration. Their selection metadata is preserved separately from the full-period quantitative evaluation.

For a newly distributed dataset, retain this schema, its construction code and the actual origin lists in the dataset directory. Do not silently substitute a different cohort, measurement mask or normalization.

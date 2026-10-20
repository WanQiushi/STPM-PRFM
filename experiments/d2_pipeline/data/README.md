# D2 synthetic thermal-deflection dataset

Version: `synthetic_deflection_abaqus_D2_1_20261012`.

`minute_synthetic.npz` contains the existing 400-day, minute-resolution synthetic dataset used in the manuscript: **576,000 rows × 8 channels**. This release copies the frozen dataset; it does not generate a new realization. No D1 measured time series, D1 observation arrays or D1 training/evaluation caches are included.

## Construction and provenance

The temperature driver combines seasonal and daily harmonics with stochastic weather. Two first-order thermal states with time constants of 2 and 4 hours drive girder and cable contributions through fixed finite-element response coefficients. Independent Gaussian noise is added to temperature and deflection.

Four rounded harmonic parameters originate from D1 training-period temperature statistics. The generator uses these scalar settings, not measured D1 sequences. Seed: `20241009`. The timestamps are fictional and do not identify monitoring dates.

- Generator: [`../scripts/generate_dataset.py`](../scripts/generate_dataset.py).
- Full settings and provenance: [`../generation_config.json`](../generation_config.json).
- Fixed response coefficients: [`../thermal_response.json`](../thermal_response.json).
- Frozen statistics: [`summary.json`](summary.json).
- Stage-specific origins and contracts: `window_origins.json`, `baseline_window_origins_499.json`, `h1_window_origins_199.json`, `model_window_contracts.json`, and `protocol.json`. These retain the different training-window definitions used by each stage.
- Distribution scope and local preparation: [`../../../docs/DATA.md`](../../../docs/DATA.md).

## Fields

| Key | Shape | Meaning |
|---|---|---|
| `values` | `[576000, 8]` | Noisy observations: temperature in °C, then seven deflections in mm |
| `clean_values` | `[576000, 8]` | Synthetic observations before noise |
| `noise` | `[576000, 8]` | Added observation noise |
| `thermal_states_c` | `[576000, 2]` | Girder and cable thermal states in °C |
| `girder_contribution_mm` | `[576000, 7]` | Girder contribution to deflection |
| `cable_contribution_mm` | `[576000, 7]` | Cable contribution to deflection |
| `timestamps` | `[576000]` | Fictional minute timestamps, 2020-01-01 to 2021-02-03 |
| `dates` | `[400]` | Fictional daily dates |
| `channels` | `[8]` | `SIM_T`, then `SIM_D01`–`SIM_D07` |
| `units` | `[8]` | `degC`, followed by seven `mm` entries |
| `observed`, `valid_observed`, `available` | `[576000, 8]` each | Boolean observation/availability indicators |
| `input_available_at` | `[576000, 8]` | Earliest available minute index |
| `input_indices` | `[8]` | Indices 0–7 |
| `output_indices` | `[7]` | Deflection indices 1–7 |
| `split_days` | `[3]` | Training/validation/test: 280/60/60 days |

The seven deflection locations are at 1/8, 1/4, 3/8, 1/2, 5/8, 3/4 and 7/8 of the main span. Chronological row intervals, using Python half-open indexing, are train `[0, 403200)`, validation `[403200, 489600)`, and test `[489600, 576000)`. Forecast-window eligibility is defined separately by the saved stage-specific origin lists.

## Read from the repository root

```python
import numpy as np

with np.load(
    "experiments/d2_pipeline/data/minute_synthetic.npz",
    allow_pickle=False,
) as data:
    values = data["values"]
    temperature = values[:, :1]
    deflection = values[:, 1:]
    train = values[:403200]
    validation = values[403200:489600]
    test = values[489600:]
```

All observation and contribution arrays are stored as float32. Re-running the generator writes into this directory, including its dataset and generated metadata; use a separate working copy if preserving the distributed files.

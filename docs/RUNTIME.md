# Runtime configuration

The public code contains neutral configuration tokens rather than the original computer/server paths. It is distributed as readable source, not as encrypted or obfuscated code.

## One entry point

From the repository root:

```bash
python tools/run_experiment.py experiments/d1_prfm/train_prfm.py --help
```

For an actual experiment, copy `config/runtime.json` to `config/runtime.local.json`, fill in the required paths and private inputs, then use:

```bash
python tools/run_experiment.py --runtime config/runtime.local.json \
  experiments/d1_prfm/train_prfm.py \
  --seed 2024 --data data/d1/minute_complete.npz \
  --output outputs/d1_prfm/seed2024
```

The launcher sets the configured physical GPU, makes the runtime module available to subprocesses, and runs from the repository root. Existing GPU ownership checks remain in the experiment scripts. Inspect resource availability before training.

`python -m pip install -e .` is also available for direct script execution; it installs the small configuration package. Install model dependencies separately from `requirements.txt`.

## Path tokens

| Token | Purpose |
|---|---|
| `${WORKSPACE}` | Experiment workspace with the expected stage-specific relative layout |
| `${DATA}` | Private or locally prepared input data |
| `${OUTPUTS}`, `${RESULTS}`, `${RECORDS}` | Generated products and records |
| `${FE_MODEL}`, `${FE_WORK}` | Privately supplied finite-element model and derived working directory |
| `${TOOLS}`, `${COMPILER}`, `${MSVC}`, `${ABAQUS}`, `${CONDA}` | Optional external tools |
| `${MANUSCRIPT}`, `${TEMP}`, `${ASSETS}` | Optional document, temporary and additional input locations |
| `${PYTHON}` | Current Python executable |
| `${USER}` | Current execution account, read locally when needed by an ownership check |
| `${GPU}` | Physical GPU selected in the runtime configuration |

Relative configured paths resolve against the repository root. The launcher does not construct missing historical workspace trees, download private data, or supply pretrained checkpoints.

## D1 private inputs

D1 observation arrays, cache arrays and normalization values are excluded. Configuration fields such as `${PRIVATE_D1:stat_0049}` refer to entries in the local configuration's `private_d1` object. `config/private_d1_fields.json` maps each identifier to its public configuration file and field; it contains no statistic values.

Provide statistics calculated from the prescribed training interval when running with authorized D1 inputs. Missing private entries raise `KeyError`; there is no substituted statistic or fabricated dataset. Preserve the original split and preprocessing rules for reproducibility.

D1 manuscript figures and aggregate result tables are included. They disclose the plotted observations and reported statistics; excluding raw datasets does not conceal information already shown in those figures.

## Files to keep local

- `*.local.json` and `hosts.json`
- `local_assets/`, `data/`, `outputs/`, `tmp/`
- Private CSV/Excel/Parquet/HDF5 arrays, checkpoints and finite-element model files

The public D2 NPZ at `experiments/d2_pipeline/data/minute_synthetic.npz` is the sole dataset exception. Runtime-generated configuration, logs and records can contain the current machine's paths, so keep them out of subsequent uploads. A new release must be checked again after adding files or running experiments.

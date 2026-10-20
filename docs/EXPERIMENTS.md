# Experiment-to-code map

The current manuscript is the authority for which experiments belong in its tables. Older model identifiers are retained where needed to read frozen configurations; `A2`/`S0` denote the full temperature/pipeline variant, `H1` the response implementation, and `D21` the manuscript's D2 dataset.

| Manuscript experiment | Directory | Computational entry points |
|---|---|---|
| D1 STPM, five seeds | `experiments/d1_stpm/` | `source/train_contract_v9.py` for initialization; `source/train.py` for the selected temperature recipe; `runs/seed2024`–`seed2028` contain actual run configuration/metadata |
| D1 independent PRFM | `experiments/d1_prfm/` | `train_prfm.py`, `evaluate.py`, `summarize.py` |
| D1 OOF-driven adaptation | `experiments/d1_oof_adaptation/` | `source/prepare.py`, `source/data.py`, `source/train.py` and the associated frozen project |
| D1 complete-period histories and extrema | `experiments/d1_full_period/` | `prepare.py`, `evaluate.py`, `extrema.py`, `update_statistics.py`, `export_native_components.py` |
| D1 budgeted retraining | `experiments/d1_small_sample/` | `a2/train.py`, `baseline/train.py`, `build_configs.py`, `aggregate_results.py` |
| Added PCFNet/LSTM budgeted runs | `experiments/d1_small_sample_additional/` | `baseline/train.py`, `aggregate_results.py` |
| Original D1 joint baselines | `experiments/d1_baselines/`, `experiments/baseline_training/` | Frozen `train.py` entries and model-specific configurations |
| D1 original core ablations | `experiments/d1_core_ablations/` | Recorded temperature/response training and `evaluate_core.py` |
| D1 added response ablations | `experiments/d1_response_ablations/source/` | `gpu1_parallel_executed_source/train_h1_parallel.py`, `evaluate_parallel.py`, `summarize.py`; original response dependencies alongside |
| D1 measured-temperature PRFM | `experiments/d1_known_response/` | `source/prepare.py`, `source/train_one.py`, `source/research_model.py`; the main conditional comparison uses arm `O0` |
| D1 measured-temperature baselines | `experiments/d1_known_temperature/` | `source/train_baseline.py`, `known_temperature_models.py`, `source/adaptations/` |
| D2 dataset | `experiments/d2_pipeline/` | `scripts/generate_dataset.py`, `generation_config.json`, `thermal_response.json`, `data/` split metadata |
| D2 STPM/PRFM | `experiments/d2_pipeline/a/` | Per-seed `run.py`, `prepare_data.py`, `prepare_forecast_cache.py`, `response_training.py`, `evaluate.py`, and `temperature_code/` |
| D2 baselines | `experiments/d2_baselines/` | `experiments/baseline_joint8/train.py`, `experiments/baseline_moderntcn_joint8/train.py`; PCFNet/LSTM trainers under `baseline_training/` |
| D2 response coefficients and static validation | `experiments/d2_finite_element/` | The Abaqus preparation/extraction scripts and validation source |
| D1 observation preprocessing | `experiments/d1_preprocessing/` | Alignment, anomaly handling, LATC filling, completion and four-location construction scripts |

## Protocols that must remain distinct

- **D1 main comparison:** five PRFMs independently initialized, pretrained and adapted, paired with STPM seeds 2024–2028. Common fixed OOF temperature cache; no test-based checkpoint selection.
- **D1 ablation reference:** one fitted PRFM per response configuration, paired with five temperature seeds. Its `16.781 ± 0.449 mm` full-model reference is intentionally different from the independently retrained main result.
- **D1 supervision budgets:** 9/6/3/1 months, with a fixed 26-window validation cohort and 39-window test cohort. Budget-specific protocols retain the actual origins and normalization. Additional PCFNet/LSTM runs complete the manuscript's current baseline set.
- **D1 measured temperature:** 222 training windows, 52 validation days, 60 test days, 15-minute scoring. The conditional comparison reuses the independently trained `O0` response models; it does not use the separately investigated original multi-stage recipe.
- **D2:** 400 days, 280/60/60 split; 499 final-temperature/response pretraining windows, 137 OOF adaptation windows; seven response channels.

## Main D1 PRFM example

Supply a D1 NPZ with the contract in `DATA.md`. The local preparation contains the fixed train/validation OOF caches under `experiments/d1_prfm/cache/`; these binary files are Git-ignored.

```bash
python tools/run_experiment.py --runtime config/runtime.local.json experiments/d1_prfm/train_prfm.py \
  --seed 2024 --data data/d1/minute_complete.npz \
  --output outputs/d1_prfm/seed2024
```

The script uses the first CUDA-visible device. Inspect GPU ownership and resources before executing training. Do not change batch sizes, objectives, origins or normalization to accommodate a different machine.

For paired evaluation, `evaluate.py` also needs the complete-period `.npy` dataset, a trained response checkpoint, and the same-seed STPM checkpoint/source specified by `config/temperature_jobs.json`.

## D2 generation

```bash
python tools/run_experiment.py experiments/d2_pipeline/scripts/generate_dataset.py
```

This creates the synthetic arrays under the D2 directory's `data/`. It uses the archived numerical coefficients, seeds, time constants and split definitions. Generating the thermal response coefficients from the finite-element model requires Abaqus and the corresponding model inputs.

## Configuration paths

The public experiment files retain stage-specific directory relationships through neutral path tokens. `if_release/runtime.py` resolves these tokens using `config/runtime.json` or a private local configuration. Original machine paths and account identifiers are not distributed.

Before using a frozen training entry:

1. Provide the observation/cache/checkpoint inputs named by that experiment.
2. Set the configuration's input and output paths to the local paths; retain numerical fields and saved origins.
3. For scripts with a historical project-root constant, point it at the matching local experiment/runtime tree.
4. Use a new output directory. Do not overwrite archived results.

See `RUNTIME.md` for the launcher and path settings. D1 normalization fields refer to private inputs described by `config/private_d1_fields.json`; their values are excluded. For the measured-temperature baseline entry, copy `experiments/d1_known_temperature/hosts.example.json` to the Git-ignored `hosts.json` in the same directory and set the prepared cache and response-run paths. The selected run configurations use the neutral `runtime` host key.

Upstream libraries include source files for models beyond the eight selected manuscript baselines. The experiment configurations and current manuscript tables determine the evaluated set; a vendored library file is not an additional reported experiment.

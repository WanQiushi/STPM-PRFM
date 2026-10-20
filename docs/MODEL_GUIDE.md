# Model source guide

## STPM

`experiments/d1_stpm/source/models.py`, `ode_core_v9.py`, `temperature_ablation.py`, and `fixed_trend.py` contain the temperature architecture and the selected shape/trend recipe. `train_contract_v9.py` supplies the initialization/window machinery; `train.py` implements the base/residual stages. Actual final-stage settings are stored under `runs/seed2024`–`seed2028`.

The temperature model uses recent minute observations, 30 daily means/coverage values and calendar features. A latent ODE generates a normalized shape, historical level/amplitude anchor it, and a residual network corrects the trajectory.

## D1 PRFM

The clearest current four-location entry is:

- `experiments/d1_prfm/engine/response_model.py`: `HybridKqsz`.
- `experiments/d1_prfm/engine/response.py`: pretraining, forecast-driver adaptation and validation selection.
- `experiments/d1_prfm/engine/data.py`: window sampling and normalization.
- `experiments/d1_prfm/config/`: the actual pretraining/adaptation settings, origins and normalization.
- `experiments/d1_prfm/vendor/h1_project/`: the original decomposition and staged-training dependencies.

Do not remove the apparently nested dependency directories without replacing their dynamic imports. They are part of the executed model definition.

## D2 PRFM

The seven-location implementation is `deflection_h1.py` inside each D2 response run directory. It is not interchangeable with the four-location D1 class. The D2 pretraining and forecast adaptation code is `response_training.py`; the archived `H1499` directories distinguish the 499-window response training configuration from earlier intermediate stages.

## Baselines and conditional adapters

The eight manuscript baselines are PCFNet, ModernTCN, vanilla LSTM, DLinear, TimeXer, PatchTST, iTransformer and TimesNet-LR. Preserve the low-rank TimesNet implementation label and the recorded PCFNet numerical adapter.

Known-temperature baselines use `experiments/d1_known_temperature/source/known_temperature_models.py`, which adds the common 30-parameter future-temperature/calendar projection. `source/adaptations/` retains the out-of-place normalization changes required for gradients through these conditional inputs. The original frozen backbone copies are also retained.

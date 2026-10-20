# D1 known-temperature H1 versus eight manuscript baselines

Five seeds2024-2028;60 common observed-temperature test days;15-minute response grid;mean±sampleSD,mm.
H1:reuse five completed full-architecture O0 runs,no retraining;forty newly trained conditional baselines.
All methods receive the same30day history and measured next24h temperature;future displacement labels are not inputs.

| Model | RMSE (mm) | MAE (mm) | Change versus H1 | H1 better days |
| --- | --- | --- | --- | --- |
| H1 (reused full architecture) | 5.8882 ± 0.1857 | 4.5635 ± 0.1843 | +0.00% | — |
| PCFNet | 10.8741 ± 0.2892 | 8.3936 ± 0.2945 | +84.67% | 59/60 |
| ModernTCN | 7.1527 ± 0.1427 | 5.2750 ± 0.1068 | +21.47% | 43/60 |
| LSTM | 5.4108 ± 0.3068 | 3.8916 ± 0.1781 | -8.11% | 24/60 |
| DLinear | 7.3449 ± 0.2051 | 5.3373 ± 0.1629 | +24.74% | 44/60 |
| TimeXer | 8.8607 ± 0.5693 | 6.7424 ± 0.4022 | +50.48% | 55/60 |
| PatchTST | 9.1442 ± 0.3837 | 6.9732 ± 0.3215 | +55.30% | 55/60 |
| iTransformer | 9.1917 ± 0.2874 | 7.0159 ± 0.2940 | +56.10% | 51/60 |
| TimesNet-LR | 9.3649 ± 4.5296 | 5.9229 ± 2.1551 | +59.04% | 33/60 |

Positive differences favor H1. Calendar-block intervals and allseed/day values are inJSON.
This experiment tests whole-H1 conditional prediction performance;it does not identify a unique physical hysteresis mechanism.
The baselines use the declared common30-parameter future-covariate projection;these are conditional adaptations of the manuscript backbones.
Learning rate:H1 already trained at0.001,baselines retain0.0001. H1 retains its existing0.001 neural-rate regularization;baselines have no corresponding field.

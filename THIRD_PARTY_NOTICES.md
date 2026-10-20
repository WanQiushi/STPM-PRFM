# Third-party source notices

Original license texts are retained beside each copied source tree. Repeated experiment-specific copies retain their licenses independently.

| Upstream project | Recorded license | Example location |
|---|---|---|
| PCFNet | MIT; copyright 2025 Jsmcnhyk | `experiments/baseline_training/vendor/PCFNet_IF2026_source_20261015/LICENSE` |
| ModernTCN | MIT; copyright 2024 luodhhh | `experiments/d1_baselines/vendor/ModernTCN_20261009/LICENSE` |
| Time-Series-Library | MIT; copyright 2021 THUML @ Tsinghua University | `experiments/d1_baselines/vendor/Time-Series-Library_20261002/LICENSE` |
| Latent ODE reference source | MIT; copyright 2019 Yulia Rubanova | `experiments/d1_stpm/source/vendor_latent_ode/LICENSE` |

The manuscript uses adaptations and frozen configurations of these implementations, documented in `docs/EXPERIMENTS.md` and `docs/MODEL_GUIDE.md`. Upstream licenses do not automatically license the authors' new code or D1 monitoring observations. This local preparation has not assigned a new project-wide license or data license.

The table lists repository-level notices, not a uniform license for every vendored file. Some copied M4/N-BEATS utilities carry their own **CC BY-NC 4.0** notice from Element AI, including `experiments/baseline_training/vendor/PCFNet_IF2026_source_20261015/utils/losses.py` and the corresponding M4/loss utilities in other vendor trees. Their file-level notices remain intact. This package does not relicense those files under MIT.

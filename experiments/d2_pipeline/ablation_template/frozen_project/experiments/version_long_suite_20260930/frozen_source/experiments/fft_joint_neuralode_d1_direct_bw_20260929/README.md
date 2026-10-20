# Direct-output BW minimum verification

This experiment removes the algebraic MF output `k*q + s*z`. The predicted MF
displacement is the first Bouc--Wen Neural ODE state, while `z` is latent only.

- LF: direct temperature mapping conditioned by annual and semiannual phase.
- MF forcing: one causal thermal-lag state per support.
- MF state: direct BW displacement with OLS initialization of `A`, `beta`, `gamma`.
- Losses: separate LF/MF/full prediction terms, derivative and complex-spectrum
  MF terms, low-weight smooth extrema and loop terms.
- Stage order: fixed-boundary BW pretrain, fixed-boundary prediction warmup,
  low-learning-rate joint cutoff fine-tuning.
- Final checkpoint selection: joint-cutoff stage only. Test split remains sealed.


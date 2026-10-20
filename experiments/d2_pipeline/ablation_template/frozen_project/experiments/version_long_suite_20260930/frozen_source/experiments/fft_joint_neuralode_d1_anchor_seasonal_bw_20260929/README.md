# LF anchor and seasonal BW amplitude ablation

Two seed-2024 runs isolate the next modifications after the direct-output BW
experiment:

1. `config_anchor_only.json`: add only a causal constant LF level correction
   computed from the last historical LF displacement. No trend is extrapolated.
2. `config_anchor_seasonal_range.json`: additionally condition BW gain `A` on
   annual/semiannual phase within a bounded ±30% envelope and add an MF range
   loss with weight 0.1.

Both runs keep the learned FFT cutoff initialization fixed for every epoch
(`cutoff_lr=0` and decomposer parameters frozen), retain the same data/windows,
use seed 2024, and keep the test split sealed. The inherited final stage label is
`joint_cutoff`, but this experiment does not update either cutoff.


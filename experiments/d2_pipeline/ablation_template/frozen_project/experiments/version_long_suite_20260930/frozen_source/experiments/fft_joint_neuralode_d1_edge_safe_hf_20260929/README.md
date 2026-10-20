# Edge-safe HF decomposition experiment

This experiment replaces circular-window rFFT decomposition with:

1. an affine endpoint bridge assigned to LF;
2. 72 h reflection padding of the detrended residual;
3. padded-grid Butterworth masks followed by cropping;
4. an HF cutoff constrained to 0.5--2 h (fixed at 1 h in the first model run).

Run `decomposition_diagnostic.py` before full model training. The diagnostic
compares the previous periodic 7.376 h split with edge-safe 1 h and 2 h splits
on the fixed validation windows. The test split remains sealed.

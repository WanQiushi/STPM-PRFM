# Software environment

The experiments were executed in more than one recorded environment:

- PRFM/conditional environment: PyTorch 2.1.0 with CUDA 11.8; NumPy 1.24.3 and pandas 2.2.1.
- Paired-evaluation environment: PyTorch 2.5.1 with CUDA 12.1; NumPy 2.2.6 and pandas 3.0.2.
- Figure export: NumPy, pandas, Matplotlib and Pillow.

`requirements.txt` lists the dependency names for the selected model and plotting code. It is not a claim that a newly resolved environment is bitwise equivalent to either experimental environment. Install a PyTorch build appropriate to the machine before installing the remaining packages.

Use Python 3.10 or later for the maintained entry points; `zoneinfo` is used by experiment records. GPU training requires a compatible CUDA runtime. The finite-element scripts require an Abaqus Python environment and the original model inputs.

The frozen source trees include model-specific upstream dependencies. ModernTCN and Time-Series-Library backbones use `einops`; the selected attention modules also import `reformer-pytorch`. The recorded latent-ODE source includes `torchdiffeq` integration utilities. Matplotlib is imported by the PCFNet source.

## Verified during local preparation

- Python syntax parsing for the organized source files.
- Actual rendering of Figures 5–9 through `tools/render_figures.py`.
- Presence of all nine final manuscript figure PDFs and PNG previews.
- Original upstream license files retained with vendored code.

No training, numerical model comparison, parameter update, environment installation, or GPU job was performed as part of packaging.

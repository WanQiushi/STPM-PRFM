# Current manuscript figures

| Figure | Content | Final stem | Source |
|---|---|---|---|
| 1 | Frequency-aware motivation | `fig01_frequency_motivation` | User-supplied vector artwork |
| 2 | Overall framework | `fig02_framework_overview` | Edited framework artwork |
| 3 | STPM architecture | `fig03_stpm_architecture` | Edited architecture artwork |
| 4 | PRFM architecture | `fig04_prfm_architecture` | Edited architecture artwork |
| 5 | D1 full 109-day forecasts | `fig05_d1_full_test` | `plotting/d1/plot_fullperiod.py` |
| 6 | D1 LF/MF illustration | `fig06_d1_components` | `plotting/d1/plot_components.py` |
| 7 | D2 complete 400-day observations | `fig07_d2_observations` | `plotting/d2_dataset/plot_d2_observations.py` |
| 8 | D2 full 60-day forecasts | `fig08_d2_full_test` | `plotting/d2/plot_fullperiod.py` |
| 9 | D2 seven-location LF/MF illustration | `fig09_d2_components` | `plotting/d2/plot_selected_components.py` |

The PDF exports under `figures/` are copied from the files actually referenced by the manuscript. PNG previews exist for all nine; Figures 5–9 also have SVG exports. Historical figure numbers in frozen source comments do not override this index.

```bash
python tools/render_figures.py --figure 5 --output outputs/figures
python tools/render_figures.py --figure all --data-dir /path/to/figure_data
```

The portable scripts retain the original plotting computations, axis ranges, line styles and manuscript typography. Development-tool profiling and layout-review calls are kept in the local original script copies, rather than introducing a dependency on a private Codex skill directory.

The editable slide deck and SVG drawing sources are retained under `local_assets/editable_figures/`. They are working drawing sources; the supplied final PDFs define the exact manuscript versions.

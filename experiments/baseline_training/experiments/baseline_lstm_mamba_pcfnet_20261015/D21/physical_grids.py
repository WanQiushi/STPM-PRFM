"""D2.1 physical scores on the same fixed15/30-minute nodes as prior baselines."""
import json
import numpy as np

def score_grids(out):
    results = {}
    for split in ['validation', 'test']:
        with np.load(out / (split + '_predictions_accuracy.npz')) as data:
            error = (data['prediction'].astype(float) - data['target'].astype(float)) * data['std']
            mask = data['mask'].astype(bool)
        results[split] = {}
        for step in [15, 30]:
            e, m = (error[:, ::step], mask[:, ::step])
            count = m.sum((0, 1))
            sse = np.where(m, e ** 2, 0).sum((0, 1))
            sae = np.where(m, abs(e), 0).sum((0, 1))
            results[split][str(step)] = dict(sample_minutes=step, windows=len(e), temperature_rmse_c=float(np.sqrt(sse[0] / count[0])), temperature_mae_c=float(sae[0] / count[0]), pooled_rmse_mm=float(np.sqrt(sse[1:].sum() / count[1:].sum())), pooled_mae_mm=float(sae[1:].sum() / count[1:].sum()), pooled_mse_mm2=float(sse[1:].sum() / count[1:].sum()), channel_rmse=np.sqrt(sse / count).tolist(), observed_counts=count.tolist(), scoring_policy='Subsample existing native1min forecasts; no averaging, filtering or refitting')
    (out / 'grid_scores.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

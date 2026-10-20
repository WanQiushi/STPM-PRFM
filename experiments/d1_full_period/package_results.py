"""Package result tables, exact evaluation labels and all160 prediction caches."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
import json, tarfile, zipfile
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
jobs = _release_json_loads((HERE / 'plan.json').read_text())['jobs']
(OUT / 'PREDICTION_CACHE_README.md').write_text('# Full D1 validation/test results\n\n80 frozen model/seed runs: nine baselines and fullS0/six ablations, each seeds2024-2028.\nEach run has validation_predictions.npz [110,1440,5] and test_predictions.npz [109,1440,5].\nKeys: prediction,target,mask,origin,mean,std. Prediction/target are normalized with the original training statistics.\nRecover physical values: physical = normalized*std+mean. Channel0 is temperature in Celsius; channels1-4 are anonymized downstream bearing longitudinal displacements in mm.\nmask=True marks retained measured labels. All other target values are filled records and MUST NOT be treated as measured truth.\norigin counts minutes since2022-01-01 00:00 in the original monitoring local timestamps. Each window contains offsets0-1439 minutes.\nValidation:2023-05-27 through2023-09-13. Test:2023-09-14 through2023-12-31. Forecasts are issued daily using history available at that origin, not as a single219day open-loop forecast.\nevaluation_labels contains physical-unit target/mask and origins for all219days, validation first110 and test last109.\nThe source dataset README also describes server-side history arrays, which are not included in this delivery archive.\nInput protocol: retain original historical values available before each origin; reconstruct only unavailable historical entries from past available anchors. See manifest for per-day counts.\nFrozen weights and their normalization are unchanged; no training or reselection. All seed metrics are separate, not an ensemble.\nSummary tables/CSV, per-seed/cohort metrics and full provenance are in result_metadata.zip.\n')
with zipfile.ZipFile(OUT / 'result_metadata.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['comparison.json', 'results_table.md', 'summary_metrics.csv', 'per_seed_metrics.csv', 'per_location_metrics.csv', 'per_horizon_metrics.csv', 'legacy_reproduction_audit.json', 'progress.json', 'test_extrema_details.csv', 'test_extrema_coverage.json', 'PREDICTION_CACHE_README.md']:
        z.write(OUT / name, name)
    for name in ['README.md', 'manifest.json']:
        z.write(OUT / 'dataset' / name, 'dataset/' + name)
    for job in jobs:
        for name in ['metrics.json', 'experiment_record.md']:
            z.write(OUT / job['key'] / name, job['key'] + '/' + name)
        z.write(OUT / (job['key'] + '.log'), 'logs/' + job['key'] + '.log')
    for p in HERE.iterdir():
        if p.suffix in ['.py', '.json']:
            z.write(p, 'source/' + p.name)
with tarfile.open(OUT / 'all_prediction_caches.tar', 'w') as t:
    t.add(OUT / 'PREDICTION_CACHE_README.md', arcname='README.md')
    for job in jobs:
        for split in ['validation', 'test']:
            p = OUT / job['key'] / (split + '_predictions.npz')
            t.add(p, arcname=job['key'] + '/' + p.name)
    for name in ['origins.npy', 'target_physical.npy', 'target_mask.npy', 'manifest.json', 'README.md']:
        t.add(OUT / 'dataset' / name, arcname='evaluation_labels/' + name)
manifest = dict(model_seed_runs=len(jobs), validation_days=110, test_days=109, prediction_files=160, metadata=str(OUT / 'result_metadata.zip'), prediction_archive=str(OUT / 'all_prediction_caches.tar'), prediction_archive_bytes=(OUT / 'all_prediction_caches.tar').stat().st_size, checkpoints_or_raw_history_included=False, score_note='Model caches contain normalized prediction/target; recover physical values with array*std+mean and use mask for measured labels. evaluation_labels includes original physical labels/masks; masked-out values are imputed, not measured truth.')
(OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

"""Reuse five completed O0 runs using saved artifacts only; no Torch import."""
from pathlib import Path
import argparse
import shutil
from common import ROOT, read, dump, host_settings, now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=['local', 'server_gpu0'], required=True)
    args = parser.parse_args()
    source = Path(host_settings(args.host)['h1'])
    root = ROOT / 'h1_reference'
    root.mkdir()
    for seed in range(2024, 2029):
        old = source / f'O0_seed{seed}'
        out = root / f'H1_seed{seed}'
        out.mkdir()
        summary = read(old / 'summary.json')
        reference = dict(status='reused_completed', model='H1', seed=seed, source_group='O0', source_directory=str(old), source_checkpoint=str(old / 'best.pt'), source_config=read(old / 'config.json'), selected_epoch=summary['selected_epoch'], epochs_ran=summary['epochs_ran'], metrics={split: summary['metrics'][f'{split}_15min'] for split in ['validation', 'test']}, execution_node=summary.get('execution_node'), torch_version=summary.get('torch_version'), numpy_version=summary.get('numpy_version'), reused_at_beijing=now(), new_training=False, scope='Full architecture,previous unified total-response objective; saved15min predictions,not original staged-recipe reproduction')
        dump(out / 'summary.json', reference)
        for split in ['validation', 'test']:
            shutil.copy2(old / f'{split}_15min_predictions.npz', out / f'{split}_predictions.npz')
            shutil.copy2(old / f'{split}_15min_daily_errors.npz', out / f'{split}_daily_errors.npz')
        dump(out / 'selected_parameters.json', read(old / 'selected_parameters.json'))
        (out / 'experiment_record.md').write_text(f"# Reused completed H1 seed{seed}\n\n- Date:{now()};IFpaper_v1.\n- Purpose:known-temperature baseline reference.\n- Source:{old};best.pt/config.json and saved15min artifacts.\n- Operation:copy saved predictions,metrics and parameter metadata only;no model initialization,inference or training.\n- Result:test RMSE{reference['metrics']['test']['rmse_mm']:.9f}mm.\n- Original code and weights unchanged;primary metric uses the common15min grid.\n", encoding='utf-8')
    print('REUSED_FIVE_COMPLETED_H1_RUNS_NO_TRAINING', flush=True)
if __name__ == '__main__':
    main()

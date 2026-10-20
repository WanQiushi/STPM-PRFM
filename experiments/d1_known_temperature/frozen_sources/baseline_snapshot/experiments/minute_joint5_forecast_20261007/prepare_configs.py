"""Write the prespecified forty minute-resolution joint-forecast configurations."""
from if_release.runtime import loads as _release_json_loads
import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'fifteen_displacement_five_seed_complete_20261006'
MODELS = ['DLinear', 'iTransformer', 'CNN', 'LSTM', 'TimeXer', 'PatchTST', 'TimesNet', 'SegRNN']
EXPECTED_PARAMETERS = dict(DLinear=124418880, iTransformer=2925536, CNN=643680, LSTM=611424, TimeXer=19563552, PatchTST=33228000, TimesNet=5694165, SegRNN=56336)
CHANNELS = ['Temperature', 'Pier1', 'Pier2', 'Pier3', 'Pier4']

def main():
    files, names, rows, sources = ([], [], [], {})
    for model in MODELS:
        source = SOURCE / f'config_{model}_15min_Yonly_MSE_TrainExpanded_seed2024.json'
        base = _release_json_loads(source.read_text(encoding='utf-8'))
        for seed in range(2024, 2029):
            cfg = _release_json_loads(json.dumps(base))
            name = f'{model}_1min_Joint5_MSE_seed{seed}'
            cfg.update(sample_minutes=1, min_training_observed_points=1080, run_name=name, seed=seed, output=f'outputs/{HERE.name}/{name}', input_channels=5, output_channels=5, channel_names=CHANNELS, channel_units=['degC', 'mm', 'mm', 'mm', 'mm'], task='1min_temperature_four_support_joint_24h_forecast', input_policy='Past 30 days of temperature and four displacements only; no future values or calendar marks.', normalization_policy='Train-only observed mean/std independently for five channels.', checkpoint_selection='Joint five-channel standardized validation RMSE; physical metrics reported separately.', loss_policy='Observed-entry masked standardized MSE across five channels; equal weight per valid normalized entry; derivative weights zero.', reference_experiment='fifteen_displacement_five_seed_complete_20261006', forecast_target_policy='Future temperature and four displacements are targets only, never model inputs.', protocol='experiments/' + HERE.name + '/dataset/protocol.json')
            args = cfg['model_args']
            args.update(seq_len=43200, pred_len=1440, label_len=1440, enc_in=5, dec_in=5, c_out=5, moving_avg=375, patch_len=240, stride=120)
            if model == 'SegRNN':
                args['seg_len'] = 240
            if model == 'DLinear':
                cfg['gpu_budget_mib'] = 8192
            filename = 'config_' + name + '.json'
            (HERE / filename).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
            files.append(filename)
            names.append(name)
            sources[name] = dict(source_config=str(source), changes=['sampling 15 -> 1 minute; history/future 2880/96 -> 43200/1440', 'historical four Y -> historical T plus four Y; joint five-channel output', 'physical patch/segment and moving-average durations retained', 'validation selection uses joint standardized error', 'new independent 150/26/39 protocol; exact validation/test origins retained'])
        rows.append(dict(model=model, expected_parameters=EXPECTED_PARAMETERS[model], parameter_policy='Analytic estimate; actual training MODEL log is authoritative', seq_len=args['seq_len'], pred_len=args['pred_len'], channels=5, d_model=args['d_model'], d_ff=args['d_ff'], e_layers=args['e_layers'], patch_len_samples=args['patch_len'], stride_samples=args['stride'], seg_len_samples=args.get('seg_len', ''), moving_avg_samples=args['moving_avg'], temporal_rank=args['temporal_rank'] if model == 'TimesNet' else '', effective_batch=cfg['batch_size'], micro_batch=cfg['micro_batch_size'], learning_rate=cfg['lr'], max_epochs=cfg['epochs'], patience=cfg['patience'], admission_budget_mib=cfg['gpu_budget_mib']))
    plan = dict(output=f'outputs/{HERE.name}', gpu=0, max_workers=4, memory_budget_mib=66560, models=MODELS, seeds=list(range(2024, 2029)), configs=files, reuse_runs=[], all_expected_runs=names, scheduling_groups=[['SegRNN', 'CNN', 'DLinear', 'iTransformer'], ['LSTM', 'PatchTST', 'TimeXer'], ['TimesNet']], example_window_index=16, sampling_minutes=1, h1_policy='Excluded; H1 and separate temperature work are not modified or launched.', start_time_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(), purpose='Forty new pure-history joint forecasts: temperature plus four displacement channels, five seeds per retained baseline.', reference_experiment=SOURCE.name)
    (HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE / 'configuration_sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    with (HERE / 'configuration_table.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(configurations=len(files), models=MODELS, gpu=0, max_workers=4)))
if __name__ == '__main__':
    main()

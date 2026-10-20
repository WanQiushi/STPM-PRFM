"""Prepare30 new runs without touching existing baseline results."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import shutil
ROOT = Path(_release_expand('${WORKSPACE}'))
BUNDLE = Path(__file__).resolve().parent
NAME = 'baseline_lstm_mamba_pcfnet_20261015'
SUITE = ROOT / 'experiments' / NAME
OUT = ROOT / 'outputs' / NAME
VENDOR = ROOT / 'vendor/PCFNet_IF2026_source_20261015'
VENDOR.mkdir()
tree = _release_json_loads((BUNDLE / 'pcfnet_source_tree.json').read_text(encoding='utf-8-sig'))
with ZipFile(BUNDLE / 'PCFNet_author_source.zip') as archive:
    for name in archive.namelist():
        relative = Path(*Path(name).parts[1:])
        if not name.endswith('/'):
            dest = VENDOR / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
shutil.copy2(VENDOR / 'layers/ema.py', VENDOR / 'layers/ema_author_original.py')
shutil.copy2(BUNDLE / 'ema_stable.py', VENDOR / 'layers/ema.py')
for name in ['models.py', 'dispatch.py', 'summarize.py', 'physical_grids.py', 'README.md', 'prepare.py', 'ema_stable.py']:
    shutil.copy2(BUNDLE / name, SUITE / name)
(OUT / 'source_snapshot').mkdir()
shutil.copy2(BUNDLE / 'pcfnet_source_tree.json', OUT / 'source_snapshot/pcfnet_source_tree.json')
shutil.copy2(BUNDLE / 'PCFNet_author_source.zip', OUT / 'source_snapshot/PCFNet_author_source.zip')

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
plan = dict(name=NAME, created_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), gpu_indices=[0], max_workers=2, memory_budget_mib=66560, jobs=[], pcfnet_author_commit=tree['sha'], pcfnet_vendor=str(VENDOR), mamba_python=str(ROOT / 'envs/mamba_py311_20261015/bin/python'), mamba_environment_status=str(OUT / 'mamba_environment_status.json'), datasets={}, source=str(SUITE), output=str(OUT), seeds=list(range(2024, 2029)))
for dataset, original, protocol, data in [('D1', ROOT / 'experiments/minute_joint5_forecast_20261007', ROOT / 'experiments/minute_joint5_forecast_20261007/dataset/protocol.json', ROOT / 'data/mainbridge_temperature_v2/minute_complete.npz'), ('D21', ROOT / 'experiments/d21_suite_20261013/experiments/baseline_joint8', ROOT / 'experiments/d21_suite_20261013/data/protocol.json', ROOT / 'experiments/d21_suite_20261013/data/minute_synthetic.npz')]:
    code = SUITE / dataset
    code.mkdir()
    (code / 'configs').mkdir()
    (code / 'dataset').mkdir()
    shutil.copy2(SUITE / 'models.py', code / 'models.py')
    shutil.copy2(SUITE / 'physical_grids.py', code / 'physical_grids.py')
    shutil.copy2(SUITE / 'README.md', code / 'README.md')
    shutil.copy2(protocol, code / 'dataset/protocol.json')
    source = (original / 'data.py').read_text().replace('PROJECT = Path(__file__).resolve().parents[2]', _release_expand("PROJECT = Path('${WORKSPACE}')"))
    if dataset == 'D21':
        source = source.replace("PROTOCOL = PROJECT / 'data/protocol.json'", "PROTOCOL = HERE / 'dataset/protocol.json'")
    (code / 'data.py').write_text(source, encoding='utf-8')
    trainer = (original / 'train.py').read_text()
    trainer = trainer.replace('import random\n', 'import random\nimport shutil\n')
    trainer = trainer.replace("    dump(out / 'config.json', cfg)", "    dump(out / 'config.json', cfg)\n    shutil.copytree(Path(__file__).resolve().parent, out / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))")
    trainer = trainer.replace("effective_batch_size=cfg['batch_size']", "effective_batch_size=cfg['batch_size'], history_points=cfg['model_args']['seq_len'], future_points=cfg['model_args']['pred_len']")
    if dataset == 'D21':
        trainer = trainer.replace("    print('COMPLETED ' + json.dumps(results), flush=True)", "    from physical_grids import score_grids\n    score_grids(out)\n    print('COMPLETED ' + json.dumps(results), flush=True)")
    (code / 'train.py').write_text(trainer, encoding='utf-8')
    p = _release_json_loads(protocol.read_text())
    plan['datasets'][dataset] = dict(data=str(data), source_protocol=str(protocol), frozen_protocol=str(code / 'dataset/protocol.json'), windows={k: len(v) for k, v in p['origins'].items()}, normalization='Frozen train-only observed mean/std', history_points=43200, prediction_points=1440, channels=5 if dataset == 'D1' else 8)
    original_cfg = next(original.glob('config_LSTM*seed2024.json'))
    base = _release_json_loads(original_cfg.read_text())
    for seed in range(2024, 2029):
        for model in ['VanillaLSTM', 'Mamba', 'PCFNet']:
            cfg = _release_json_loads(json.dumps(base))
            cfg.update(model=model, seed=seed, run_name=f'{dataset}_{model}_1min_seed{seed}', output=str(OUT / dataset / f'{model}_seed{seed}'), data=str(data), protocol=str(code / 'dataset/protocol.json'), trainer=str(code / 'train.py'), experiment_version=NAME, official_source_commit=tree['sha'] if model == 'PCFNet' else None, pcfnet_vendor=str(VENDOR), micro_batch_size=4 if model == 'Mamba' else 8, gpu_budget_mib=24576 if model == 'Mamba' else 16384, input_policy='Raw43200 one-minute historical values only; no future values/predicted T/calendar; no patches or downsampling')
            args = cfg['model_args']
            for key in ['patch_len', 'stride', 'temporal_rank']:
                args.pop(key)
            args.update(d_model=64, d_ff=128, e_layers=2, dropout=0.1)
            if model == 'Mamba':
                args.update(d_state=16, d_conv=4, expand=2)
                cfg['model_definition'] = 'Official original Mamba1 blocks, pre-norm residual stack, input Linear and last-state direct horizon Linear; no graph/frequency/ODE additions'
                cfg['initialization_policy'] = 'Preserve official Mamba initialization; zero only forecasting output Linear'
                python = plan['mamba_python']
            elif model == 'VanillaLSTM':
                cfg['model_definition'] = 'Standard torch.nn.LSTM,2 layers64, raw minute sequence, last hidden state direct horizon Linear; no patch convolution'
                cfg['initialization_policy'] = 'PyTorch LSTM defaults; zero forecasting output Linear, historical last-value anchor as frozen baseline'
                python = _release_expand('${PYTHON}')
            else:
                args.update(data='bridge_no_calendar', encoder_layers=2, fusion_layers=1, top_k=3, num_kernels=3, n_heads=4, attn_dropout=0.1, hidden_dim=128, coef_scale=0.1, tfactor=0.5, revin=1, ma_type='ema', ema_a=0.3, ema_b=0.3)
                cfg['model_definition'] = 'Author PCFNet; unchanged backbone, no calendar marks; explicit trend hidden128 avoids default2*seq_len width'
                cfg['initialization_policy'] = 'Author PCFNet defaults; no global reinitialization or output replacement'
                cfg['numerical_change'] = 'Same EMA recurrence via double-precision parallel prefix; no division/cumsum, original global powers underflow at43200 points'
                python = _release_expand('${PYTHON}')
            config = code / 'configs' / f'{model}_seed{seed}.json'
            dump(config, cfg)
            plan['jobs'].append(dict(key=cfg['run_name'], dataset=dataset, model=model, seed=seed, config=str(config), cwd=str(code), output=cfg['output'], gpu=0, python=python, gpu_budget_mib=cfg['gpu_budget_mib'], status='pending', command=[python, '-u', str(code / 'train.py'), '--config', str(config)]))
order = {'PCFNet': 0, 'VanillaLSTM': 1, 'Mamba': 2}
plan['jobs'].sort(key=lambda j: (j['seed'], j['dataset'], order[j['model']]))
dump(OUT / 'plan.json', plan)
dump(SUITE / 'experiment_definition.json', dict(plan=str(OUT / 'plan.json'), datasets=plan['datasets'], models=['VanillaLSTM', 'Mamba', 'PCFNet'], seeds=plan['seeds'], jobs=30, gpu_indices=[0], training='Frozen Adam1e-4/effectivebatch16/max300/patience60/clip1; joint standardized observed-mask MSE; derivative weights0', model_sources=dict(PCFNet=dict(author_commit=tree['sha'], vendor=str(VENDOR)), Mamba='mamba-ssm2.2.6.post3 original Mamba1', LSTM='torch.nn.LSTM'), changes='No existing baseline or A2 retraining; historical SegRNN/PatchLSTM results preserved; PCFNet numerical EMA equivalence documented'))
print(json.dumps(dict(prepared=True, jobs=30, gpu_indices=[0], source=str(SUITE), output=str(OUT), datasets=plan['datasets'])))

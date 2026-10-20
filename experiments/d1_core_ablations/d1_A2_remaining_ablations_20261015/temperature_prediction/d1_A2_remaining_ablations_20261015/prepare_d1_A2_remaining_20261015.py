"""Prepare missing D1 fixed-A2 ablations; reuse completed stages, GPU0 only."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import shutil
import torch
import numpy as np
ROOT = Path(_release_expand('${WORKSPACE}'))
NAME = 'd1_A2_remaining_ablations_20261015'
HERE = ROOT / 'temperature_prediction' / NAME
OUT = ROOT / 'temperature_prediction/outputs' / NAME
OLD = ROOT / 'temperature_prediction/outputs/va_core_ablation_all_20261008'
FROZEN_H1 = ROOT / 'outputs/va_1_20261004'
NO_TREND = ROOT / 'temperature_prediction/outputs/d1_A2_no_trend_20261015'

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
HERE.mkdir()
OUT.mkdir()
(HERE / 'configs').mkdir()
(OUT / 'experiment_records').mkdir()
SOURCE = OLD / 'A2_seed2024/frozen_source'
modules = ['train.py', 'models.py', 'temperature_ablation.py', 'fixed_trend.py', 'data_contract_v8.py', 'train_contract_v9.py', 'ode_core_v9.py', 'temperature_metrics.py', 'h1_ablation.py']
for name in modules:
    shutil.copy2(SOURCE / name, HERE / name)
shutil.copytree(SOURCE / 'vendor_latent_ode', HERE / 'vendor_latent_ode', ignore=shutil.ignore_patterns('__pycache__'))
text = (HERE / 'temperature_ablation.py').read_text()
text = text.replace("        self.ablation = cfg['ablation']", "        self.ablation = cfg['ablation']\n        self.remove_ode = cfg.get('remove_temperature_ode', False)\n        self.remove_residual = cfg.get('remove_temperature_residual', False)")
text = text.replace("if self.ablation == 'A1' else SmallLatentODE(cfg)", "if self.ablation == 'A1' or self.remove_ode else SmallLatentODE(cfg)")
text = text.replace('        nn.init.zeros_(self.residual[-1].bias)', '        nn.init.zeros_(self.residual[-1].bias)\n        if self.remove_residual:\n            del self.residual')
text = text.replace("        if self.ablation == 'A1':", "        if self.ablation == 'A1' or self.remove_ode:")
text = text.replace('        residual = self.residual(features)', '        residual = torch.zeros_like(base) if self.remove_residual else self.residual(features)')
(HERE / 'temperature_ablation.py').write_text(text)
text = (SOURCE / 'evaluate_core.py').read_text()
text = text.replace("    if group in ['A5', 'A6']:", "    if group in ['S4', 'S5']:")
text = text.replace("OUT / (group + '_h1/adapted_best.pt')", "ROOT / 'temperature_prediction/outputs/va_core_ablation_all_20261008' / ({'S4':'A5','S5':'A6'}[group] + '_h1/adapted_best.pt')")
text = text.replace("    elif group == 'A7':", "    elif group == 'S6':")
text = text.replace("        source = OUT / f'{group}_seed{seed}' if group in ['A1', 'A2', 'A4'] else REFERENCE / f'1min_seed{seed}'", "        source = OUT / f'{group}_seed{seed}' if group in ['S1', 'S2'] else ROOT / 'temperature_prediction/outputs/va_core_ablation_all_20261008' / f'A2_seed{seed}'")
text = text.replace("        with np.load(REFERENCE / f'evaluation_1min/1min_seed{seed}__adapted_h1.npz') as raw:\n            full_prediction = raw['prediction_mm'].copy()", "        reference_path = ROOT / 'temperature_prediction/outputs/va_core_ablation_all_20261008' / f'evaluation_A2/seed{seed}/metrics.json'\n        reference_metrics = json.loads(reference_path.read_text())")
text = text.replace("        full_y = score(full_prediction, ytruth, data['mask'])", "        full_y = dict(rmse=reference_metrics['displacement_rmse_mm'], mae=reference_metrics['displacement_mae_mm'])")
text = text.replace("        if group in ['A1', 'A2', 'A4']:", "        if group in ['S1', 'S2']:")
text = text.replace('            params_t = 78523', '            params_t = 78345')
text = text.replace("'A0_displacement_rmse_mm'", "'reference_A2_displacement_rmse_mm'")
text = text.replace("'A0_displacement_mae_mm'", "'reference_A2_displacement_mae_mm'")
text = text.replace("'device': 'cpu', 'h1_checkpoint_count': 1", "'device': 'cpu', 'architecture_reference': 'current fixed A2', 'reference_metrics': str(reference_path), 'h1_checkpoint_count': 1")
(HERE / 'evaluate_core.py').write_text(text)
normal = _release_json_loads((FROZEN_H1 / 'normalization.json').read_text())
original_h1 = torch.load(FROZEN_H1 / 'h1_original.pt', map_location='cpu', weights_only=False)
adaptation = _release_json_loads((FROZEN_H1 / 'settings.json').read_text())
reuse = {'H1': {}, 'no_residual': [], 'no_ODE_initializations': []}
for arm in ['A5', 'A6']:
    folder = OLD / (arm + '_h1')
    cfg = _release_json_loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'adapted_best.pt', map_location='cpu', weights_only=False)
    reuse['H1'][arm] = dict(checkpoint=str(folder / 'adapted_best.pt'), original_pretraining_configuration_differences={key: {'full': value, 'ablated': cfg.get(key)} for key, value in original_h1['config'].items() if cfg.get(key) != value}, adaptation_settings_equal=cfg['adaptation'] == adaptation, mean_equal=np.array_equal(np.asarray(checkpoint['mean']), np.asarray(normal['h1_mean'])), std_equal=np.array_equal(np.asarray(checkpoint['std']), np.asarray(normal['h1_std'])), training_source_equal=(folder / 'frozen_source/train_h1.py').read_text() == (SOURCE / 'train_h1.py').read_text(), protocol_equal=(folder / 'frozen_source/h1_pretraining_protocol.json').read_text() == (SOURCE / 'h1_pretraining_protocol.json').read_text(), report=_release_json_loads((folder / 'data_report.json').read_text()), justification='H1 training depends on measured T and original fixed-v11 causal OOF cache, not the final A0 or A2 temperature head; reuse weights, compute new A2-input evaluation.')
runs = []
nodes = []
names = {'S0': 'Full fixed A2', 'S1': 'No temperature ODE', 'S2': 'No temperature residual', 'S3': 'No trend supervision', 'S4': 'No thermal lag', 'S5': 'No BW physical rate', 'S6': 'No forecast-temperature adaptation'}
for seed in range(2024, 2029):
    reference = OLD / f'A2_seed{seed}'
    cfg = _release_json_loads((reference / 'config.json').read_text())
    no_ode = dict(cfg, run_name=f'S1_seed{seed}', remove_temperature_ode=True, initial_checkpoint_kind='same-seed completed SharedTimeReadout no-ODE warm initialization; original matched protocol', initial_checkpoint=str(OLD / f'warm_A1_seed{seed}/best.pt'))
    config_file = HERE / 'configs' / f'S1_seed{seed}.json'
    dump(config_file, no_ode)
    old_warm = torch.load(no_ode['initial_checkpoint'], map_location='cpu', weights_only=False)
    reuse['no_ODE_initializations'].append(dict(seed=seed, checkpoint=no_ode['initial_checkpoint'], warm_configuration=old_warm['config'], final_configuration_changes={key: {'reference': cfg.get(key), 'control': value} for key, value in no_ode.items() if cfg.get(key) != value}, training_source_equal=(HERE / 'train.py').read_text() == (reference / 'frozen_source/train.py').read_text()))
    train_key = f'S1_seed{seed}_temperature'
    nodes.append(dict(key=train_key, phase='temperature', kind='gpu', gpu=0, arm='S1', seed=seed, weight=3.0, cwd=str(HERE), command=[_release_expand('${PYTHON}'), '-u', str(HERE / 'train.py'), '--config', str(config_file), '--device', 'cuda:0'], dependencies=[], status='pending'))
    dest = OUT / f'S2_seed{seed}'
    dest.mkdir()
    no_residual = dict(cfg, run_name=f'S2_seed{seed}', remove_temperature_residual=True)
    base = torch.load(reference / 'base_best.pt', map_location='cpu', weights_only=False)
    zero = {key: float(base['model'][key].abs().max()) for key in ['residual.2.weight', 'residual.2.bias']}
    state = {key: value for key, value in base['model'].items() if not key.startswith('residual.')}
    checkpoint = dict(base, model=state, config=no_residual, source_checkpoint=str(reference / 'base_best.pt'))
    torch.save(checkpoint, dest / 'best.pt')
    dump(dest / 'config.json', no_residual)
    shutil.copy2(reference / 'base_best_predictions.npz', dest / 'validation_predictions.npz')
    shutil.copy2(reference / 'base_best_validation.json', dest / 'best_validation.json')
    for filename in ['data_report.json', 'window_origins.json']:
        shutil.copy2(reference / filename, dest / filename)
    summary = dict(status='reused_completed_base_stage', seed=seed, arm='S2', sample_minutes=1, parameters=dict(base=7577, residual=0, total=7577), source_checkpoint=str(reference / 'base_best.pt'), base_epoch=base['epoch'], residual_output_layer_absolute_max=zero, training_repeated=False, test_evaluated=False, source_predictions=str(reference / 'base_best_predictions.npz'))
    dump(dest / 'summary.json', summary)
    reuse['no_residual'].append(summary)
    (dest / 'experiment_record.md').write_text('# Fixed A2 no-temperature-residual: existing base reused\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '\n- Same seed base selected under original A2 loss/protocol; its residual was frozen and exactly zero.\n- Removed residual parameter keys; retained all fitted base parameters, normalization, masks and cached Celsius predictions.\n- Source: ' + str(reference) + '\n- No retraining, no test scoring.\n', encoding='utf-8')
    for arm in ['S1', 'S2', 'S4', 'S5', 'S6']:
        metrics = OUT / f'evaluation_{arm}/seed{seed}/metrics.json'
        runs.append(dict(arm=arm, name=names[arm], seed=seed, metrics=str(metrics), temperature_source=str(OUT / f'{arm}_seed{seed}') if arm in ['S1', 'S2'] else str(reference), action='new temperature training and fixed-H1 validation' if arm == 'S1' else 'reuse completed weights, new fixed-A2-input validation'))
        nodes.append(dict(key=f'{arm}_seed{seed}_validation', phase='validation', kind='cpu', gpu=None, arm=arm, seed=seed, weight=3.0, cwd=str(HERE), command=[_release_expand('${PYTHON}'), '-u', str(HERE / 'evaluate_core.py'), '--group', arm, '--seeds', str(seed)], dependencies=[train_key] if arm == 'S1' else [], status='pending'))
references = []
for seed in range(2024, 2029):
    references.extend([dict(arm='S0', seed=seed, metrics=str(OLD / f'evaluation_A2/seed{seed}/metrics.json')), dict(arm='S3', seed=seed, metrics=str(NO_TREND / f'evaluation_A2/seed{seed}/metrics.json'))])
plan = dict(created_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), scope='D1 current fixed-A2 missing five core ablations; no historical A0 metrics reused', gpu_indices=[0], max_workers_per_gpu=2, max_cpu_validations=2, runs=runs, references=references, nodes=nodes, names=names, contract='D1 original304train/66temperatureval,26commonvalidation,1minute1440nodes,4bearings;original A2 recipe,5temperature seeds;shared corresponding H1 per intervention;no test scoring', reuse_report=str(OUT / 'reuse_report.json'), new_training_count=5, new_validation_count=25, reused_reference_count=10, gpu_admission='GPU0 only; existing owner-aware gate waits for external processes; never use GPU1')
dump(OUT / 'stage_plan.json', plan)
dump(OUT / 'reuse_report.json', reuse)
dump(HERE / 'experiment_definition.json', {key: value for key, value in plan.items() if key != 'nodes'})
controller = (ROOT / 'experiments/d21_suite_20261013/dispatch_trend_controls_five_seeds_20261015.py').read_text()
controller = controller[:controller.index('rows=[]')]
controller = controller.replace("ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/trend_weight_control_five_seeds_20261015'", _release_expand(f"ROOT=Path('${{WORKSPACE}}');OUT=ROOT/'temperature_prediction/outputs/{NAME}'"))
controller = controller.replace("ROOT/'experiments/baseline_joint8/run_queue.py'", "ROOT/'experiments/d21_suite_20261013/experiments/baseline_joint8/run_queue.py'")
controller = controller.replace('D2.1 trend-weight validation control', 'D1 fixed-A2 remaining core ablations')
controller = controller.replace('same paired v9 and immutable H1', 'stage_plan.json and reuse_report.json; fixed A2 and corresponding reused H1')
controller = controller.replace('Two checked-empty GPUs', 'User-selected GPU0 only; wait for external users')
controller += '\nrows=[]\nfor run in plan[\'references\']+plan[\'runs\']:\n m=json.loads(Path(run[\'metrics\']).read_text())\n rows.append(dict(arm=run[\'arm\'],seed=run[\'seed\'],temperature_rmse_c=m[\'temperature_rmse_celsius\'],response_rmse_mm=m[\'displacement_rmse_mm\'],response_mae_mm=m[\'displacement_mae_mm\'],response_mse_mm2=m[\'displacement_mse_mm2\'],temperature_full66_rmse_c=m[\'full66_temperature_rmse_celsius\'],parameters=m[\'parameters\'],observed_t_nodes=m[\'observed_t_nodes\'],observed_y_nodes=m[\'observed_y_nodes\'],source=run[\'metrics\']))\nfields=[\'temperature_rmse_c\',\'response_rmse_mm\',\'response_mae_mm\',\'response_mse_mm2\',\'temperature_full66_rmse_c\']\nstats={arm:{key:dict(mean=float(np.mean([row[key] for row in rows if row[\'arm\']==arm])),sample_std=float(np.std([row[key] for row in rows if row[\'arm\']==arm],ddof=1))) for key in fields} for arm in plan[\'names\']}\npaired=[]\nfor arm in [\'S1\',\'S2\',\'S3\',\'S4\',\'S5\',\'S6\']:\n for seed in range(2024,2029):\n  ref=next(row for row in rows if row[\'arm\']==\'S0\' and row[\'seed\']==seed);v=next(row for row in rows if row[\'arm\']==arm and row[\'seed\']==seed)\n  paired.append(dict(arm=arm,seed=seed,temperature_rmse_delta_c=v[\'temperature_rmse_c\']-ref[\'temperature_rmse_c\'],response_rmse_delta_mm=v[\'response_rmse_mm\']-ref[\'response_rmse_mm\']))\ndump(OUT/\'comparison.json\',dict(rows=rows,statistics=stats,paired_deltas=paired,contract=plan[\'contract\'],reuse_report=plan[\'reuse_report\'],test_evaluated=False,aggregation=\'Five temperature seeds2024-2028, mean/sample SDddof1; one shared response checkpoint per intervention, no ensemble\'))\nlines=[\'# D1 current fixed A2: all six core ablations\',\'\',\'One-minute,26 common validation windows; independent temperature seeds2024-2028; mean/sample SD. Temperature full66 validation reported separately; no test evaluation.\',\'\',\'|Arm|Temperature common26 RMSE/C|Displacement common26 RMSE/mm|Displacement MAE/mm|Delta displacement RMSE vs full/mm|\',\'|---|---:|---:|---:|---:|\']\nfor arm,name in plan[\'names\'].items():\n s=stats[arm];t=s[\'temperature_rmse_c\'];y=s[\'response_rmse_mm\'];mae=s[\'response_mae_mm\'];delta=y[\'mean\']-stats[\'S0\'][\'response_rmse_mm\'][\'mean\'];lines.append(f"|{name}|{t[\'mean\']:.6f} +/- {t[\'sample_std\']:.6f}|{y[\'mean\']:.6f} +/- {y[\'sample_std\']:.6f}|{mae[\'mean\']:.6f} +/- {mae[\'sample_std\']:.6f}|{delta:+.6f}|")\nlines+=[\'\',\'No historical A0 metrics substituted. S0/S3 reuse completed fixed-A2 evaluations; remaining25 evaluations newly computed with current fixed-A2 inputs. Only no-ODE temperature base/residual stages trained; matching warm initializers and completed response/base checkpoints reused.\']\n(OUT/\'results_table.md\').write_text(\'\\n\'.join(lines)+\'\\n\',encoding=\'utf-8\')\nstate.update(status=\'completed\',finish_beijing=now(),comparison=str(OUT/\'comparison.json\'));save();print(\'ALL_D1_FIXED_A2_CORE_ABLATIONS_COMPLETED\',flush=True)\n'
(HERE / 'dispatch.py').write_text(controller)
readme = _release_expand('# D1 current fixed A2: complete missing core ablations\n\nReference is frozen A2 registered2026-10-09. Historical A0 result metrics are never substituted.\n\nData: ${WORKSPACE}/data/mainbridge_temperature_v2/minute_complete.npz; original511/110/109day split,304temperature training and66temperature validation windows. Final temperature/displacement comparison uses original26 common validation windows and measured masks,1minute1440nodes,4bearings.\n\nS0 full A2 and S3 no-trend: reuse completed five-seed metrics and checkpoints; no retraining or rescoring.\nS1 no temperature ODE: replace only latent ODE trajectory with existing SharedTimeReadout; retain A2 historical anchor/amplitude, no learned L/A head, trend3,shape0.25,L/Aaux0,Adam0.001,batch32,80base/60residual,patience12. Reuse already completed matched no-ODE warm initializers; train only five final temperature base/residual stages.\nS2 no temperature residual: reuse original selected base stage, whose residual output was frozen and exactly zero. Store checkpoint with residual keys removed,7577temperature parameters, and exact cached base predictions. No repeat training or temperature inference.\nS4 no thermal lag and S5 no BW physics: reuse completed structurally ablated H1 checkpoints after confirming original pretraining protocol,normalization and fixed-v11 OOF adaptation match. Those trainers do not consume final A0/A2 temperature heads. Compute new evaluations with CURRENT A2 minute temperature predictions. No reuse of old A0 metrics. No-BW retains neural state memory.\nS6 no predicted-temperature adaptation: reuse original unadapted h1_original.pt with CURRENT A2 temperature predictions. No retraining.\n\nAll interventions have five temperature seeds; response model has one shared checkpoint per corresponding intervention, matching frozen A2. No five independently trained H1 seeds are claimed. H1 validation runs on CPU4threads,up to2processes. No OOF/H1 training, no test predictions or scoring.\n\nEntry: python -u dispatch.py. GPU0 only,up to2training jobs; existing owner-aware gate waits for external GPU0 processes. GPU1 is never selected even if idle. Existing resource gate is retained.\nConfig: configs/S1_seed*.json; model ablation=A2 intentionally preserved,remove_temperature_ode=True changes only shape generator. Modified source is temperature_ablation.py; training/loss/data modules unchanged. Actual configs/source are copied by original train.py into each run.\nEvaluation: python evaluate_core.py --group S4 --seeds 2024 (other arms/seeds analogous). Inference/scoring follows frozen evaluate_core.py; only declared checkpoint/temperature source and full-A2 reference fields changed.\nOutputs: ${WORKSPACE}/temperature_prediction/outputs/d1_A2_remaining_ablations_20261015. stage_plan.json/reuse_report.json declare all actions and provenance; progress.json/controller.log track queue; experiment_records saves per-stage success/failure. Per-arm evaluation metrics/predictions are under evaluation_S*/seed*. Final comparison.json/results_table.md combine all35 fixed-A2 evaluations,including ten already completed.\nNo hashes,extra smoke tests,automatic retries or fallback training were added. This preparation does not launch training; dispatch.py is launched separately after resource inspection.\n')
(HERE / 'README.md').write_text(readme)
(OUT / 'README.md').write_text(readme)
shutil.copy2(Path(__file__), HERE / 'prepare_d1_A2_remaining_20261015.py')
print(json.dumps(dict(source=str(HERE), output=str(OUT), new_training_count=5, new_validation_count=25, reused_evaluations=10, resource_policy=plan['gpu_admission'], reuse_report=str(OUT / 'reuse_report.json'))))

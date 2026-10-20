"""All approved remaining core ablations, two GPU workers plus A7 CPU evaluation."""
from if_release.runtime import loads as _release_json_loads
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from datetime import datetime
from zoneinfo import ZoneInfo
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
LOGS = ROOT / 'temperature_prediction/logs' / HERE.name
LOCK = Lock()
STOP = Event()
ACTIVE = {}
FINISHED = []
FAILED = []
GPUS = [0, 1]

def progress():
    (OUT / 'progress.json').write_text(json.dumps({'model': 'va', 'status': 'failed' if FAILED else 'running', 'physical_gpus': GPUS, 'active_jobs': ACTIVE, 'completed_jobs': FINISHED, 'failed_jobs': FAILED, 'remaining_core_groups': ['A1', 'A2', 'A4', 'A5', 'A6', 'A7'], 'temperature_seeds_per_group': 5, 'temperature_models': 15, 'h1_structural_retrains': 2, 'references_reused': ['A0', 'A3']}, indent=2))
exec((HERE / 'queue_job.py').read_text(encoding='utf-8'), globals())

def worker(gpu, jobs):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    while True:
        with LOCK:
            if STOP.is_set() or not jobs:
                return
            group, seed = jobs.pop(0)
        commands = []
        if group in ['A5', 'A6']:
            commands.append((group + '_h1', [sys.executable, '-u', str(HERE / 'train_h1.py'), '--config', str(HERE / (group + '_h1.json'))], 'h1_pretraining_adaptation'))
            commands.append(('evaluate_' + group, [sys.executable, '-u', str(HERE / 'evaluate_core.py'), '--group', group, '--seeds', '2024,2025,2026,2027,2028'], 'cpu_h1_evaluation'))
        else:
            name = f'{group}_seed{seed}'
            if group == 'A1':
                commands.append(('warm_' + name, [sys.executable, '-u', str(HERE / 'train_warm.py'), '--config', str(HERE / ('warm_' + name + '.json'))], 'no_ODE_initialization'))
            commands.append((name, [sys.executable, '-u', str(HERE / 'train.py'), '--config', str(HERE / (name + '.json'))], 'temperature_base_residual'))
            commands.append(('evaluate_' + name, [sys.executable, '-u', str(HERE / 'evaluate_core.py'), '--group', group, '--seeds', str(seed)], 'cpu_h1_evaluation'))
        for name, command, stage in commands:
            code = run_job(name, command, env, stage, gpu)
            if code != 0 or STOP.is_set():
                return

def cpu_control():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    run_job('evaluate_A7', [sys.executable, '-u', str(HERE / 'evaluate_core.py'), '--group', 'A7', '--seeds', '2024,2025,2026,2027,2028'], env, 'unadapted_h1_control', None)

def main():
    OUT.mkdir(parents=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE, OUT / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(HERE / 'README.md', OUT / 'README.md')
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    jobs = [('A1', 2024), ('A5', None), ('A1', 2025), ('A6', None), *[('A1', seed) for seed in range(2026, 2029)], *[('A2', seed) for seed in range(2024, 2029)], *[('A4', seed) for seed in range(2024, 2029)]]
    (OUT / 'launch.json').write_text(json.dumps({'model': 'va', 'started_beijing': started, 'controller_pid': os.getpid(), 'physical_gpus': GPUS, 'queued_bundles': jobs, 'new_temperature_models': 15, 'new_no_ODE_initializations': 5, 'h1_retrained_architectures': 2, 'A7': 'CPU evaluation of original unadapted H1', 'A0_A3': 'reused completed results', 'test_evaluated': False}, indent=2))
    progress()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, gpu, jobs) for gpu in GPUS]
        futures.append(pool.submit(cpu_control))
        for future in futures:
            future.result()
    if FAILED:
        code = FAILED[0]['exit']
        (LOGS / 'suite.exit').write_text(str(code))
        raise SystemExit(code)
    rows = [_release_json_loads((OUT / f'evaluation_{group}/seed{seed}/metrics.json').read_text()) for group in ['A1', 'A2', 'A4', 'A5', 'A6', 'A7'] for seed in range(2024, 2029)]
    stats = []
    for group in ['A1', 'A2', 'A4', 'A5', 'A6', 'A7']:
        items = [r for r in rows if r['group'] == group]
        item = {'group': group, 'seeds': 5}
        for key in ['displacement_rmse_mm', 'displacement_mae_mm', 'displacement_mse_mm2', 'temperature_rmse_celsius', 'temperature_mae_celsius', 'full66_temperature_rmse_celsius', 'full66_temperature_mae_celsius', 'tail2h_displacement_rmse_mm', 'delta_rmse_mm', 'delta_mae_mm']:
            values = [r[key] for r in items]
            item[key] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
        stats.append(item)
    report = {'model': 'va', 'status': 'completed', 'started_beijing': started, 'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'groups': stats, 'per_seed': rows, 'physical_gpus': GPUS, 'A0_reference': str(ROOT / 'temperature_prediction/outputs/va_2_2_sampling_20261008/evaluation_1min/comparison.json'), 'A3_reference': str(ROOT / 'temperature_prediction/outputs/va_core_ablation_20261008_A3_no_temperature_residual/comparison.json'), 'test_evaluated': False, 'five_seed_scope': 'temperature randomness; one H1 checkpoint per structural group'}
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT / 'experiment_record.md').write_text('# All va core ablations completed\n\n- Date: ' + report['finished_beijing'] + '; IFpaper_v1.\n- Operation: A1/A2/A4 five temperature seeds each, A5/A6 original H1 pretraining+adaptation retrained, A7 original-H1 evaluation; A0/A3 reused, main va frozen.\n- Dataset/labels: existing native1min26-window measured masks, original normalization; no test.\n- Results: comparison.json, per-seed evaluation metrics and prediction NPZ, training histories/config/frozen_source, separate stage records and logs.\n', encoding='utf-8')
    (OUT / 'progress.json').write_text(json.dumps({'status': 'completed', 'physical_gpus': GPUS, 'finished_beijing': report['finished_beijing'], 'groups': ['A1', 'A2', 'A4', 'A5', 'A6', 'A7']}))
    (LOGS / 'suite.exit').write_text('0')
    print('SUITE COMPLETED', flush=True)
if __name__ == '__main__':
    main()

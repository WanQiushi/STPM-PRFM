"""GPU0+1 independent-seed queue for10/5/1-minute va-2.2."""
from if_release.runtime import loads as _release_json_loads
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from zoneinfo import ZoneInfo
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
LOGS = ROOT / 'temperature_prediction/logs' / HERE.name
LOCK = Lock()
STOP = Event()
ACTIVE = {}
FINISHED = []
FAILED = []
GPUS = []

def progress():
    (OUT / 'progress.json').write_text(json.dumps({'version': 'va-2.2', 'status': 'failed' if FAILED else 'running', 'physical_gpus': GPUS, 'active_jobs': ACTIVE, 'completed_jobs': FINISHED, 'failed_jobs': FAILED, 'sample_minutes': [10, 5, 1], 'temperature_seeds_per_resolution': 5}, indent=2))

def run_job(name, command, env, stage, gpu):
    with (LOGS / (name + '.log')).open('w') as log:
        with LOCK:
            if STOP.is_set():
                return None
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (LOGS / (name + '.pid')).write_text(str(process.pid))
            ACTIVE[str(gpu)] = {'run': name, 'stage': stage, 'child_pid': process.pid, 'physical_gpu': gpu, 'log': str(LOGS / (name + '.log'))}
            progress()
        print('START ' + name + ' GPU ' + str(gpu) + ' PID ' + str(process.pid), flush=True)
        code = process.wait()
    (LOGS / (name + '.exit')).write_text(str(code))
    with LOCK:
        del ACTIVE[str(gpu)]
        if code != 0:
            FAILED.append({'run': name, 'stage': stage, 'exit': code, 'physical_gpu': gpu})
            STOP.set()
            (OUT / (name + '_failure_record.md')).write_text('# va-2.2 sampling experiment failed\n\n- Date: ' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '; IFpaper_v1.\n- Purpose:10/5/1min, five seeds each, dual GPU independent-seed queue.\n- Stage/job: ' + stage + '/' + name + '; GPU ' + str(gpu) + '; PID ' + str(process.pid) + '; exit ' + str(code) + '.\n- Exact command: ' + json.dumps(command) + '.\n- Log: ' + str(LOGS / (name + '.log')) + '.\n- Products: ' + str(OUT) + '; stop assigning new stages, retain finished outputs and let already-running child finish; no retry.\n', encoding='utf-8')
        else:
            FINISHED.append(name)
        progress()
    print(('FAILED ' if code else 'DONE ') + name + ' GPU ' + str(gpu), flush=True)
    return code

def queue_worker(gpu, jobs, evaluation=False):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    while True:
        with LOCK:
            if STOP.is_set() or not jobs:
                return
            item = jobs.pop(0)
        if evaluation:
            dt = item
            run_job(f'evaluate_{dt}min', [sys.executable, '-u', str(HERE / 'evaluate_sampling.py'), '--sample-minutes', str(dt), '--device', 'cuda:0'], env, 'fixed_h1_evaluation', gpu)
        else:
            dt, seed = item
            name = f'{dt}min_seed{seed}'
            code = run_job('v9_' + name, [sys.executable, '-u', str(HERE / 'train_warm.py'), '--config', str(HERE / ('v9_' + name + '.json')), '--device', 'cuda:0'], env, 'v9_initialization', gpu)
            if code != 0 or STOP.is_set():
                return
            run_job(name, [sys.executable, '-u', str(HERE / 'train.py'), '--config', str(HERE / (name + '.json')), '--device', 'cuda:0'], env, 'temperature_base_residual', gpu)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    GPUS.extend((int(value) for value in args.gpus.split(',')))
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / 'README.md', OUT / 'README.md')
    (LOGS / 'suite.pid').write_text(str(os.getpid()))
    launch = {'version': 'va-2.2', 'started_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'controller_pid': os.getpid(), 'physical_gpus': GPUS, 'sample_minutes': [10, 5, 1], 'seeds': list(range(2024, 2029)), 'new_temperature_training_runs': 15, 'retrained_v9_initializations': 15, 'h1_checkpoint_count': 1, 'h1_retrained': False, 'h1_checkpoint': str(ROOT / 'outputs/va_1_20261004/best_va1.pt'), 'common_grid_minutes': 30, 'test_evaluated': False, 'stage_caps': {'v9_initialization': 80, 'base': 80, 'residual': 60}, 'patience': 12}
    (OUT / 'launch.json').write_text(json.dumps(launch, indent=2))
    cpu_env = os.environ.copy()
    cpu_env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for name, command, stage in [('build_sampling_data', [sys.executable, '-u', str(HERE / 'build_sampling_data.py')], 'dataset_construction'), ('evaluate_15min', [sys.executable, '-u', str(HERE / 'evaluate_sampling.py'), '--sample-minutes', '15', '--device', 'cpu'], 'reference_rescoring')]:
        code = run_job(name, command, cpu_env, stage, None)
        if code != 0:
            (LOGS / 'suite.exit').write_text(str(code))
            raise SystemExit(code)
    jobs = [(dt, seed) for dt in [10, 5, 1] for seed in range(2024, 2029)]
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(queue_worker, gpu, jobs) for gpu in GPUS]
        for future in futures:
            future.result()
    if not FAILED:
        eval_jobs = [10, 5, 1]
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            futures = [pool.submit(queue_worker, gpu, eval_jobs, True) for gpu in GPUS]
            for future in futures:
                future.result()
    if FAILED:
        code = FAILED[0]['exit']
        (LOGS / 'suite.exit').write_text(str(code))
        raise SystemExit(code)
    reports = [_release_json_loads((OUT / f'evaluation_{dt}min/comparison.json').read_text()) for dt in [15, 10, 5, 1]]
    report = {'version': 'va-2.2', 'status': 'completed', 'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'sampling_comparisons': reports, 'physical_gpus': GPUS, 'temperature_seeds_per_resolution': 5, 'new_temperature_models': 15, 'h1_checkpoint_count': 1, 'h1_parameters_updated': False, 'common_grid_minutes': 30, 'test_evaluated': False}
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.concat([pd.read_csv(OUT / f'evaluation_{dt}min/per_seed.csv') for dt in [15, 10, 5, 1]], ignore_index=True).to_csv(OUT / 'per_seed_sampling.csv', index=False)
    pd.concat([pd.read_csv(OUT / f'evaluation_{dt}min/five_seed_summary.csv') for dt in [15, 10, 5, 1]], ignore_index=True).to_csv(OUT / 'sampling_summary.csv', index=False)
    (OUT / 'experiment_record.md').write_text('# va-2.2 sampling comparison completed\n\n- Date: ' + report['finished_beijing'] + '; IFpaper_v1.\n- Purpose:10/5/1min, five independent temperature seeds per grid;15min results reused.\n- Operation:GPU0+1 independent-seed queue; same λ3 and original hyperparameters; fixed adapted H1 grid migration.\n- Results: native grid and same30min grid means/sampleSD in sampling_summary.csv; no test evaluation.\n- Products: comparison.json, per_seed_sampling.csv, datasets,15warm+15temperature directories, four evaluations, logs and frozen_source/config.\n- No plots requested; no ensemble.\n', encoding='utf-8')
    (LOGS / 'suite.exit').write_text('0')
    (OUT / 'progress.json').write_text(json.dumps({'status': 'completed', 'physical_gpus': GPUS, 'sample_minutes': [10, 5, 1], 'temperature_seeds_per_resolution': 5, 'new_temperature_models': 15, 'h1_checkpoint_count': 1, 'test_evaluated': False}))
    print('SUITE EXIT 0', flush=True)
if __name__ == '__main__':
    main()

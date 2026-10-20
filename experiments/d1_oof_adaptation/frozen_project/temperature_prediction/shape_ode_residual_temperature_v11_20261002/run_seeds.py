"""Run the two predeclared seeds sequentially on one physical GPU."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'temperature_prediction/outputs' / HERE.name
LOGS = ROOT / 'temperature_prediction/logs' / HERE.name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / 'suite.pid').write_text(str(os.getpid()))
    (OUT / 'launch.json').write_text(json.dumps({'pid': os.getpid(), 'physical_gpu': args.gpu, 'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': ['base', 'residual'], 'seeds': [2024, 2025]}))
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': args.gpu, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    for seed in [2024, 2025]:
        name = 'shape_ode_residual_seed' + str(seed)
        with (LOGS / (name + '.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', str(HERE / 'train.py'), '--config', str(HERE / (name + '.json')), '--device', 'cuda:0'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (LOGS / (name + '.pid')).write_text(str(child.pid))
            (OUT / 'progress.json').write_text(json.dumps({'status': 'running', 'seed': seed, 'child_pid': child.pid, 'log': str(LOGS / (name + '.log'))}))
            print('START ' + name + ' PID ' + str(child.pid), flush=True)
            code = child.wait()
        (LOGS / (name + '.exit')).write_text(str(code))
        if code != 0:
            (OUT / (name + '_failure_record.md')).write_text(f"# {name}: failed\n\n- Date: 2026-10-02; project IFpaper_v1.\n- Purpose: shape ODE + scalar heads + frozen-base residual.\n- Exit {code}; PID {child.pid}.\n- Original log: {LOGS / (name + '.log')}.\n- Config/source: {HERE / (name + '.json')}; actual source snapshot under {OUT / name / 'frozen_source'}.\n- Suite stops without retry.\n")
            (OUT / 'progress.json').write_text(json.dumps({'status': 'failed', 'seed': seed, 'exit': code}))
            (LOGS / 'suite.exit').write_text(str(code))
            raise SystemExit(code)
        print('DONE ' + name, flush=True)
    report = subprocess.run([sys.executable, str(HERE / 'make_report.py')], cwd=ROOT, env=env)
    (LOGS / 'suite.exit').write_text(str(report.returncode))
    (OUT / 'progress.json').write_text(json.dumps({'status': 'completed' if report.returncode == 0 else 'report_failed', 'completed_training_runs': 2, 'test_evaluated': False}))
    print('SUITE EXIT ' + str(report.returncode), flush=True)
if __name__ == '__main__':
    main()

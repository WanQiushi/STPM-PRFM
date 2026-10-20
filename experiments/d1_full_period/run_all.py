"""Run the80 frozen inference jobs sequentially on physical GPU1."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, os, subprocess, sys
ROOT = Path(_release_expand('${WORKSPACE}'))
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / HERE.name
plan = _release_json_loads((HERE / 'plan.json').read_text())
state = dict(status='running', controller_pid=os.getpid(), start_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), physical_gpu=1, total_runs=len(plan['jobs']), completed=[], failed=[])

def save():
    (OUT / 'progress.json').write_text(json.dumps(state, indent=2))
save()
subprocess.run([sys.executable, str(HERE / 'prepare.py')], check=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
for i, job in enumerate(plan['jobs']):
    logfile = OUT / (job['key'] + '.log')
    with logfile.open('w') as log:
        process = subprocess.Popen([job['python'], '-u', str(HERE / 'evaluate.py'), '--job', str(i)], cwd=job['source'], env=env, stdout=log, stderr=subprocess.STDOUT)
        state['active'] = dict(key=job['key'], pid=process.pid, log=str(logfile))
        save()
        code = process.wait()
    (OUT / (job['key'] + '_status.json')).write_text(json.dumps(dict(key=job['key'], exit_code=code)))
    if code != 0:
        state['failed'].append(dict(key=job['key'], exit_code=code, log=str(logfile)))
        state['status'] = 'failed'
        save()
        (OUT / (job['key'] + '_failure.md')).write_text('# Inference failure: ' + job['key'] + '\n\n- Exit: ' + str(code) + '; log ' + str(logfile) + '.\n- Frozen weights not modified; no retry.\n')
        sys.exit(code)
    state['completed'].append(job['key'])
    save()
state.update(status='completed', active=None, finish_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
save()
print(json.dumps(state), flush=True)

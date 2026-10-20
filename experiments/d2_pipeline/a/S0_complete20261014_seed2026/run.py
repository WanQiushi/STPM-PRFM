"""Explicit method stages, with existing GPU-owner wait and CPU H1 overlap."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, json, os, shutil, subprocess, sys, time
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
cfg = _release_json_loads((HERE / 'config.json').read_text())
pipeline = _release_json_loads((HERE / 'pipeline.json').read_text())
out = HERE / 'output'
out.mkdir()
state = dict(status='running', started_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), arm=cfg['experiment_arm'], seed=cfg['seed'], configuration=str(HERE / 'config.json'), phases=[])

def save():
    (out / 'run_status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

def execute(phase, command):
    began = time.time()
    log = out / (phase + '.log')
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(phase=phase, command=command, pid=process.pid, log=str(log), status='running')
        state['phases'].append(entry)
        save()
        print('PHASE ' + json.dumps(entry), flush=True)
        code = process.wait()
    entry.update(status='completed' if code == 0 else 'failed', exit_code=code, seconds=time.time() - began)
    save()
    if code:
        state.update(status='failed', finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        save()
        record()
        sys.exit(code)

def record():
    (out / 'experiment_record.md').write_text('# D2.1 method experiment\n\n- Date: ' + state['finished_beijing'] + '\n- Project: IFpaper_v1.\n- Purpose/change: experiment_definition.json.\n- Configuration and input paths: config.json, pipeline.json, temperature_configs/.\n- Result: ' + state['status'] + '\n- Products and exact stage commands/PIDs/logs: ' + str(out / 'run_status.json') + '\n', encoding='utf-8')
jobs = pipeline['jobs'] if cfg['temperature_action'] in ['full', 'full_reuse_warm'] else [pipeline['jobs'][-1]]
if cfg['temperature_action'] == 'reference':
    jobs = []
if jobs and args.device.startswith('cuda'):
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=gpu_uuid', '--format=csv,noheader'], text=True).strip()

    def occupied_processes():
        occupied = []
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()
        for row in rows:
            fields = [part.strip() for part in row.split(',')]
            if fields[0] == uuid:
                owner = subprocess.check_output(['ps', '-p', fields[1], '-o', 'user='], text=True).strip()
                occupied.append(dict(pid=fields[1], owner=owner, memory=fields[2]))
        return occupied
    occupied = [item for item in occupied_processes() if int(item['pid']) not in cfg['authorized_shared_gpu_pids']]
    while occupied:
        print('RESOURCE_WAIT ' + json.dumps(dict(gpu=gpu, processes=occupied)), flush=True)
        time.sleep(30)
        occupied = [item for item in occupied_processes() if int(item['pid']) not in cfg['authorized_shared_gpu_pids']]
execute('prepare', [sys.executable, '-u', str(HERE / 'prepare_data.py')])
pretrain = None
if cfg['response_action'] == 'train':
    log = (out / 'H1_pretraining.log').open('w')
    command = [sys.executable, '-u', str(HERE / 'response_training.py'), '--config', str(HERE / 'config.json'), '--device', 'cpu', '--phase', 'pretraining']
    pretrain = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    pretrain_entry = dict(phase='H1_pretraining', pid=pretrain.pid, command=command, log=str(out / 'H1_pretraining.log'), status='running')
    state['phases'].append(pretrain_entry)
    save()
elif cfg['response_action'] == 'reuse_pretrain_adapt':
    reference = HERE / cfg['pretraining_reference'] / 'output/response'
    for name in ['pretrained_best.pt', 'pretraining_summary.json', 'pretraining_history.csv']:
        shutil.copy2(reference / name, out / 'response' / name)
for job in jobs:
    if cfg['temperature_action'] in ['full', 'final_fresh']:
        execute(job['name'] + '_v9', [sys.executable, '-u', str(HERE / 'temperature_code/train_warm.py'), '--config', str(HERE / job['warm_config']), '--device', args.device])
    if cfg['temperature_action'] != 'base_export':
        execute(job['name'] + '_A2', [sys.executable, '-u', str(HERE / 'temperature_code/train.py'), '--config', str(HERE / job['config']), '--device', args.device])
    execute(job['name'] + '_forecast', [sys.executable, '-u', str(HERE / 'temperature_code/export_forecasts.py'), '--config', str(HERE / job['config']), '--device', args.device])
if pretrain is not None:
    code = pretrain.wait()
    log.close()
    pretrain_entry.update(status='completed' if code == 0 else 'failed', exit_code=code)
    save()
    if code:
        state.update(status='failed', finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        save()
        record()
        sys.exit(code)
if cfg['response_action'] in ['train', 'reuse_pretrain_adapt']:
    execute('forecast_cache', [sys.executable, '-u', str(HERE / 'prepare_forecast_cache.py')])
    execute('H1_adaptation', [sys.executable, '-u', str(HERE / 'response_training.py'), '--config', str(HERE / 'config.json'), '--device', 'cpu', '--phase', 'adaptation'])
execute('evaluation', [sys.executable, '-u', str(HERE / 'evaluate.py'), '--device', 'cpu'])
state.update(status='completed', finished_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
save()
record()
print('RUN_FINISHED ' + json.dumps(state), flush=True)

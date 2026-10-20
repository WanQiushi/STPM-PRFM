"""Detach all user-authorized core ablations; report PID/log for initial checks."""
from if_release.runtime import expand as _release_expand
import json
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(_release_expand('${WORKSPACE}'))
NAME = 'va_core_ablation_all_20261008'
LOGS = ROOT / 'temperature_prediction/logs' / NAME
LOGS.mkdir(parents=True, exist_ok=True)
command = [_release_expand('${PYTHON}'), '-u', str(ROOT / 'temperature_prediction' / NAME / 'run_all.py')]
with (LOGS / 'suite.log').open('w') as log:
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {'model': 'va', 'started_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'controller_pid': process.pid, 'command': command, 'physical_gpus': [0, 1], 'workdir': str(ROOT), 'log': str(LOGS / 'suite.log'), 'output': str(ROOT / 'temperature_prediction/outputs' / NAME)}
(LOGS / 'dispatch.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
(LOGS / 'suite.pid').write_text(str(process.pid))
print(json.dumps(record), flush=True)

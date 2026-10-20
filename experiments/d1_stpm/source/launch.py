"""Detach user-requested sampling experiments on GPU0+1."""
from if_release.runtime import expand as _release_expand
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root = Path(_release_expand('${WORKSPACE}'))
name = 'va_2_2_sampling_20261008'
logs = root / 'temperature_prediction/logs' / name
logs.mkdir(parents=True, exist_ok=True)
command = [_release_expand('${PYTHON}'), '-u', str(root / 'temperature_prediction' / name / 'run_dual_suite.py'), '--gpus', '0,1']
with (logs / 'suite.log').open('w') as log:
    process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
dispatch = {'version': 'va-2.2', 'controller_pid': process.pid, 'command': command, 'physical_gpus': [0, 1], 'workdir': str(root), 'log': str(logs / 'suite.log'), 'sample_minutes': [10, 5, 1], 'temperature_seeds_per_resolution': 5, 'started_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
(logs / 'dispatch.json').write_text(json.dumps(dispatch, indent=2), encoding='utf-8')
print(json.dumps(dispatch))

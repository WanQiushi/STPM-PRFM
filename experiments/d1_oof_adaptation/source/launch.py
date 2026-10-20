"""Detach the va-1 controller; report the process and its log."""
from if_release.runtime import loads as _release_json_loads
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = _release_json_loads((HERE / 'settings.json').read_text(encoding='utf-8'))
LOG = ROOT / 'outputs' / (CFG['experiment_id'] + '_controller.log')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=str(CFG['gpu']), OMP_NUM_THREADS=str(CFG['threads']), MKL_NUM_THREADS=str(CFG['threads']))
with LOG.open('w', encoding='utf-8') as log:
    child = subprocess.Popen([sys.executable, '-u', str(HERE / 'run.py')], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
info = dict(version=CFG['version'], controller_pid=child.pid, gpu=CFG['gpu'], working_directory=str(ROOT), controller_log=str(LOG), output=str(ROOT / CFG['output']), launched_utc=datetime.now(timezone.utc).isoformat())
(HERE / 'launch_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(info, ensure_ascii=False), flush=True)

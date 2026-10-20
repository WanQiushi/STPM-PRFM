"""Paths and explicit experiment I/O; no GPU or training at import time."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
ROOT = Path(__file__).resolve().parent.parent
ZONE = timezone(timedelta(hours=8))

def now():
    return datetime.now(ZONE).isoformat()

def read(path):
    return _release_json_loads(Path(path).read_text(encoding='utf-8'))

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def host_settings(host):
    return read(ROOT / 'hosts.json')[host]

def run_key(model, seed):
    return f'{model}_seed{seed}'

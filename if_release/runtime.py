"""Resolve public path tokens using a local runtime configuration."""
from pathlib import Path
import getpass
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\$\{([A-Z_]+)(?::([^}]+))?\}")


def settings():
    path = Path(os.environ.get("IF_RELEASE_CONFIG", ROOT / "config/runtime.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def expand(value):
    if not isinstance(value, str) or "${" not in value:
        return value
    cfg = settings()

    def resolve(match):
        key, item = match.groups()
        if key == "PRIVATE_D1":
            return cfg["private_d1"][item]
        if key == "PYTHON":
            return Path(sys.executable).as_posix()
        if key == "USER":
            return getpass.getuser()
        if key == "GPU":
            return cfg["physical_gpu"]
        path = Path(cfg["paths"][key])
        return (path if path.is_absolute() else ROOT / path).as_posix()

    match = TOKEN.fullmatch(value)
    if match:
        return resolve(match)
    return TOKEN.sub(lambda m: str(resolve(m)), value)


def expand_tree(value):
    if isinstance(value, dict):
        return {k: expand_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_tree(v) for v in value]
    return expand(value)


def loads(*args, **kwargs):
    return expand_tree(json.loads(*args, **kwargs))


def load(*args, **kwargs):
    return expand_tree(json.load(*args, **kwargs))

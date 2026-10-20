"""Run an experiment with the public runtime configuration."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from if_release.runtime import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=ROOT / "config/runtime.json")
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ["IF_RELEASE_CONFIG"] = str(args.runtime.resolve())
    cfg = settings()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["physical_gpu"])
    result = subprocess.run(
        [sys.executable, str(ROOT / args.script), *args.script_args],
        cwd=ROOT,
        env=env,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

from if_release.runtime import expand as _release_expand
from pathlib import Path
import subprocess, json, datetime
from zoneinfo import ZoneInfo
r = Path(_release_expand('${WORKSPACE}'))
o = r / 'outputs/baseline_lstm_mamba_pcfnet_20261015'
env = r / 'envs/mamba_py311_20261015'

def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
commands = [[_release_expand('${CONDA}'), 'create', '-p', str(env), 'python=3.11', 'pip', '-y'], [_release_expand('${PYTHON}'), '-m', 'pip', '--python', str(env / 'bin/python'), 'install', 'torch==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], [_release_expand('${PYTHON}'), '-m', 'pip', '--python', str(env / 'bin/python'), 'install', 'numpy', 'pandas', 'einops', 'packaging', 'ninja', 'setuptools', 'wheel', 'transformers==4.46.3'], [_release_expand('${PYTHON}'), '-m', 'pip', '--python', str(env / 'bin/python'), 'install', '--no-deps', 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl', 'https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl']]
state = dict(status='running', start_beijing=now(), environment=str(env), completed_steps=[])
(o / 'mamba_environment_status.json').write_text(json.dumps(state, indent=2))
for i, cmd in enumerate(commands):
    print('ENVIRONMENT_STEP', i, cmd, flush=True)
    code = subprocess.run(cmd).returncode
    if code != 0:
        state.update(status='failed', failed_step=i, exit_code=code, finish_beijing=now())
        (o / 'mamba_environment_status.json').write_text(json.dumps(state, indent=2))
        (o / 'experiment_records/mamba_environment_setup.md').write_text('# Mamba dependency setup\n\n- Outcome: failed at step' + str(i) + '; exit' + str(code) + '.\n- Log: mamba_environment.log; no training or automatic retry.\n')
        raise SystemExit(code)
    state['completed_steps'].append(i)
    (o / 'mamba_environment_status.json').write_text(json.dumps(state, indent=2))
state.update(status='completed', finish_beijing=now())
(o / 'mamba_environment_status.json').write_text(json.dumps(state, indent=2))
(o / 'experiment_records/mamba_environment_setup.md').write_text('# Mamba dependency setup\n\n- Outcome: completed.\n- Environment: ' + str(env) + '\n- Purpose: official CUDA Mamba1 kernels, isolated from shared base Python.\n- Log: mamba_environment.log.\n')

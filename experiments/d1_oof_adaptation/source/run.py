"""Snapshot and run the authorized va-1 stages on GPU0."""
from if_release.runtime import loads as _release_json_loads
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = _release_json_loads((HERE / 'settings.json').read_text(encoding='utf-8'))
OUT = ROOT / CFG['output']
ORIGINAL = ROOT / CFG['original_source_output']
FORECAST = ROOT / CFG['forecast_source_output']

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def record(stage, result):
    (OUT / (stage + '_experiment_record.md')).write_text('# va-1 / ' + stage + '\n\n- 日期UTC：' + datetime.now(timezone.utc).isoformat() + '\n- 项目：IFpaper_v1；目的：固定v11、更新原H1活跃参数以适应因果温度预报。\n- 数据/接口/配置：README.md、settings.json、normalization.json、data_report.json和temperature_cache折溯源。\n- 本次操作与结果：' + stage + '；' + result + '。\n- 代码与权重：frozen_source、frozen_project、h1_original.pt、v11_original.pt。\n- 输出：' + str(OUT) + '；对应阶段log、equivalence.json、history.csv、best_va1.pt、comparison.json。\n', encoding='utf-8')
OUT.mkdir(parents=True)
shutil.copytree(HERE, OUT / 'frozen_source', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(ORIGINAL / 'frozen_project', OUT / 'frozen_project', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(FORECAST / 'temperature_cache', OUT / 'temperature_cache')
for filename in ['h1_original.pt', 'v11_original.pt', 'normalization.json']:
    shutil.copy(ORIGINAL / filename, OUT / filename)
shutil.copy(FORECAST / 'window_origins.json', OUT / 'window_origins.json')
shutil.copy(HERE / 'README.md', OUT / 'README.md')
dump(OUT / 'settings.json', CFG)
os.environ['CUDA_VISIBLE_DEVICES'] = str(CFG['gpu'])
os.environ['OMP_NUM_THREADS'] = str(CFG['threads'])
os.environ['MKL_NUM_THREADS'] = str(CFG['threads'])
started = datetime.now(timezone.utc).isoformat()
record('startup', '代码/权重/固定预报已快照；准备运行，尚未完成')
stages = [('prepare', [str(OUT / 'frozen_source/prepare.py'), str(ROOT), str(OUT)]), ('equivalence', [str(OUT / 'frozen_source/equivalence.py'), '--project', str(OUT / 'frozen_project'), '--h1-checkpoint', str(OUT / 'h1_original.pt'), '--v11-checkpoint', str(OUT / 'v11_original.pt'), '--cache', str(OUT / 'cache/train.npz'), '--output', str(OUT / 'equivalence.json'), '--device', 'cuda:0']), ('fit', [str(OUT / 'frozen_source/train.py'), str(ROOT), str(OUT), 'fit']), ('finish', [str(OUT / 'frozen_source/train.py'), str(ROOT), str(OUT), 'finish'])]
for stage, arguments in stages:
    with (OUT / (stage + '.log')).open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', *arguments], stdout=log, stderr=subprocess.STDOUT)
        dump(OUT / 'job_status.json', dict(status='running', stage=stage, pid=os.getpid(), child_pid=child.pid, gpu=CFG['gpu'], started_utc=started))
        code = child.wait()
    record(stage, '完成' if code == 0 else '失败；exit_code=' + str(code))
    if code != 0:
        dump(OUT / 'job_status.json', dict(status='failed', stage=stage, exit_code=code, pid=os.getpid(), gpu=CFG['gpu'], started_utc=started))
        sys.exit(code)
dump(OUT / 'job_status.json', dict(status='completed', pid=os.getpid(), gpu=CFG['gpu'], started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat()))
record('completed', '验证选模与最终测试完成；结果见comparison.json，尚未人工分析')

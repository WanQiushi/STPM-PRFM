"""Run one experiment and keep the requested completion/failure archive record."""
from if_release.runtime import loads as _release_json_loads
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
config = Path(sys.argv[1]).resolve()
cfg = _release_json_loads(config.read_text(encoding='utf-8'))
project = Path(__file__).resolve().parents[2]
out = project / cfg['output']
out.mkdir(parents=True, exist_ok=True)
zone = timezone(timedelta(hours=8))
started = datetime.now(zone)
with (out / 'train.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('train.py')), '--config', str(config), '--device', 'cuda:0'], cwd=project, stdout=log, stderr=subprocess.STDOUT)
    status = dict(status='running', pid=process.pid, gpu=0, started_at=started.isoformat(), config=str(config), log=str(out / 'train.log'), output=str(out))
    (out / 'job_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)
    code = process.wait()
ended = datetime.now(zone)
status.update(status='completed' if code == 0 else 'failed', ended_at=ended.isoformat(), returncode=code)
(out / 'job_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
(out / 'exit_code.txt').write_text(str(code), encoding='utf-8')
record_dir = project / 'outputs/experiment_records'
record_dir.mkdir(exist_ok=True)
result_text = '失败；完整原始错误保留在train.log，未自动重试。'
if code == 0:
    result = _release_json_loads((out / 'suite_results.json').read_text(encoding='utf-8'))
    result_text = '\n'.join((f"- {key}：验证RMSE {value['validation']['pooled_rmse_mm']:.6f} mm；测试RMSE {value['test']['pooled_rmse_mm']:.6f} mm。" for key, value in result['results'].items()))
record = record_dir / (ended.strftime('%Y-%m-%d_%H%M%S_') + cfg['run_name'] + ('_完成.md' if code == 0 else '_失败.md'))
record.write_text(f"# {cfg['version_label']} 组合模型实验\n\n- 项目：IFpaper_v1；开始{started.isoformat()}，结束{ended.isoformat()}；GPU0；seed2024；返回码{code}。\n- 目的：统一端点分解，检验LF锚点/关闭显式季节映射与可逆+BW记忆MF是否可以结合。\n- 配置：{config}；入口：{Path(__file__).with_name('train.py')}。\n- 数据合同沿用2026-09-30统一长测试及单种子记录；分项损失权重沿用v2，q面积权重0；MF残差尺度采用v#的0.1。无原模型删除。\n{result_text}\n- 检查点、完整指标及NPZ：{out}；日志：{out / 'train.log'}。\n", encoding='utf-8')
status['record'] = str(record)
(out / 'job_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
print(json.dumps(status), flush=True)
sys.exit(code)

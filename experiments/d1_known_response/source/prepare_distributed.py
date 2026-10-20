"""Prepare the local seed assignment and portable, unchanged model inputs."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import shutil
import tarfile
import torch
OUT = Path(__file__).resolve().parent.parent
progress = _release_json_loads((OUT / 'progress.json').read_text())
assignment = dict(created_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(), local_seed=2028, local_jobs=[f'O{i}_seed2028' for i in [7, 6, 0, 1, 2, 3, 4, 5, 8]], local_max_workers=2, gpu0_max_workers=8, gpu1_max_workers=4, gpu1_budget_seconds=9000, gpu1_budget_policy='Admit complete jobs fitting estimated remaining budget; finish launched jobs; remaining work goes toGPU0.', previous_progress=progress, paused_scheduler_pids=[], paused_training_pids=[job['pid'] for job in progress['active']], model_training_changed=False)
(OUT / 'distributed_assignment.json').write_text(json.dumps(assignment, indent=2), encoding='utf-8')
portable = OUT.parent / 'd1_observed_temperature_local_input_20261020'
portable.mkdir()
(portable / 'source').mkdir()
for name in ['train_one.py', 'research_model.py', 'local_controller.py']:
    shutil.copy2(OUT / 'source' / name, portable / 'source' / name)
shutil.copytree(OUT / 'source/frozen_project', portable / 'source/frozen_project')
shutil.copytree(OUT / 'dataset', portable / 'dataset')
old = torch.load(_release_expand('${WORKSPACE}/outputs/va_1_20261004/h1_original.pt'), map_location='cpu', weights_only=False)
cutoffs = {'model': {key: old['model'][key] for key in ['decomposer.theta_static', 'decomposer.theta_noise']}}
torch.save(cutoffs, portable / 'cutoffs_only.pt')
config = _release_json_loads((OUT / 'protocol.json').read_text())
config.update(original_checkpoint=_release_expand('${RESULTS}/d1_observed_temperature_mechanism_20261020_051745/local_seed2028/cutoffs_only.pt'), execution_node='runtime', physical_gpu=_release_expand('${GPU}'))
(portable / 'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
(portable / 'local_assignment.json').write_text(json.dumps(assignment, indent=2), encoding='utf-8')
with tarfile.open(portable.parent / 'd1_observed_temperature_local_input_20261020.tar.gz', 'w:gz') as archive:
    for item in portable.iterdir():
        archive.add(item, arcname=item.name)
print('PREPARED_LOCAL', str(portable.parent / 'd1_observed_temperature_local_input_20261020.tar.gz'), flush=True)

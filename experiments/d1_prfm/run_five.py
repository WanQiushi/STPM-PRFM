"""Sequential single-GPU training and paired evaluation; no automatic retry."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--gpu', default='0', choices=['0', '1'])
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    args.records.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    raw = args.root / 'data/mainbridge_temperature_v2/minute_complete.npz'
    for seed in range(2024, 2029):
        stamp = datetime.now().astimezone().isoformat()
        tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        response_output = args.output / f'PRFM_seed{seed}'
        commands = [('training', [sys.executable, str(HERE / 'train_prfm.py'), '--seed', str(seed), '--data', str(raw), '--output', str(response_output)]), ('evaluation', [sys.executable, str(HERE / 'evaluate.py'), '--seed', str(seed), '--root', str(args.root), '--dataset', str(args.dataset), '--output', str(args.output / 'evaluation'), '--response-checkpoint', str(response_output / 'adapted_best.pt')])]
        status, phase = ('interrupted', 'training')
        try:
            for phase, command in commands:
                with (args.output / f'seed{seed}_{phase}.log').open('w', encoding='utf-8') as log:
                    code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                if code != 0:
                    status = f'failed ({phase}, exit {code})'
                    return code
            status = 'completed'
        finally:
            record = f"# D1 independent PRFM seed {seed}\n\n- Started: {stamp}; project: IFpaper_v1.\n- Purpose: repeat PRFM pretraining and forecast adaptation from a fresh initialization.\n- Source/configuration: {HERE}; seed {seed}; physical GPU {args.gpu}.\n- Dataset: {raw}; original normalization and 68/26 pretraining origins in config/.\n- Adaptation inputs: {HERE / 'cache'}; fixed chronological OOF 110/26 windows.\n- Paired evaluation data: {args.dataset}; original 110 validation and 109 test days.\n- STPM: existing independently trained checkpoint with matching seed; see config/temperature_jobs.json.\n- Result: {status}; last phase: {phase}.\n- Checkpoints and histories: {response_output}.\n- Metrics and predictions: {args.output / 'evaluation' / f'S0_seed{seed}'}.\n- Logs: {args.output / f'seed{seed}_training.log'} and seed{seed}_evaluation.log.\n- This run holds the OOF temperature producer fixed; it does not repeat the complete OOF pipeline.\n"
            (args.records / f'{tag}_d1_prfm_seed{seed}.md').write_text(record, encoding='utf-8')
    subprocess.run([sys.executable, str(HERE / 'summarize.py'), '--output', str(args.output)], check=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

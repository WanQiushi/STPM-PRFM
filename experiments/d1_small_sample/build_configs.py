"""Write configurations and a job index without importing or running a model."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
import argparse
import json
from pathlib import Path
PROJECT = Path(__file__).resolve().parent
MODELS = ['A2', 'DLinear', 'iTransformer', 'LSTM', 'TimeXer', 'PatchTST', 'TimesNet', 'SegRNN', 'ModernTCN']
MONTHS = [9, 6, 3, 1]
SEEDS = list(range(2024, 2029))
TEMPERATURE = {'width': 32, 'latent_dim': 8, 'residual_width': 16, 'batch_size': 32, 'epochs': 80, 'epochs_base': 80, 'epochs_residual': 60, 'patience': 12, 'lr': 0.001, 'lr_base': 0.001, 'lr_residual': 0.001, 'level_weight': 0.0, 'amplitude_weight': 0.0, 'shape_weight': 0.25, 'trend_weight': 3.0}
ADAPTATION = {'batch_size': 16, 'epochs': 80, 'patience': 20, 'lr': 0.0001, 'grad_clip': 1.0, 'ode_weight': 0.001}
BASELINE_FIELDS = ['model', 'model_args', 'batch_size', 'micro_batch_size', 'epochs', 'patience', 'lr', 'grad_clip', 'torch_threads']

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def build(record_dir, output_prefix):
    response = _release_json_loads((PROJECT / 'vendor/h1_project/experiments/v2_hybrid_kqsz_20261001/config_h1_seed2024.json').read_text(encoding='utf-8'))
    jobs = []
    for month in MONTHS:
        budget = f'{month}m'
        for model in MODELS:
            for seed in SEEDS:
                run_name = f'D1_{budget}_{model}_seed{seed}'
                config_path = f'configs/{budget}/{model}_seed{seed}.json'
                output = (Path(output_prefix) / budget / f'{model}_seed{seed}').as_posix()
                cfg = {'protocol_path': f'protocols/{budget}/protocol.json', 'model': model, 'seed': seed, 'run_name': run_name, 'output': output, 'record_dir': record_dir, 'budget': budget, 'full_data_weights_used': False, 'test_used_for_selection': False}
                if model == 'A2':
                    cfg.update(device='cuda:0', temperature=TEMPERATURE, response=response, adaptation=ADAPTATION, reference_configuration='metadata/a2_original_config.json')
                else:
                    source_path = f'metadata/frozen_baseline_configs/{model}_seed{seed}.json'
                    frozen = _release_json_loads((PROJECT / source_path).read_text(encoding='utf-8'))
                    cfg.update({key: frozen[key] for key in BASELINE_FIELDS})
                    cfg.update(reference_configuration=source_path, first_difference_weight=0.0, second_difference_weight=0.0, loss_policy='Observed-entry masked standardized five-channel MSE.', checkpoint_selection='Joint five-channel standardized validation RMSE.')
                write(PROJECT / config_path, cfg)
                jobs.append({'budget': budget, 'month': month, 'model': model, 'seed': seed, 'config': config_path, 'output': output, 'entry': 'a2/train.py' if model == 'A2' else 'baseline/train.py'})
    write(PROJECT / 'job_index.json', {'version': '20261013', 'training_started': False, 'months': MONTHS, 'models': MODELS, 'seeds': SEEDS, 'jobs': jobs})
    print(f'Prepared {len(jobs)} configurations; no training was started.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir', default=_release_expand('${RECORDS}/d1_small_sample_20261013'))
    parser.add_argument('--output-prefix', default='outputs')
    args = parser.parse_args()
    build(args.record_dir, args.output_prefix)
if __name__ == '__main__':
    main()

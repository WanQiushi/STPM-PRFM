"""One formal response training run, validation selection, and final evaluation."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import csv
import json
import os
import platform
import random
import time
import numpy as np
import torch
from research_model import ResearchPRFM
OUT = Path(__file__).resolve().parent.parent
BEIJING = timezone(timedelta(hours=8))

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def load_split(name, minutes, device):
    folder = OUT / 'dataset'
    arrays = np.load(folder / f'{name}_{minutes}min.npz')
    return dict(history=torch.as_tensor(np.load(folder / f'{name}_history_{minutes}min.npy'), device=device), temperature=torch.as_tensor(arrays['temperature'], device=device), target=torch.as_tensor(arrays['target'], device=device), mask=torch.as_tensor(arrays['mask'], device=device), origin=torch.as_tensor(arrays['origin'], device=device), std=torch.as_tensor(arrays['std'][1:], dtype=torch.float32, device=device))

@torch.no_grad()
def cache(model, data, batch_size):
    pieces = []
    for first in range(0, len(data['origin']), batch_size):
        sl = slice(first, first + batch_size)
        pieces.append(model.prepare_inputs(data['history'][sl], data['temperature'][sl], data['origin'][sl]))
    return {k: torch.cat([x[k] for x in pieces]) for k in pieces[0]}

@torch.no_grad()
def evaluate(model, inputs, data, batch_size):
    sse = 0.0
    count = 0
    for first in range(0, len(data['origin']), batch_size):
        sl = slice(first, first + batch_size)
        prediction = model({k: v[sl] for k, v in inputs.items()})['prediction']
        error = (prediction - data['target'][sl]) * data['std']
        mask = data['mask'][sl]
        sse += float(torch.where(mask, error.double().square(), 0).sum())
        count += int(mask.sum())
    return (sse / count) ** 0.5

@torch.no_grad()
def export(model, name, minutes, folder, device, batch_size):
    data = load_split(name, minutes, device)
    collected = {}
    for first in range(0, len(data['origin']), batch_size):
        sl = slice(first, first + batch_size)
        inputs = model.prepare_inputs(data['history'][sl], data['temperature'][sl], data['origin'][sl])
        result = model(inputs, trace=True)
        result['mf_temperature'] = inputs['mf_temperature']
        for key, value in result.items():
            if key != 'rate_penalty':
                collected.setdefault(key, []).append(value.cpu().numpy())
    collected = {k: np.concatenate(v) for k, v in collected.items()}
    target = data['target'].cpu().numpy()
    mask = data['mask'].cpu().numpy()
    origins = data['origin'].cpu().numpy()
    std = data['std'].cpu().numpy().astype(np.float64)
    error = (collected['prediction'].astype(np.float64) - target.astype(np.float64)) * std
    daily_sse = np.where(mask, error ** 2, 0).sum(1)
    daily_sae = np.where(mask, np.abs(error), 0).sum(1)
    daily_n = mask.sum(1)
    metrics = dict(windows=len(origins), valid_labels=int(daily_n.sum()), rmse_mm=float(np.sqrt(daily_sse.sum() / daily_n.sum())), mae_mm=float(daily_sae.sum() / daily_n.sum()), mse_mm2=float(daily_sse.sum() / daily_n.sum()), per_location_rmse_mm=np.sqrt(daily_sse.sum(0) / daily_n.sum(0)).tolist())
    np.savez_compressed(folder / f'{name}_{minutes}min_predictions.npz', **collected, target=target, mask=mask, origin=origins, response_std=std, temperature=data['temperature'].cpu().numpy())
    np.savez_compressed(folder / f'{name}_{minutes}min_daily_errors.npz', origin=origins, sse=daily_sse, sae=daily_sae, n=daily_n)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    config = _release_json_loads((OUT / 'protocol.json').read_text())
    config.update(arm=args.arm, seed=args.seed)
    folder = OUT / f'{args.arm}_seed{args.seed}'
    folder.mkdir()
    dump(folder / 'config.json', config)
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0')
    model = ResearchPRFM(config, args.arm).to(device)
    frozen = torch.load(config['original_checkpoint'], map_location='cpu', weights_only=False)
    model.decomposer.theta_static.data.copy_(frozen['model']['decomposer.theta_static'].to(device))
    model.decomposer.theta_noise.data.copy_(frozen['model']['decomposer.theta_noise'].to(device))
    torch.save(dict(model=model.state_dict(), config=config), folder / 'initial.pt')
    dump(folder / 'initial_parameters.json', model.parameter_report())
    train = load_split('train', 15, device)
    validation = load_split('validation', 15, device)
    train_inputs = cache(model, train, config['batch_size'])
    val_inputs = cache(model, validation, config['batch_size'])
    del train['history'], validation['history']
    active = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active, lr=config['learning_rate'])
    order_generator = torch.Generator(device='cpu').manual_seed(args.seed)
    best = float('inf')
    bad = 0
    start = time.time()
    print('TRAIN_START', args.arm, args.seed, model.parameter_report(), flush=True)
    with (folder / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_objective', 'train_mse', 'train_rate_penalty', 'validation_rmse_mm', 'best_validation_rmse_mm', 'best_epoch', 'elapsed_seconds'])
        writer.writeheader()
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            order = torch.randperm(len(train['origin']), generator=order_generator).to(device)
            sums = np.zeros(3)
            nb = 0
            for first in range(0, len(order), config['batch_size']):
                ix = order[first:first + config['batch_size']]
                output = model({k: v[ix] for k, v in train_inputs.items()})
                err = output['prediction'] - train['target'][ix]
                mask = train['mask'][ix]
                mse = torch.where(mask, err.square(), 0).sum() / mask.sum()
                rate = output['rate_penalty']
                loss = mse + config['ode_weight'] * rate
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(active, config['grad_clip'])
                optimizer.step()
                sums += np.array([float(loss.detach()), float(mse.detach()), float(rate.detach())])
                nb += 1
            model.eval()
            score = evaluate(model, val_inputs, validation, config['batch_size'])
            if score < best:
                best = score
                best_epoch = epoch
                bad = 0
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch, validation_rmse_mm=score), folder / 'best.pt')
            else:
                bad += 1
            row = dict(epoch=epoch, train_objective=sums[0] / nb, train_mse=sums[1] / nb, train_rate_penalty=sums[2] / nb, validation_rmse_mm=score, best_validation_rmse_mm=best, best_epoch=best_epoch, elapsed_seconds=time.time() - start)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, epoch=epoch, order_generator_state=order_generator.get_state()), folder / 'last.pt')
            if bad >= config['patience']:
                break
    checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dump(folder / 'selected_parameters.json', model.parameter_report())
    metrics = {}
    for name in ['validation', 'test']:
        metrics[f'{name}_15min'] = export(model, name, 15, folder, device, config['batch_size'])
    del train_inputs, val_inputs, train, validation, optimizer
    minute_config = {**config, 'sample_minutes': 1}
    minute_model = ResearchPRFM(minute_config, args.arm).to(device).eval()
    state = dict(checkpoint['model'])
    for key in ['decomposer.frequency', 'decomposer.padded_frequency']:
        state[key] = minute_model.state_dict()[key]
    minute_model.load_state_dict(state)
    for name in ['validation', 'test']:
        metrics[f'{name}_1min'] = export(minute_model, name, 1, folder, device, 8)
    finished = datetime.now(BEIJING).isoformat()
    summary = dict(status='completed', arm=args.arm, seed=args.seed, epochs_ran=epoch, selected_epoch=checkpoint['epoch'], selection='52 real-temperature validation days,15min measured response RMSE', metrics=metrics, elapsed_seconds=time.time() - start, finished_beijing=finished, frozen_code_or_weights_modified=False, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES', '0'), execution_node=os.environ.get('D1_EXECUTION_NODE', 'server_gpu0'), execution_host=platform.node(), python_version=platform.python_version(), torch_version=torch.__version__, numpy_version=np.__version__)
    dump(folder / 'summary.json', summary)
    folder.joinpath('experiment_record.md').write_text(f"# D1 observed-temperature response: {args.arm}, seed{args.seed}\n\n- Date: {finished}; IFpaper_v1.\n- Purpose: independent response-structure comparison with observed future temperature.\n- Dataset/interface/config: ../dataset/README.md, ../protocol.json, config.json; train15min, final1min.\n- Changes: research variant only; original frozen source/weights unchanged.\n- Result: selected epoch{checkpoint['epoch']}; test RMSE{metrics['test_1min']['rmse_mm']:.9f}mm.\n- Products: initial.pt, best.pt, last.pt, history.csv, selected_parameters.json, summary.json, predictions/daily_errors NPZ.\n", encoding='utf-8')
    print('COMPLETED', json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()

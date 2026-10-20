"""One new baseline run on its explicitly assigned GPU. No H1 training entry."""
import os
import argparse
from common import ROOT, read, dump, host_settings, now
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
cfg = read(args.config)
host = host_settings(cfg['host'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(host['physical_gpu'])
import csv
import platform
import random
import time
import numpy as np
import torch
from data import load_split, model_inputs, response_metrics
from known_temperature_models import KnownTemperatureBaseline

@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    sse = 0.0
    count = 0
    for start in range(0, len(data['origin']), batch_size):
        ix = slice(start, start + batch_size)
        prediction = model(*model_inputs(data, ix))['prediction']
        error = (prediction - data['target'][ix]) * data['std']
        mask = data['mask'][ix]
        sse += float(torch.where(mask, error.double().square(), 0).sum())
        count += int(mask.sum())
    return (sse / count) ** 0.5

@torch.no_grad()
def export(model, data, batch_size, out, split):
    model.eval()
    saved = []
    for start in range(0, len(data['origin']), batch_size):
        saved.append(model(*model_inputs(data, slice(start, start + batch_size)))['prediction'].cpu().numpy())
    prediction = np.concatenate(saved)
    target = data['target'].cpu().numpy()
    mask = data['mask'].cpu().numpy()
    origin = data['origin'].cpu().numpy()
    std = data['std'].cpu().numpy().astype(np.float64)
    metric, daily = response_metrics(prediction, target, mask, std)
    np.savez_compressed(out / f'{split}_predictions.npz', prediction=prediction, target=target, mask=mask, origin=origin, response_std=std, response_mean=data['mean'], temperature=data['temperature'].cpu().numpy())
    np.savez_compressed(out / f'{split}_daily_errors.npz', origin=origin, **daily)
    return metric

def main():
    key = f"{cfg['model']}_seed{cfg['seed']}"
    out = ROOT / 'runs' / key
    out.mkdir(parents=True)
    dump(out / 'config.json', cfg)
    start = time.time()
    try:
        torch.set_num_threads(cfg['cpu_threads'])
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        device = torch.device('cuda:0')
        train = load_split(host['dataset'], 'train', device)
        validation = load_split(host['dataset'], 'validation', device)
        model = KnownTemperatureBaseline(cfg).to(device)
        dump(out / 'parameters.json', model.parameter_report())
        torch.save(dict(model=model.state_dict(), config=cfg), out / 'initial.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        order_generator = torch.Generator(device='cpu').manual_seed(cfg['seed'])
        best = float('inf')
        bad = 0
        best_epoch = 0
        print('TRAIN_START', key, model.parameter_report(), flush=True)
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_mse', 'validation_rmse_mm', 'best_validation_rmse_mm', 'best_epoch', 'elapsed_seconds'])
            writer.writeheader()
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                order = torch.randperm(len(train['origin']), generator=order_generator).to(device)
                total = 0.0
                batches = 0
                for first in range(0, len(order), cfg['batch_size']):
                    ix = order[first:first + cfg['batch_size']]
                    prediction = model(*model_inputs(train, ix))['prediction']
                    error = prediction - train['target'][ix]
                    mask = train['mask'][ix]
                    loss = torch.where(mask, error.square(), 0).sum() / mask.sum()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    optimizer.step()
                    total += float(loss.detach())
                    batches += 1
                score = evaluate(model, validation, cfg['batch_size'])
                if score < best:
                    best = score
                    best_epoch = epoch
                    bad = 0
                    torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, validation_rmse_mm=score), out / 'best.pt')
                else:
                    bad += 1
                row = dict(epoch=epoch, train_mse=total / batches, validation_rmse_mm=score, best_validation_rmse_mm=best, best_epoch=best_epoch, elapsed_seconds=time.time() - start)
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg, epoch=epoch, order_generator_state=order_generator.get_state()), out / 'last.pt')
                if bad >= cfg['patience']:
                    break
        selected = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(selected['model'])
        metrics = dict(validation=export(model, validation, cfg['batch_size'], out, 'validation'))
        del train, validation, optimizer
        test = load_split(host['dataset'], 'test', device)
        metrics['test'] = export(model, test, cfg['batch_size'], out, 'test')
        summary = dict(status='completed', model=cfg['model'], label=cfg['label'], seed=cfg['seed'], host=cfg['host'], physical_gpu=host['physical_gpu'], selected_epoch=best_epoch, epochs_ran=epoch, sample_minutes=15, metrics=metrics, parameters=model.parameter_report(), elapsed_seconds=time.time() - start, finished_beijing=now(), execution_host=platform.node(), python_version=platform.python_version(), torch_version=torch.__version__, numpy_version=np.__version__, future_response_input=False, selection=cfg['selection'], test_used_for_selection=False)
        dump(out / 'summary.json', summary)
        (out / 'experiment_record.md').write_text(f"# Known-temperature {cfg['label']},seed{cfg['seed']}\n\n- Date:{now()};IFpaper_v1.\n- Purpose:H1 versus manuscript baselines under the same measured future-temperature condition.\n- Dataset:{host['dataset']};../../configs/{key}.json,../../dataset/README.md;222/52/60windows,15min.\n- Operation:frozen backbone snapshot plus common30-parameter known-T context projection;no future response input.\n- Result:selected epoch{best_epoch};test RMSE{metrics['test']['rmse_mm']:.9f}mm;GPU{host['physical_gpu']}.\n- Products:initial/best/last.pt,history.csv,summary.json,parameters.json,validation/test predictions and daily errors.\n- Original frozen model code/weights and manuscript unchanged.\n", encoding='utf-8')
        print('COMPLETED', summary, flush=True)
    except BaseException as error:
        dump(out / 'failed.json', dict(status='failed', model=cfg['model'], seed=cfg['seed'], date_beijing=now(), exception_type=type(error).__name__, message=str(error)))
        (out / 'failed_experiment_record.md').write_text(f"# Known-temperature baseline failure:{key}\n\n- Date:{now()};IFpaper_v1.\n- Dataset/config:{host['dataset']};config.json.\n- Operation:baseline conditional training on GPU{host['physical_gpu']}.\n- Result:{type(error).__name__}:{error};no automatic retry.\n- Products:partial files here;full traceback at ../../logs/{key}.log.\n", encoding='utf-8')
        raise
if __name__ == '__main__':
    main()

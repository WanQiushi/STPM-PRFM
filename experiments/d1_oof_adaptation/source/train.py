"""Fit existing H1 parameters on fixed causal temperature forecasts; select by validation."""
from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import CachedWindows, build_cache, dump, response_metrics
from va_model import UnifiedVA1
ROOT, OUT = map(Path, sys.argv[1:3])
STAGE = sys.argv[3]
CFG = _release_json_loads((OUT / 'settings.json').read_text(encoding='utf-8'))
NORMAL = _release_json_loads((OUT / 'normalization.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(OUT / 'frozen_project' / CFG['h1_source']))
sys.path.insert(0, str(OUT / 'frozen_project' / CFG['temperature_source']))
from hybrid_model import HybridKqsz
from models import ShapeODEResidual
torch.set_num_threads(CFG['threads'])
DEVICE = torch.device('cuda:0')

def load_model():
    hstate = torch.load(OUT / 'h1_original.pt', map_location=DEVICE, weights_only=False)
    tstate = torch.load(OUT / 'v11_original.pt', map_location=DEVICE, weights_only=False)
    h1 = HybridKqsz(hstate['config']).to(DEVICE)
    h1.load_state_dict(hstate['model'])
    h1.configure_stage(CFG['h1_stage'])
    v11 = ShapeODEResidual(tstate['config']).to(DEVICE)
    v11.load_state_dict(tstate['model'])
    model = UnifiedVA1(v11, h1, tstate['mean'], tstate['std'], hstate['mean'], hstate['std']).to(DEVICE)
    return model

def loader(split, shuffle=False):
    return DataLoader(CachedWindows(OUT / 'cache' / (split + '.npz')), batch_size=CFG['batch_size'], shuffle=shuffle, num_workers=0)

def on_device(batch):
    return {key: value.to(DEVICE) for key, value in batch.items()}

def forward(model, batch):
    return model.forward_cached(batch['history'], batch['forecast_temperature_celsius'], batch['origin'])

def evaluate(model, data_loader):
    model.eval()
    saved = {}
    with torch.no_grad():
        for batch in data_loader:
            result = forward(model, on_device(batch))
            arrays = {key: result[key].cpu().numpy() for key in ['prediction', 'prediction_mm', 'lf_prediction', 'mf_prediction', 'temperature_lf', 'temperature_mf', 'effective_temperature_mf', 'reversible_prediction', 'memory_prediction', 'bw_latent', 'bw_physics_rate', 'bw_residual_rate', 'lf_anchor_offset']}
            arrays.update({key: batch[key].numpy() for key in ['target', 'mask', 'origin', 'forecast_temperature_celsius']})
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(value, axis=0) for key, value in saved.items()}
    return (response_metrics(arrays, NORMAL), arrays)

def save_predictions(path, arrays):
    np.savez_compressed(path, **arrays, mean=np.asarray(NORMAL['h1_mean']), std=np.asarray(NORMAL['h1_std']), query_offsets_minutes=np.arange(0, 1440, CFG['sample_minutes']), units='original_H1_normalized', prediction_mm_units='mm', forecast_temperature_units='celsius')

def save_checkpoint(model, epoch, validation):
    torch.save(dict(version=CFG['version'], model=model.state_dict(), epoch=epoch, validation=validation, settings=CFG, normalization=NORMAL, h1_original='h1_original.pt', v11_original='v11_original.pt'), OUT / 'best_va1.pt')

def parameter_report(model):
    return dict(v11_total=sum((p.numel() for p in model.v11.parameters())), v11_trainable=sum((p.numel() for p in model.v11.parameters() if p.requires_grad)), h1_total=sum((p.numel() for p in model.h1.parameters())), h1_trainable=sum((p.numel() for p in model.h1.parameters() if p.requires_grad)), h1_trainable_names=[name for name, p in model.h1.named_parameters() if p.requires_grad], h1_fixed_names=[name for name, p in model.h1.named_parameters() if not p.requires_grad], independent_displacement_head=False)

def fit():
    random.seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    model = load_model()
    dump(OUT / 'parameters.json', parameter_report(model))
    train_loader = loader('train', shuffle=True)
    val_loader = loader('validation')
    optimizer = torch.optim.Adam([p for p in model.h1.parameters() if p.requires_grad], lr=CFG['lr'])
    baseline, arrays = evaluate(model, val_loader)
    save_predictions(OUT / 'original_validation.npz', arrays)
    dump(OUT / 'original_validation.json', baseline)
    save_checkpoint(model, 0, baseline)
    best, best_epoch, bad = (baseline['pooled_rmse_mm'], 0, 0)
    started = time.time()
    rows = [dict(epoch=0, train_measured_mse=None, train_ode_residual_penalty=None, validation_pooled_rmse_mm=best, selected=True, seconds=0.0, **{f'validation_rmse_{c + 1}_mm': value for c, value in enumerate(baseline['rmse_mm'])})]
    pd.DataFrame(rows).to_csv(OUT / 'history.csv', index=False)
    dump(OUT / 'training_status.json', dict(status='running', epoch=0, best_epoch=0, best_validation_rmse_mm=best))
    print('BASELINE ' + json.dumps(dict(validation=baseline, parameters=parameter_report(model))), flush=True)
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        train_sse, train_count, energy_sum, windows = (0.0, 0, 0.0, 0)
        for index, batch in enumerate(train_loader):
            moved = on_device(batch)
            result = forward(model, moved)
            sse = ((result['prediction'] - moved['target']).square() * moved['mask']).sum()
            count = moved['mask'].sum()
            loss = sse / count + CFG['ode_weight'] * result['ode_residual_penalty']
            optimizer.zero_grad()
            loss.backward()
            if epoch == 1 and index == 0:
                gradients = {name: float(p.grad.norm().item()) for name, p in model.h1.named_parameters() if p.grad is not None}
                dump(OUT / 'first_training_batch_gradients.json', dict(origin=batch['origin'].tolist(), loss=float(loss.item()), h1_gradient_norms=gradients, v11_parameters_with_gradient=[name for name, p in model.v11.named_parameters() if p.grad is not None], explanation='Recorded from the actual first optimizer batch, before clipping; not an extra test batch.'))
                print('FIRST_BACKWARD ' + json.dumps(dict(epoch=epoch, measured_nodes=int(count.item()), loss=float(loss.item()), h1_parameters_with_gradient=len(gradients), v11_parameters_with_gradient=sum((p.grad is not None for p in model.v11.parameters())))), flush=True)
            torch.nn.utils.clip_grad_norm_(model.h1.parameters(), CFG['grad_clip'])
            optimizer.step()
            train_sse += sse.item()
            train_count += count.item()
            energy_sum += result['ode_residual_penalty'].item() * len(batch['origin'])
            windows += len(batch['origin'])
        validation, arrays = evaluate(model, val_loader)
        selected = validation['pooled_rmse_mm'] < best
        if selected:
            best, best_epoch, bad = (validation['pooled_rmse_mm'], epoch, 0)
            save_checkpoint(model, epoch, validation)
        else:
            bad += 1
        row = dict(epoch=epoch, train_measured_mse=train_sse / train_count, train_ode_residual_penalty=energy_sum / windows, validation_pooled_rmse_mm=validation['pooled_rmse_mm'], selected=selected, seconds=time.time() - started, **{f'validation_rmse_{c + 1}_mm': value for c, value in enumerate(validation['rmse_mm'])})
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / 'history.csv', index=False)
        dump(OUT / 'training_status.json', dict(status='running', epoch=epoch, best_epoch=best_epoch, best_validation_rmse_mm=best, epochs_without_improvement=bad))
        print('EPOCH ' + json.dumps(dict(**row, best_epoch=best_epoch, best_validation_rmse_mm=best)), flush=True)
        if bad >= CFG['patience']:
            break
    selected = torch.load(OUT / 'best_va1.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(selected['model'])
    validation, arrays = evaluate(model, val_loader)
    save_predictions(OUT / 'va1_validation.npz', arrays)
    dump(OUT / 'va1_validation.json', validation)
    summary = dict(status='completed', best_epoch=best_epoch, epochs_run=rows[-1]['epoch'], original_validation=baseline, va1_validation=validation, parameters=parameter_report(model), elapsed_seconds=time.time() - started)
    original_v11 = torch.load(OUT / 'v11_original.pt', map_location=DEVICE, weights_only=False)['model']
    summary['frozen_v11_weights_unchanged'] = all((torch.equal(value, original_v11[name]) for name, value in model.v11.state_dict().items()))
    dump(OUT / 'fit_summary.json', summary)
    dump(OUT / 'training_status.json', dict(status='completed', epoch=rows[-1]['epoch'], best_epoch=best_epoch, best_validation_rmse_mm=best))
    print('FIT_COMPLETED ' + json.dumps(summary), flush=True)

def finish():
    data_report = _release_json_loads((OUT / 'data_report.json').read_text(encoding='utf-8'))
    data_report['splits']['test'] = build_cache(ROOT, OUT, CFG, 'test')
    data_report['test_targets_cached'] = True
    dump(OUT / 'data_report.json', data_report)
    model = load_model()
    test_loader = loader('test')
    original, arrays = evaluate(model, test_loader)
    save_predictions(OUT / 'original_test.npz', arrays)
    dump(OUT / 'original_test.json', original)
    selected = torch.load(OUT / 'best_va1.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(selected['model'])
    adapted, arrays = evaluate(model, test_loader)
    save_predictions(OUT / 'va1_test.npz', arrays)
    dump(OUT / 'va1_test.json', adapted)
    fit_summary = _release_json_loads((OUT / 'fit_summary.json').read_text(encoding='utf-8'))
    comparison = dict(version=CFG['version'], best_epoch=selected['epoch'], original_validation=fit_summary['original_validation'], va1_validation=fit_summary['va1_validation'], original_test=original, va1_test=adapted, test_rmse_reduction_mm=original['pooled_rmse_mm'] - adapted['pooled_rmse_mm'], frozen_temperature=True, same_forecasts_in_both_arms=True, test_used_for_model_selection=False)
    dump(OUT / 'comparison.json', comparison)
    print('COMPLETED ' + json.dumps(comparison), flush=True)
if STAGE == 'fit':
    fit()
elif STAGE == 'finish':
    finish()

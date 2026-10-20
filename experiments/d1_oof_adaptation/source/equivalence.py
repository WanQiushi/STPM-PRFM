"""Compare original and component-view H1 on one actual cached training window."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from va_model import UnifiedVA1

def tensor_leaves(value, prefix=''):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(tensor_leaves(item, key + '.'))
        elif torch.is_tensor(item):
            result[key] = item
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--h1-checkpoint', required=True)
    parser.add_argument('--v11-checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--history-key', default='history')
    parser.add_argument('--temperature-key', default='forecast_temperature_celsius')
    parser.add_argument('--origin-key', default='origin')
    args = parser.parse_args()
    project = Path(args.project)
    sys.path.insert(0, str(project / 'experiments' / 'v2_hybrid_kqsz_20261001'))
    sys.path.insert(0, str(project / 'temperature_prediction' / 'shape_ode_residual_temperature_v11_20261002'))
    from hybrid_model import HybridKqsz
    from models import ShapeODEResidual
    device = torch.device(args.device)
    hstate = torch.load(args.h1_checkpoint, map_location=device, weights_only=False)
    tstate = torch.load(args.v11_checkpoint, map_location=device, weights_only=False)
    original = HybridKqsz(hstate['config']).to(device)
    original.load_state_dict(hstate['model'])
    original.configure_stage('forecast_adaptation')
    original.eval()
    h1 = copy.deepcopy(original)
    v11 = ShapeODEResidual(tstate['config']).to(device)
    v11.load_state_dict(tstate['model'])
    model = UnifiedVA1(v11, h1, tstate['mean'], tstate['std'], hstate['mean'], hstate['std']).to(device)
    model.eval()
    with np.load(args.cache, allow_pickle=False) as data:
        section = slice(args.index, args.index + 1)
        history = torch.as_tensor(data[args.history_key][section], dtype=torch.float32, device=device)
        forecast = torch.as_tensor(data[args.temperature_key][section], dtype=torch.float32, device=device)
        origin = torch.as_tensor(data[args.origin_key][section], dtype=torch.long, device=device)
    hmean = torch.as_tensor(hstate['mean'], dtype=torch.float32, device=device).reshape(1, 1, 5)
    hstd = torch.as_tensor(hstate['std'], dtype=torch.float32, device=device).reshape(1, 1, 5)
    with torch.no_grad():
        before = original(history, (forecast - hmean[..., :1]) / hstd[..., :1], origin)
        after = model.forward_cached(history, forecast, origin)
        before_mm = before['prediction'] * hstd[..., 1:] + hmean[..., 1:]
    old_outputs = tensor_leaves(before)
    new_outputs = tensor_leaves(after)
    comparisons = {name: {'max_abs_difference': (item - new_outputs[name]).abs().max().item(), 'exactly_equal': torch.equal(item, new_outputs[name])} for name, item in old_outputs.items()}
    comparisons['prediction_mm'] = {'max_abs_difference': (before_mm - after['prediction_mm']).abs().max().item(), 'exactly_equal': torch.equal(before_mm, after['prediction_mm'])}
    old_parameters = dict(original.named_parameters())
    new_parameters = dict(model.h1.named_parameters())
    neural_parameters = dict(model.neural_parts.named_parameters())
    temperature_parameters = dict(model.temperature_parts.named_parameters())
    h1_ids = {id(parameter) for parameter in new_parameters.values()}
    v11_ids = {id(parameter) for parameter in model.v11.parameters()}
    report = {'purpose': 'same-input same-weight forward equivalence and parameter ownership', 'project': str(project), 'h1_checkpoint': args.h1_checkpoint, 'v11_checkpoint': args.v11_checkpoint, 'cache': args.cache, 'window_index': args.index, 'origin': origin.cpu().tolist(), 'history_shape': list(history.shape), 'forecast_celsius_shape': list(forecast.shape), 'device': str(device), 'outputs': comparisons, 'all_outputs_exactly_equal': all((item['exactly_equal'] for item in comparisons.values())), 'h1_checkpoint_keys_unchanged': list(original.state_dict()) == list(model.h1.state_dict()), 'v11_checkpoint_keys_unchanged': list(tstate['model']) == list(model.v11.state_dict()), 'h1_parameter_names_unchanged': list(old_parameters) == list(new_parameters), 'h1_requires_grad_unchanged': {name: parameter.requires_grad == new_parameters[name].requires_grad for name, parameter in old_parameters.items()}, 'neural_components_reference_h1_modules': {name: getattr(model.neural_parts, name) is getattr(model.h1, name) for name in ('initial_z', 'lf_residual', 'residual_field')}, 'neural_parameters_owned_by_h1': all((id(parameter) in h1_ids for parameter in neural_parameters.values())), 'temperature_components_reference_v11_modules': {'output_residual': model.temperature_parts.output_residual is model.v11.residual}, 'temperature_components_parameters_owned_by_v11': all((id(parameter) in v11_ids for parameter in temperature_parameters.values())), 'temperature_components_all_parameters_frozen': all((not parameter.requires_grad for parameter in temperature_parameters.values())), 'v11_all_parameters_frozen': all((not parameter.requires_grad for parameter in model.v11.parameters())), 'parameters': {'h1_total': sum((parameter.numel() for parameter in new_parameters.values())), 'original_h1_trainable': sum((parameter.numel() for parameter in old_parameters.values() if parameter.requires_grad)), 'wrapped_h1_trainable': sum((parameter.numel() for parameter in new_parameters.values() if parameter.requires_grad)), 'neural_components_total': sum((parameter.numel() for parameter in neural_parameters.values())), 'neural_components_trainable': sum((parameter.numel() for parameter in neural_parameters.values() if parameter.requires_grad)), 'temperature_components_total': sum((parameter.numel() for parameter in temperature_parameters.values())), 'temperature_components_frozen': sum((parameter.numel() for parameter in temperature_parameters.values() if not parameter.requires_grad)), 'temperature_components_trainable': sum((parameter.numel() for parameter in temperature_parameters.values() if parameter.requires_grad)), 'v11_total': sum((parameter.numel() for parameter in model.v11.parameters())), 'v11_trainable': sum((parameter.numel() for parameter in model.v11.parameters() if parameter.requires_grad)), 'unified_trainable': sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))}, 'h1_trainable_parameter_names': [name for name, parameter in new_parameters.items() if parameter.requires_grad], 'neural_parameter_names': list(neural_parameters), 'temperature_component_parameter_names': list(temperature_parameters)}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    assert report['all_outputs_exactly_equal'], 'Same-input same-weight H1 forward equivalence failed'
if __name__ == '__main__':
    main()

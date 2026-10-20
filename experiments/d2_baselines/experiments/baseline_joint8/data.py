"""One-minute past temperature and seven synthetic deflections -> joint eight-channel forecast."""
from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = PROJECT / 'data/protocol.json'
ORIGIN_SOURCE = PROJECT / 'data/baseline_window_origins_499.json'
CHANNELS = ['SIM_T', 'SIM_D01', 'SIM_D02', 'SIM_D03', 'SIM_D04', 'SIM_D05', 'SIM_D06', 'SIM_D07']
UNITS = ['degC', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm']
DEFAULT_CONFIG = dict(data='data/minute_synthetic.npz', split_days=[280, 60, 60], sample_minutes=1, history_days=30, horizon_days=1, train_stride=720, validation_stride=1440, test_stride=1440, min_training_observed_points=1080)

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def read_protocol():
    return _release_json_loads(PROTOCOL.read_text(encoding='utf-8'))

def load_data(cfg):
    with np.load(PROJECT / cfg['data']) as source:
        values = source['values'].astype(np.float64)
        observed = source['valid_observed'].astype(bool)
        available = source['input_available_at'].copy()
    train_end = cfg['split_days'][0] * 1440
    measured = np.where(observed[:train_end], values[:train_end], np.nan)
    mean, std = (np.nanmean(measured, axis=0), np.nanstd(measured, axis=0))
    ho = np.arange(-cfg['history_days'] * 1440, 0, cfg['sample_minutes'])
    fo = np.arange(0, cfg['horizon_days'] * 1440, cfg['sample_minutes'])
    return (values, observed, available, mean, std, ho, fo)

def training_difference_scales(values, observed, train_end, std):
    """Each chronological observed minute pair/triple is counted once."""
    y, mask = (values[:train_end], observed[:train_end])
    d1, d2 = (np.diff(y, axis=0), np.diff(y, n=2, axis=0))
    m1, m2 = (mask[:-1] & mask[1:], mask[:-2] & mask[1:-1] & mask[2:])
    n1, n2 = (m1.sum(axis=0), m2.sum(axis=0))
    r1 = np.sqrt(np.where(m1, d1 * d1, 0).sum(axis=0) / n1)
    r2 = np.sqrt(np.where(m2, d2 * d2, 0).sum(axis=0) / n2)
    return dict(channels=CHANNELS, units=UNITS, interval_minutes=1, source_interval=[0, int(train_end)], first_difference_count=n1.tolist(), second_difference_count=n2.tolist(), first_difference_rms_raw_units=r1.tolist(), second_difference_rms_raw_units=r2.tolist(), first_difference_rms_normalized=(r1 / std).tolist(), second_difference_rms_normalized=(r2 / std).tolist(), use='Training chronological one-minute grid only; observed pair/triple counted once. Derivative loss weights remain zero.')

def make_protocol(cfg):
    values, observed, available, mean, std, ho, fo = load_data(cfg)
    public = _release_json_loads(ORIGIN_SOURCE.read_text(encoding='utf-8'))
    previous = dict(origins={'train': public['origins']['train'], 'val': public['origins']['validation'], 'test': public['origins']['test']}, mean=mean.tolist(), std=std.tolist())
    train_end = cfg['split_days'][0] * 1440
    candidates = previous['origins']['train']
    selected = []
    causal_candidates = 0
    for origin in candidates:
        if not np.all(available[origin + ho] < origin):
            continue
        causal_candidates += 1
        if not np.all(observed[origin + fo].sum(axis=0) >= cfg['min_training_observed_points']):
            continue
        selected.append(origin)
    origins = dict(train=selected, val=previous['origins']['val'], test=previous['origins']['test'])
    counts = {}
    for split, starts in origins.items():
        for origin in starts:
            if not np.all(available[origin + ho] < origin):
                raise ValueError(f'Non-causal eight-channel minute history at origin {origin}')
        target_counts = observed[np.asarray(starts)[:, None] + fo].sum(axis=1)
        counts[split] = dict(selected=len(starts), causal_history=len(starts), measured_targets=target_counts.sum(axis=0).tolist(), minimum_observed_per_channel=target_counts.min(axis=0).tolist())
    retained = sorted(set(selected) & set(previous['origins']['train']))
    added = sorted(set(selected) - set(previous['origins']['train']))
    removed = sorted(set(previous['origins']['train']) - set(selected))
    split_ends = (np.cumsum(cfg['split_days']) * 1440).tolist()
    protocol = dict(config=cfg, channels=CHANNELS, units=UNITS, source_path=str(PROJECT / cfg['data']), source_shape=list(values.shape), source_period='400 synthetic days, fictional dates from2020-01-01', source_arrays=['values', 'valid_observed', 'input_available_at'], source_arrays_modified=False, history_points=len(ho), future_points=len(fo), origins=origins, counts=counts, mean=mean.tolist(), std=std.tolist(), training_observed_counts=observed[:train_end].sum(axis=0).tolist(), training_difference_scales=training_difference_scales(values, observed, train_end, std), split_time_bounds_minutes=dict(train=[0, split_ends[0]], val=[split_ends[0], split_ends[1]], test=[split_ends[1], split_ends[2]], endpoint_convention='Half-open intervals; full future window stays within its split.'), origin_source=str(ORIGIN_SOURCE), origin_policy='Rebuild training origins for all eight minute channels; freeze the previous 60 validation and60 test origins.', training_selection=dict(candidate_grid_minutes=cfg['train_stride'], candidates=len(candidates), causal_candidates=causal_candidates, selected=len(selected), min_observed_per_channel=cfg['min_training_observed_points'], observed_target_fraction=0.75, retained_reference_train=len(retained), added=len(added), removed=len(removed), added_origins=added, removed_origins=removed, target_end_within_training=True, full_future_temperature_required=False), reference_contract=dict(validation_origins_equal=origins['val'] == previous['origins']['val'], test_origins_equal=origins['test'] == previous['origins']['test'], displacement_mean_equal=bool(np.array_equal(mean[1:], previous['mean'][1:])), displacement_std_equal=bool(np.array_equal(std[1:], previous['std'][1:]))), legacy_origin_selection='Not used for the synthetic experiment; all splits use the 499/60/60 origins without filtering by future temperature values.', task='past 30-day historical temperature plus seven synthetic deflections -> next 24-hour joint eight-channel forecast', input_policy='Only past eight-channel history enters model forward. No future measured/predicted temperature, calendar marks, or future displacement input.', causal_history='All 43200 consumed timestamps and all eight channels have input_available_at < origin.', target_policy='Future eight-channel valid_observed masks are used for training-origin label coverage, losses, and scores only. Filled future values are not supervised.', normalization='Eight-channel mean/std from valid_observed values in the first 280 days only.', score_units='Temperature RMSE in deg C is separate from seven-deflection pooled RMSE in mm; do not pool incompatible physical units.', training_objective='Masked normalized eight-channel MSE; equal weighting per valid normalized entry, no physical-unit mixing. Derivative loss weights remain zero.', selection='Joint eight-channel normalized validation RMSE (dimensionless) selects checkpoints; temperature deg C and displacement mm are reported separately. Test evaluated after training, never used to tune configuration.')
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    dump(PROTOCOL, protocol)
    print('PROTOCOL ' + json.dumps(dict(counts=counts, training_selection=protocol['training_selection'], reference_contract=protocol['reference_contract'])), flush=True)
    return protocol

def prepare(cfg):
    values, observed, _, mean, std, ho, fo = load_data(cfg)
    protocol = read_protocol()
    normalized = ((values - mean) / std).astype('float32')
    return (normalized, observed, protocol['origins'], mean, std, protocol['counts'], ho, fo)

class Windows(Dataset):

    def __init__(self, values, observed, origins, history_offsets, future_offsets):
        self.values, self.observed, self.origins = (values, observed, origins)
        self.ho, self.fo = (history_offsets, future_offsets)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        h, f = (origin + self.ho, origin + self.fo)
        return dict(history=self.values[h].copy(), history_mask=self.observed[h].copy(), target=self.values[f].copy(), mask=self.observed[f].copy(), origin=np.int64(origin))

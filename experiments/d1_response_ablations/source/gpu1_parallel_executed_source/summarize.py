"""Keep original table rows and append the three approved new ablations."""
from if_release.runtime import expand as _release_expand, loads as _release_json_loads
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = Path(_release_expand('${WORKSPACE}'))
OUT = ROOT / 'outputs' / HERE.name
OLD = ROOT / 'outputs/d1_full219_frozen_inference_20261017'
ARMS = ['S0', 'S2', 'S3', 'S4', 'S5', 'S6', 'no_lf_anchor', 'mean_split', 'no_neural_memory']
NAMES = ['完整模型', '去掉温度残差修正', '去掉趋势监督', '去掉热滞后', '去掉Bouc–Wen物理速率项', '去掉预测温度适配', '去掉LF历史水平锚定', '历史均值替代时变分频表示', '去掉神经记忆残差']
CORE = ['no_lf_anchor', 'mean_split', 'S4', 'S5', 'no_neural_memory']

def main():
    collected = {}
    table = []
    for arm, name in zip(ARMS, NAMES):
        rows = []
        daily = []
        for seed in range(2024, 2029):
            folder = (OLD if arm.startswith('S') else OUT) / f'{arm}_seed{seed}'
            with np.load(folder / 'test_predictions.npz') as a:
                e = (a['prediction'].astype('float64') - a['target'].astype('float64')) * a['std']
                m = a['mask']
                origin = a['origin'].copy()
                sse = np.where(m[:, :, 1:], e[:, :, 1:] ** 2, 0).sum(1)
                sae = np.where(m[:, :, 1:], np.abs(e[:, :, 1:]), 0).sum(1)
                n = m[:, :, 1:].sum(1)
            record = _release_json_loads((folder / 'metrics.json').read_text())
            metrics = record['metrics']['test']['all_days'] if arm.startswith('S') else record['metrics']['test']
            rows.append(metrics)
            daily.append(dict(sse=sse, sae=sae, n=n))
        arrays = {k: np.stack([d[k] for d in daily]) for k in ['sse', 'sae', 'n']}
        collected[arm] = arrays
        np.savez_compressed(OUT / 'analysis' / (arm + '_daily_errors.npz'), **arrays, origin=origin)
        rmse = np.array([r['response_rmse_mm'] for r in rows])
        mae = np.array([r['response_mae_mm'] for r in rows])
        table.append(dict(arm=arm, name=name, rmse_mean=float(rmse.mean()), rmse_sd=float(rmse.std(ddof=1)), mae_mean=float(mae.mean()), per_seed_rmse=rmse.tolist()))
    full = table[0]['rmse_mean']
    rng = np.random.default_rng(20261020)
    intervals = {arm: {} for arm in ARMS[1:]}
    D = len(origin)
    B = 10000
    for length in [3, 7, 14]:
        starts = rng.integers(0, D - length + 1, size=(B, int(np.ceil(D / length))))
        ids = (starts[:, :, None] + np.arange(length)).reshape(B, -1)[:, :D]
        boots = {}
        for arm, a in collected.items():
            s = a['sse'].sum(2)
            n = a['n'].sum(2)
            boots[arm] = np.sqrt(s[:, ids].sum(2) / n[:, ids].sum(2)).mean(0)
        for arm in ARMS[1:]:
            delta = boots[arm] - boots['S0']
            report = dict(ci95=np.quantile(delta, [0.025, 0.975]).tolist())
            if arm in CORE:
                report['five_core_comparisons_ci99'] = np.quantile(delta, [0.005, 0.995]).tolist()
            intervals[arm][str(length)] = report
    effects = []
    for row in table:
        row['increase_percent'] = 100 * (row['rmse_mean'] / full - 1)
        if row['arm'] == 'S0':
            continue
        a = collected[row['arm']]
        f = collected['S0']
        diff = (a['sse'].sum(2) / a['n'].sum(2) - f['sse'].sum(2) / f['n'].sum(2)).mean(0)
        effect = dict(arm=row['arm'], delta_rmse_mm=row['rmse_mean'] - full, full_win_days=int((diff > 0).sum()), ties=int((diff == 0).sum()), full_win_temperature_seeds=int((np.array(row['per_seed_rmse']) > table[0]['per_seed_rmse']).sum()), intervals=intervals[row['arm']])
        effects.append(effect)
    report = dict(rows=table, effects=effects, bootstrap_repetitions=B, random_seed=20261020, interval_scope='conditional on fitted response models and five temperature seeds', original_rows='copied from existing original metrics without changing old results', response_repetitions=1, temperature_repetitions=5)
    (OUT / 'analysis/results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ['# D1完整预测管线：原结果与新增消融', '', '同一STPM预测温度，109测试日，四位置，一分钟评价。五温度seed共享各结构的一份H1；不是五次独立响应训练。', '', '|模型配置|RMSE (mm)|相对完整模型增加 (%)|', '|---|---:|---:|']
    for r in table:
        change = '—' if r['arm'] == 'S0' else f"{r['increase_percent']:.2f}"
        lines.append(f"|{r['name']}|{r['rmse_mean']:.3f} ± {r['rmse_sd']:.3f}|{change}|")
    lines += ['', '## 逐日配对稳定性', '', '|配置|ΔRMSE (mm)|7日块95%区间|完整组胜日数|', '|---|---:|---|---:|']
    for e in effects:
        ci = e['intervals']['7']['ci95']
        lines.append(f"|{e['arm']}|{e['delta_rmse_mm']:+.3f}|[{ci[0]:+.3f}, {ci[1]:+.3f}]|{e['full_win_days']}/109|")
    lines += ['', '完整JSON包含3/7/14日历块区间及五项核心比较校正区间。幅度/误差差异不是物理状态识别证据。']
    (OUT / 'results_table.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / 'completed_experiment_record.md').write_text(f"# Predicted-temperature ablations completed\n\n- Date: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}; IFpaper_v1.\n- Purpose: three approved response removals, same original staged training and causal OOF adaptation.\n- Results: see results_table.md, analysis/results.json; original six table rows reused.\n- Products: {OUT}; source/configs and individual training/evaluation records retained.\n")
    print('SUMMARY_COMPLETED ' + str(OUT), flush=True)
if __name__ == '__main__':
    (OUT / 'analysis').mkdir(exist_ok=True)
    main()

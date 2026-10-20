"""Summarize minute-grid joint forecasts with separate temperature and displacement units."""
from if_release.runtime import loads as _release_json_loads
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CHANNELS = ['Temperature', 'Pier1', 'Pier2', 'Pier3', 'Pier4']

def curve_metrics(path):
    with np.load(path) as data:
        mask = data['mask'].astype(bool)
        prediction = data['prediction'].astype('float64') * data['std']
        target = data['target'].astype('float64') * data['std']

    def rms(values, valid):
        return float(np.sqrt((values * values * valid).sum() / valid.sum()))
    result = {}
    for selection, unit in [(slice(1, 5), 'mm'), (slice(0, 1), 'c')]:
        valid = mask[:, :, selection]
        p, y = (prediction[:, :, selection], target[:, :, selection])
        pair = valid[:, 1:] & valid[:, :-1]
        triple = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        prefix = 'temperature_' if unit == 'c' else ''
        result.update({prefix + key + '_' + unit: value for key, value in dict(step_rms=rms(np.diff(p, axis=1), pair), curvature_rms=rms(np.diff(p, n=2, axis=1), triple), measured_step_rms=rms(np.diff(y, axis=1), pair), measured_curvature_rms=rms(np.diff(y, n=2, axis=1), triple), derivative_error_rms=rms(np.diff(p - y, axis=1), pair)).items()})
    return result

def main():
    plan = _release_json_loads((HERE / 'plan.json').read_text())
    out = PROJECT / plan['output']
    names = sys.argv[1:]
    rows, examples, labels, comparability = ([], {}, {}, {})
    example_index = plan['example_window_index']
    reference = None
    for name in names:
        folder = out / name
        cfg = _release_json_loads((folder / 'config.json').read_text())
        summary = _release_json_loads((folder / 'summary.json').read_text())
        accuracy = _release_json_loads((folder / 'suite_results.json').read_text())['results']['accuracy']
        job = _release_json_loads((folder / 'job_status.json').read_text())
        row = dict(run=name, model=cfg['model'], scientific_label=summary['scientific_label'], seed=cfg['seed'], parameters=summary['parameters'], best_epoch=summary['best_accuracy_epoch'], epochs_ran=summary['epochs_ran'], training_seconds=summary['training_seconds'], train_and_export_seconds=summary['elapsed_seconds'], process_wall_seconds=job['wall_seconds'], peak_cuda_allocated_mib=summary['peak_cuda_allocated_mib'])
        for split in ['validation', 'test']:
            score = accuracy[split]
            row.update({split + '_' + key: score[key] for key in ['temperature_rmse_c', 'temperature_mae_c', 'joint_standardized_rmse']})
            row[split + '_rmse_mm'] = score['pooled_rmse_mm']
            row[split + '_tail_2h_rmse_mm'] = score['tail_2h_pooled_rmse_mm']
            row[split + '_temperature_tail_2h_rmse_c'] = score['temperature_tail_2h_rmse_c']
            row[split + '_peak_mae_mm'] = float(np.mean(score['peak_mae_mm']))
            row[split + '_trough_mae_mm'] = float(np.mean(score['trough_mae_mm']))
            for channel, value in zip(CHANNELS[1:], score['rmse_mm']):
                row[split + '_' + channel + '_rmse_mm'] = value
            for channel, value in zip(CHANNELS[1:], score['mae_mm']):
                row[split + '_' + channel + '_mae_mm'] = value
            row.update({split + '_1min_' + key: value for key, value in curve_metrics(folder / (split + '_predictions_accuracy.npz')).items()})
        with np.load(folder / 'test_predictions_accuracy.npz') as data:
            if reference is None:
                reference = {key: data[key].copy() for key in ['origin', 'target', 'mask', 'mean', 'std']}
            comparability[name] = {key + '_equal': bool(np.array_equal(reference[key], data[key])) for key in ['origin', 'target', 'mask', 'mean', 'std']}
            if cfg['seed'] == 2024:
                examples[cfg['model']] = data['prediction'][example_index] * data['std'] + data['mean']
                labels[cfg['model']] = summary['scientific_label']
        rows.append(row)
    table = pd.DataFrame(rows, columns=list(rows[0]) if rows else ['run', 'model', 'seed'])
    table.to_csv(out / 'comparison.csv', index=False, encoding='utf-8-sig')
    statistics = []
    statistic_fields = ['parameters', 'epochs_ran', 'best_epoch', 'training_seconds', 'train_and_export_seconds', 'process_wall_seconds', 'peak_cuda_allocated_mib', 'validation_joint_standardized_rmse', 'test_joint_standardized_rmse', 'validation_rmse_mm', 'test_rmse_mm', 'validation_temperature_rmse_c', 'test_temperature_rmse_c', 'test_temperature_mae_c', 'test_peak_mae_mm', 'test_trough_mae_mm', 'test_tail_2h_rmse_mm', 'test_temperature_tail_2h_rmse_c', 'test_1min_step_rms_mm', 'test_1min_curvature_rms_mm', 'test_1min_derivative_error_rms_mm', 'test_1min_temperature_step_rms_c', 'test_1min_temperature_curvature_rms_c', 'test_1min_temperature_derivative_error_rms_c']
    for model in plan['models']:
        group = table[table['model'] == model]
        if len(group):
            statistics.append(dict(model=model, scientific_label=group['scientific_label'].iloc[0], n=len(group), expected_n=len(plan['seeds']), seeds=group['seed'].tolist(), **{key + '_mean': float(group[key].mean()) for key in statistic_fields}, **{key + '_sample_std': float(group[key].std(ddof=1)) if len(group) > 1 else None for key in statistic_fields}))
    seed_table = pd.DataFrame(statistics)
    seed_table.to_csv(out / 'five_seed_summary.csv', index=False, encoding='utf-8-sig')
    seed_table.to_csv(out / 'seed_summary.csv', index=False, encoding='utf-8-sig')
    queue = _release_json_loads((out / 'queue_status.json').read_text())
    failures = [job for job in queue['jobs'] if job['status'] == 'failed']
    result = dict(per_seed=rows, seed_statistics=statistics, comparability=comparability, completed_jobs=len(rows), failed_jobs=len(failures), failures=failures, planned_jobs=len(plan['configs']), new_training_jobs=len(plan['configs']), reused_jobs=[], seeds=plan['seeds'], sampling_minutes=1, split_windows=dict(train=150, val=26, test=39), history_points=43200, forecast_points=1440, channel_order=CHANNELS, h1_policy='Excluded; H1 belongs to separate temperature-driven work', input_policy='Past 30 days of temperature plus four displacements; no future values or calendar features', output_policy='Future 24 hours of temperature and four displacements, jointly forecast', selection_metric='Validation joint_standardized_rmse; pooled across five standardized channels', physical_metric_policy='Temperature in degrees C; pooled displacement RMSE in mm uses only four displacement channels', test_used_for_configuration_selection=False, time_policy='Observed shared-GPU wall times; not isolated-card speed measurements.', roughness_policy='Raw one-minute predictions; no postprocessing filter; differences computed separately by physical unit on the minute grid.')
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    text = f"# 一分钟温度与四位移联合预测ModernTCN五种子对照\n\n计划 {len(plan['configs'])} 个全新配置，成功汇总 {len(rows)} 组、训练失败 {len(failures)} 组；不复用旧实验。ModernTCN seed2024–2028；冻结150训练/26验证/39测试，标准化mask MSE。历史30天43200×5（Temperature温度＋四个位移），预测未来24小时1440×5；没有未来实测输入或日历输入，不混入H1。验证集五通道联合标准化RMSE选择检查点；℃和mm分别统计，不合成混单位的物理RMSE。各模型按成功种子数报告均值与样本标准差；失败不能计入成功五种子。\n\n|模型|成功/计划种子|参数量|验证位移 RMSE/mm|测试位移 RMSE/mm|验证温度 RMSE/℃|测试温度 RMSE/℃|完整进程时间/s|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    summary_by_model = {item['model']: item for item in statistics}
    for model in plan['models']:
        if model not in summary_by_model:
            text += f"|{model}|0/{len(plan['seeds'])}|—|—|—|—|—|—|\n"
            continue
        item = summary_by_model[model]
        values = []
        for key in ['validation_rmse_mm', 'test_rmse_mm', 'validation_temperature_rmse_c', 'test_temperature_rmse_c', 'process_wall_seconds']:
            mean = item[key + '_mean']
            deviation = item[key + '_sample_std']
            values.append(f'{mean:.4f} ± {deviation:.4f}' if deviation is not None else f'{mean:.4f}（单种子）')
        text += f"|{item['scientific_label']}|{item['n']}/{item['expected_n']}|{int(item['parameters_mean'])}|" + '|'.join(values) + '|\n'
    if failures:
        text += '\n未达到全部成功。失败配置及原始错误日志：\n\n'
        for job in failures:
            text += f"- {job['model']} seed{job['seed']}：返回码 {job['returncode']}；{job['log']}。\n"
    text += '\n每支座误差、位移峰谷、最后120分钟、原始一分钟间隔的差分和差分误差见CSV/JSON。粗糙度描述变化，不单独证明精度改善；与旧十五分钟四位移输入协议的数据覆盖、输入信息和指标不同，不能归因于采样间隔单项改动。时间包含单卡并发资源竞争，不是独占显卡速度排名。固定seed2024、预声明window index ' + str(example_index) + '绘制原始曲线，无预测后处理滤波；测试集没有用于配置选择。\n'
    (out / 'comparison.md').write_text(text, encoding='utf-8')
    families = [model for model in plan['models'] if model in examples]
    if families:
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
        figure, axes = plt.subplots(len(families), 5, figsize=(16, 2.1 * len(families)), layout='constrained', squeeze=False)
        truth = reference['target'][example_index] * reference['std'] + reference['mean']
        truth = np.where(reference['mask'][example_index], truth, np.nan)
        time_hours = np.arange(truth.shape[0]) / 60
        for row_index, model in enumerate(families):
            for channel, ax in enumerate(axes[row_index]):
                ax.plot(time_hours, truth[:, channel], color='#000000', lw=1.0, label='Measured')
                ax.plot(time_hours, examples[model][:, channel], color='#0072B2', lw=0.85, label='Forecast')
                ax.set_title(labels[model] + ' | ' + CHANNELS[channel], fontsize=8)
                ax.set_xlabel('Forecast time (h)')
                ax.set_ylabel('Temperature (deg C)' if channel == 0 else 'Displacement (mm)')
                ax.grid(alpha=0.2)
                ax.legend(frameon=False, fontsize=7)
        figure.suptitle('One-minute joint forecasts | seed2024 | fixed window ' + str(example_index), fontsize=11)
        figure.savefig(out / 'prediction_comparison.svg')
        figure.savefig(out / 'prediction_comparison.png', dpi=150)
        plt.close(figure)
    ended = datetime.now(timezone(timedelta(hours=8)))
    record = PROJECT / 'outputs/experiment_records' / (ended.strftime('%Y-%m-%d_%H%M%S') + '_一分钟温度四位移联合预测ModernTCN五种子汇总.md')
    record.write_text(text + '\n产物：' + str(out) + '；代码：' + str(HERE) + '。\n', encoding='utf-8')
    print('COMPARISON ' + json.dumps(statistics), flush=True)
if __name__ == '__main__':
    main()

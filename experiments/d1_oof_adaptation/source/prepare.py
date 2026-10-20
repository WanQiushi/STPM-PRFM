"""Rebuild train/validation histories from raw data; defer test targets until selection."""
from if_release.runtime import loads as _release_json_loads
import json
from pathlib import Path
import sys
from data import build_cache, dump
ROOT, OUT = map(Path, sys.argv[1:3])
CFG = _release_json_loads((OUT / 'settings.json').read_text(encoding='utf-8'))
(OUT / 'cache').mkdir()
report = dict(data=str(ROOT / CFG['data']), source_forecasts=CFG['forecast_source_output'], source_weights=CFG['original_source_output'], sample_minutes=CFG['sample_minutes'], history_days=CFG['history_days'], horizon_days=CFG['horizon_days'], test_targets_cached=False, splits={})
for split in ['train', 'validation']:
    report['splits'][split] = build_cache(ROOT, OUT, CFG, split)
    print('CACHE ' + json.dumps(dict(split=split, **report['splits'][split])), flush=True)
dump(OUT / 'data_report.json', report)
(OUT / 'cache' / 'README.md').write_text('# va-1 固定温度预报 / 完整H1训练窗口\n\n- 数据：mainbridge_temperature_v2/minute_complete.npz；实际路径与规模见../data_report.json。来源与原划分引用../README.md。\n- 构建：../frozen_source/prepare.py及data.py；测试缓存由train.py finish在验证选模完成后构建。\n- 原始五通道：Temperature温度℃、Pier1/Pier2/Pier3/Pier4位移mm。传感器是否气温未确认。\n- train/validation/test预期110/26/39窗；起点为2022-01-01起分钟，见../window_origins.json；原时间划分511/110/109天。\n- history[N,2880,5]为30天、15分钟采样、原H1五通道标准化；最后一点origin−15分钟。固定输入窗口只缓存原始历史，不缓存LF/MF/q/z。\n- forecast_temperature_celsius[N,96,1]为未来24小时15分钟预报；来自../temperature_cache；训练为过去-only OOF，验证/测试为固定最终v11。\n- target/mask[N,96,4]为原H1尺度位移标签和valid_observed实测掩码；补值只可作历史输入，不作未来位移监督。\n- *_temperature_diagnostics.npz保存真实温度、实测mask和折元数据，仅供诊断，不被训练DataLoader读取。\n- 尺度见../normalization.json；v11标准化→℃→H1温度标准化；位移反变换prediction*std[1:]+mean[1:]。\n- 输入读取：d=np.load("train.npz"); model.forward_cached(torch.tensor(d["history"]), torch.tensor(d["forecast_temperature_celsius"]), torch.tensor(d["origin"]))。\n- 输出与指标：预测/标签采用原H1标准化，独立prediction_mm为mm；选择指标为实测四位移pooled RMSE mm。\n- 折预测偏差/样本数见../data_report.json，因果producer边界见../temperature_cache/fold_*_provenance.json。\n- 关联../prepare_experiment_record.md及归档IFpaper_v1/2026-10-06_va-1设计与实现.md。\n', encoding='utf-8')
print('PREPARED', flush=True)

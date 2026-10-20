# va：全部剩余核心模块消融

2026-10-08，IFpaper_v1。用户明确“全部开始运行”，范围为已列A0—A7：A0固定主模型、A3已完成结果复用，新增A1/A2/A4/A5/A6/A7。B1/B2为第二轮补充，不在本次表内。主模型锁定清单va_model_manifest.json及既有checkpoint不修改。

## 数据与来源

- 原分钟数据 `${WORKSPACE}/data/mainbridge_temperature_v2/minute_complete.npz`，1051200×5，Temperature温度和Pier1/04-02/12-02/14-02四个位移，2022—2023，split_days=[511,110,109]。源数据及输入可用时间规则沿原目录README。
- 温度训练原304窗/验证66窗，7天分钟GRU10080步、30个日均/覆盖率、已知日历；未来1440分钟节点，只用实测mask监督；标准化仍只用训练期实测T。
- H1结构预训练同原68train/26val，15分钟网格，30天历史；窗口冻结在h1_pretraining_protocol.json。按原响应辨识阶段使用真实未来温度训练响应模块，不用于最终部署预测；之后仍用原因果OOF预测T适配。
- H1适配复用 `${WORKSPACE}/outputs/va_1_20261004/cache/{train,validation}.npz`，110train/26val，与主模型使用相同固定OOF预测T；归一化复用同目录normalization.json。结构修改后从原随机初始化重训预训练＋适配，未套完整H1权重后在推理时关闭模块。
- 最终评价复用 `temperature_prediction/outputs/va_2_2_sampling_20261008/datasets/h1_1min.npz`：26共同origin、1440分钟节点，每seed四位移实测133727、温度37426；history[26,43200,5]、未来T[26,1440,1]、未来Y[26,1440,4]。原标准化五通道history、目标4Y、实测mask、origin、mean/std、真实T/mask与query_offsets；数据集原README不改变。
- 温度预测℃用H1原T mean/std归一化一次后调用HybridKqsz(history,future_temperature,origin)。真实未来T只在响应辨识预训练及评分中使用，不作为适配或最终预测输入。未评价测试集。

## 单项变化

|组|变化|训练/初始化|
|---|---|---|
|A1|以同历史初态＋已知时间的共享读出13→32→1替代潜ODE，仍归一化形状；L/A、r_T与λ3保留|5个新无ODE点MSE初始化，再5个基础/残差训练；初始化阶段也无ODE|
|A2|L取历史锚点，A取上一日标准差，不学习L/A|复用同seed现有一分钟v9；5次重训，删除对应0.5水平/0.25幅度辅助项，形状0.25和λ3保留|
|A4|最终温度趋势λ3→0|复用同seed一分钟v9，5次原两阶段训练；独立源码明确将cfg.trend_weight传入损失，避免原固定常数3仍生效|
|A5|历史/未来q直接取u_MF，移除thermal_lag_raw；其余物理/残差保持|原15分钟H1预训练＋OOF适配重训后，同样迁移1分钟；接5套固定完整va温度|
|A6|只删BW物理演化，保留z初态/神经残差/输出sz，dz/dt=f_r；移除不再使用β/γ/季节a参数|原H1预训练＋OOF适配重训，随后迁移1分钟；接同5套温度|
|A7|完整va温度接适配前原H1|只评价，同原H1权重和1分钟FFTbuffer迁移；无新训练|

温度宽32、潜维8、残差隐层16、Adam0.001、batch32；初始化cap80、基础cap80、残差cap60，p12；基础后缓存并冻结基础。A2不再注册L/A头，但保持其随机初始化消耗以对齐其余共同网络初始化。A1替代读出仅用于消融，报告其参数/耗时差，不加入主模型。

H1结构重训使用原cfg：15轮BW预训练＋20轮prediction_warmup＋joint阶段，cap300、joint p60、Adam0.001、batch16；原分项损失和固定FFT截止保持；最佳accuracy checkpoint作适配起点。适配cap80/p20/Adam0.0001、batch16、实测Y MSE＋0.001神经状态残差能量，包含epoch0选模。参数删减之外其余训练合同保留。各H1结构仅seed2024一份，五seed仍仅覆盖温度随机性/输入实现，不能称五次完整H1重训。

## 运行与产物

- 入口：`${PYTHON} ${WORKSPACE}/temperature_prediction/va_core_ablation_all_20261008/launch_all.py`，工作目录 `${WORKSPACE}`。
- GPU0＋GPU1各一个独立队列worker、CPU4线程；A7 CPU4与GPU任务并行。原同步LOCK/STOP/失败独立记录机制复用，失败停止派发新阶段、保留在跑子任务自然结束；无新增重试、兜底或测试。H1训练原owner-aware GPU等待逻辑保留。
- 队列为17个逻辑训练组合：15个温度seed＋2个H1结构；共5个新增初始化、15个温度两阶段、2个H1预训练/适配；每组训练后接CPU4同节点评价。仅最多2个GPU训练同时运行，全部组合登记，不为每个模型单独占一张卡。
- 输出 `temperature_prediction/outputs/va_core_ablation_all_20261008`；日志 `temperature_prediction/logs/va_core_ablation_all_20261008`。各温度目录保存config、frozen_source、checkpoint、缓存/预测、阶段history和独立记录；H1目录保存原cfg、frozen_source、预训练/适配history与best；evaluation_{group}/seed{seed}保存metrics.json与预测NPZ。
- launch.json、progress.json保存实际PID/GPU/阶段、已完成/失败任务；每job日志、pid、exit与suite.log/dispatch.json；最后comparison.json汇总各seed及mean/sampleSD(ddof1)，不集成。
- 指标：温度共同26窗及完整66窗分别℃RMSE/MAE；四Y共同26窗全分钟实测节点pooled mm RMSE/MAE/MSE，末2hRMSE、逐通道RMSE/MAE/bias、与A0逐seed差；模型参数与实际阶段时间同步保存。CPU/GPU微小舍入差如实保留，不修正预测。
- 原源码依赖：冻结H1工程 `${WORKSPACE}/outputs/va_1_20261004/frozen_project`，包含原hybrid_model、FFT/BW/RK4和原训练objective；温度原vendor与数据/指标模块完整复制。此目录遗留原采样实验的入口/配置是复制来源，实际只运行本README指定launch_all.py→run_all.py与本次A组配置。

本次预计超过一小时，客户端开始17:39:35、连续工作截止18:39:35。后台正常启动与首日志核对后停止监控；若到界限还有未启动工作，保存进度并向用户报告，不终止已运行任务。

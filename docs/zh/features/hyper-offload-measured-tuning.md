# HyperOffload 测量驱动的有限步调优

本流程在调用方提供的可行初始策略和合法区域内，联合选择细粒度重计算、常驻、卸载及搬运窗口。它按完整训练步测量候选，确认或回退后冻结。`measured` 保留默认10步、最多12步的原有有限邻域；`explore` 使用全局搬运容量驱动的大步探索，默认40步、最多60步，并接入选择性无损压缩候选。稳定阶段不再搜索、采集轨迹、校准或执行调优共识。

## 使用边界

- `MeasuredOffloadSession` 不读取模型名称、固定层号或固定张量大小。区域名称只是身份；适配器负责证明重计算和常驻边界合法。
- `measured` 自动候选仍使用 raw 传输；`explore` 在存在兼容的原生库时，使用真实激活自动校准和生成按当前执行图绑定的选择性 SplitZip 候选。是否接受压缩由完整步时决定。
- 必须提供整卡容量、Host 容量和可行的初始策略。容量估算和事后实测不会保证任意初始策略都不发生 OOM。
- `measured` 至多测试两个新候选，`explore` 使用更大的有限预算；两者都不保证非单调空间的全局最优。非 Qwen 模型须先认证适配器；当前大模型性能验证集中于 Qwen。

## 全局容量驱动的探索入口

`mindspeed.core.memory.hyper_offload.tuning.exploration_session.ExplorationOffloadSession`
接受相同的模型、区域适配器、容量和完整训练步观测。初始策略须采用命名区域和原始传输。

```python
session = ExplorationOffloadSession(
    initial_policy, model=model, regions=regions,
    limits=TuningLimits(
        max_steps=40, max_candidates=11, confirmation_steps=4,
        max_search_seconds=180.,
        device_peak_bytes=device_budget, host_peak_bytes=host_budget,
    ),
    host_mb=host_budget // 1024**2,
    transfer_size_calibration=True,
    enable_compression=True,
)
```

探索主路径按以下顺序执行：

1. **整体估算窗口容量。** 先根据当前轨迹中的前向、反向时长和实测链路带宽，计算一次全局原始往返字节预算；初始大卸载方案先扣除预计移除的重计算时长。D2H、H2D各自须装入允许区间，二者的时间之和还须装入共享复制队列的区间并集，重叠窗口只计一次。预算模块不接受节点身份。
2. **构造大卸载起点。** 对具有当前生命周期证据和已观测结构成本的重计算区域，批量展开为原始卸载。新增Host预留、整卡容量和算子工作集均须满足约束。没有观测到的成本不视为零；不能仅凭模型名称推测区域可展开。
3. **在全局预算下分配节点。** 保留已测的较大卸载图，即使它暂时较慢。使用现有峰值模型筛选真正跨越压力位置的存储空闲区间，再按降低搬运需求所需的重计算成本分配细粒度区域。初始探索全局预算的100%、75%、50%、25%，可行且不同的策略才进入实测。相同存储版本按集合计数，新出现的可卸载重计算输入也占用预算。相同成本时优先保留较早产生的激活用于卸载。
4. **实测后重新分配并微调。** 全局分配腾出搬运容量后，再批量展开初始受限、且内部尚未观测过的不透明父区域；这些初始决定不能永久锁定搜索范围。已经观测过的区域由全局分配继续组合，避免反复支付同一大步展开的代价。保留大流量参考图、较快图和新展开图；较快策略的内部全部隐藏于重计算时，仍保留生成该策略的参考图。围绕较快的预算点探索两侧中点，同时测试搬运窗口、恢复分配流、释放时机、缓存清理周期、容量允许的常驻组合和选择性压缩。重计算也可批量改为常驻，但须使用当前不透明调用的生命周期，并保守预留历史已观测结构的全部输入、输出和内部存储，不提前扣除旧重计算的预留。改变结构后重新采集轨迹和构建计划，旧图的存储ID只用于估算，不直接装入新图的执行器。
5. **确认或回退并冻结。** 完整步时、设备峰值和Host峰值仍是最终依据；不同rank使用一致的观测和搜索预算。两步均值更快、但未过波动门槛的较早候选，也能在预算允许时恢复并独立确认。必须同时容纳恢复步、完整确认周期和回退验证步；接受时仍使用原收益和波动门槛，失败则返回已测的较稳妥策略。`explore` 启用 `TuningLimits.confirm_best_observed`，默认控制器保留原行为。冻结后关闭观察器，释放搜索引用的图和候选状态。训练步数、候选数或搜索时间耗尽都不表示达到全局最优。

全局容量是用于确定探索起点的估算上界，不能保证所有字节都能被计算掩盖。具体节点的产生时间、最后使用、释放完成时刻、反向恢复时限、包延迟、预取分配和共享队列仍会限制可执行的方案。峰值队列预测只适用于其当前捕获图，也不能等同于已测关键路径停顿。`PhaseCapacity` 使用分配器累计高水位的保守包络，不把它当作可随意分配的瞬时空闲显存。

微调集合按不同执行结构保留候选，每个结构使用其当前较快的策略。同一结构的压缩和清理周期变体不会重复占位、挤掉其他有潜力的卸载方案。在平衡压缩、清理、调度与常驻等调整类别后，优先给该类别尚未测试过的候选结构一次机会，再重复微调同一结构。这个顺序只决定测试覆盖，接受和冻结仍依据完整步时与原有确认门槛。

Qwen示例使用 `--offload-tuning-method explore`；默认11个候选、40步。显式增加预算时最多60步，需同时为确认与回退留出空间。`--offload-tuning-search-seconds` 单独设置搜索计算预算，默认180秒；该预算包含候选生成和分布式等待，不是训练墙钟预算，也不改变稳定步时目标。`--offload-disable-compression` 可关闭压缩采样与候选，`--offload-compression-library` 可指定原生库。性能验证目前集中于Qwen；两种合成结构的完整数值与RNG测试用于验证适配器行为，不代替其他模型的性能证据。

`exploration_diagnostics` 保留全局容量、各预算点预计流量、改变的区域、候选来源和排除原因。`compression_calibration` 记录原生能力、真实数据校准和不可变节点范围。请将这些估算与实际候选样本、冻结后的稳定步时分别报告。

## 通用入口与训练循环

`mindspeed.core.memory.hyper_offload.tuning.measured_session.MeasuredOffloadSession`
接受 `model`、认证后的 `regions`、命名 `JointPolicy`、`TuningLimits` 和 `host_mb`。

```python
limits = TuningLimits(
    max_steps=10, device_peak_bytes=device_budget, host_peak_bytes=host_budget,
)
session = MeasuredOffloadSession(
    initial_policy, model=model, regions=regions, limits=limits,
    host_mb=host_budget // 1024**2,
    transfer_size_calibration=True,
)
```

训练循环在 session 上下文内执行前向与反向。完整优化步结束后，若尚未稳定，构造 `StepObservation`，将完整步时、设备峰值和 Host 峰值在同一进程组内取最大值，再调用 `session.observe()`。所有 rank 必须参与相同顺序的调优调用，并校验决策一致。稳定后跳过调优观测和决策归约。示例入口实现了这一完整流程。

`transfer_size_calibration=True` 仅在有限 warmup 内按实际流量选择探针，并受可用内存约束；默认关闭。`joint_refinement=True` 可在第二个候选中同时细化区域和清理周期，默认关闭。周期候选必须测量和确认完整周期，预算不足则不继续尝试。

若当前候选的完整周期分数更低，但短样本尚未越过噪声门槛，控制器可用剩余预算执行一次独立完整确认，再决定是否接受。该路径要求回退对象为已测量的非周期策略，并预留一个完整回退验证步；确认仍使用相同的收益与噪声门槛。未通过确认就恢复先前策略，最多12步冻结，不追加候选，也不按模型名称调整门槛。`selection_log` 记录原先胜者与待确认候选，避免把待确认误报为已接受。

Qwen 示例的 `--offload-tuning-method measured` 是默认调优入口，结构布局默认 `gated`。使用时显式提供 `--offload-tuning-device-peak-mb` 和激活/Host 预算；可选 `--offload-transfer-size-calibration`、`--offload-joint-refinement`。`parallel_mlp` 用于经认证的并行残差结构，不能仅凭模型名称选取。

历史 Qwen 手工候选已移到同示例目录的 `legacy_tuning.py`，只有显式 `--offload-tuning-method legacy` 才使用。旧的 `mindspeed...tuning.proposals.QwenRegionNeighborhood` 导入路径不再提供；通用包仅保留通用候选逻辑。固定层数扫描和本机部署脚本独立保存为实验材料，不是生产算法默认规则。

## 新缓存机制与成本口径

NPU 运行时延续 PR #14：在选定分配操作及 FSDP all-gather 分配前，当 reserved 达整卡 93.5%、reserved 与 allocated 差至少 512 MiB 时，调用虚拟缓存回收；步入口清理优先保留物理句柄。接口缺失或明确不支持配置时回退传统 `empty_cache()`；其他后端错误继续抛出。

固定内存压力检查在稳定阶段仍存在，它属于普通分配运行时，不属于策略搜索。关闭周期性步入口清理也不会关闭压力回收。保留物理句柄不等于将物理内存归还系统，容量验收不能只看 allocator reserved 的下降。

有限调优期间记录 `cache_reclaim_stats`：`checks`、`calls`、`check_seconds` 和 `reclaim_seconds`。其中 `check_seconds` 已包含 `reclaim_seconds`，不能相加。两者都不属于步入口的 `last_trim_seconds`，也不能将全部压力回收时间当作减少清理次数可以节省的时间。候选仍由完整训练步时决定是否接受。

冻结后 `record_cache_reclaim=False`，压力回收照常执行，但这组调优计数和计时停止更新。普通 `OffloadSession` 默认不记录它们；可用 `OffloadConfig(record_cache_reclaim=True)` 做显式诊断。

稳定性能与调优成本分别报告。严格调优成本应与相同实际最终策略、相同捕获和校准配置的直接固定运行比较；步骤耗时、包含训练的 warmup 时间和嵌套子项不能重复相加。

## 可选无损压缩

`mindspeed.core.memory.compression.SplitZipCodec` 保留完整 BF16 位模式、原始格式回退、异步传输、选择性压缩和有界包缓冲复用。其他 dtype 或被选择器排除的请求走 raw。每个 Host 快照保留自己的编码信息；切换策略后恢复旧快照不依赖当前策略猜测格式。

先在目标 CANN 环境中编译源码：

```sh
bash mindspeed/core/memory/compression/native/build.sh /absolute/build/directory
```

通过 `MINDSPEED_SPLITZIP_LIBRARY` 或构造参数指定生成的动态库。调用方可提供真实校准样本及选择器：

```python
codec = SplitZipCodec(library_path, sample=calibration_sample, selector=selector)
session = OffloadSession(OffloadConfig(
    max_resident_activation_mb=activation_mb,
    max_offload_activation_mb=host_mb,
    codec=codec,
))
```

选择器接收 `TransferContext`，可使用 dtype、实际字节数及当前轨迹信息；op/storage ID 只在当前轨迹中有效。改变重计算结构后不能复用旧 ID 规则。调用方负责 codec 生命周期，在所有相关请求结束后关闭；会话不会擅自关闭调用方共享的 codec。

自定义联合候选可用 `JointOffloadSession(codecs={...})` 注册不可变的传输配置，并在 `JointPolicy.codec_key` 中引用。`raw` 为保留键。`MeasuredJointNeighborhood` 与 `calibrated_transport.py` 是可复用的成本组件。`ExplorationOffloadSession` 的自动压缩接线位于 `codec_calibration.py` 和 `selective_codec.py`：真实BF16样本仅在初始warmup有界采集；同一会话使用固定码本，节点选择器绑定执行图；无兼容原生库或校准未通过时明确记录原因并继续raw候选。包复用默认关闭；压缩率不能用于折扣最坏临时空间或Host预留容量。

`tools/hyper_offload/compression_applicability.py` 提供有界真实激活诊断；带采样和校准的实验不计训练性能。`splitzip_basic_trial.py` 提供原生资格和小模型对照。`audit_results.py --pairs MANIFEST --protocol PROTOCOL` 要求显式冻结统计协议，拒绝复制运行和不匹配证据。

## 全局探索实测结果

见[hyper-offload-exploration-validation-20261104.md](hyper-offload-exploration-validation-20261104.md)：三组同卡新旧自动对照、实际最终策略的完整调优成本、压缩执行及时间线证据。8.5秒基础目标和8.0秒挑战目标未达到，限制与未验证范围分别保留。

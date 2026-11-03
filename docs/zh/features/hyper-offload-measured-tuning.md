# HyperOffload 测量驱动的有限步调优

本流程基于 PR #14 的 NPU 缓存回收运行时，在调用方提供的可行初始策略和合法区域内，联合选择细粒度重计算、常驻、原始卸载及搬运窗口。它按完整训练步测量候选，确认或回退后冻结。默认预算 10 步，测量入口最多接受 12 步；稳定阶段不再搜索、采集轨迹、校准或执行调优共识。

## 使用边界

- `MeasuredOffloadSession` 不读取模型名称、固定层号或固定张量大小。区域名称只是身份；适配器负责证明重计算和常驻边界合法。
- 当前测量驱动的自动候选使用 raw 传输。SplitZip 无损压缩实现、选择性接口和独立联合成本模块已保留，但真实数据自动校准与压缩候选的生产接线尚未完成。
- 必须提供整卡容量、Host 容量和可行的初始策略。容量估算和事后实测不会保证任意初始策略都不发生 OOM。
- 至多两个新候选的有限搜索不保证非单调空间的全局最优。非 Qwen 模型须先认证适配器；本轮新缓存基线的性能验证集中于 Qwen。

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

自定义联合候选可用 `JointOffloadSession(codecs={...})` 注册不可变的传输配置，并在 `JointPolicy.codec_key` 中引用。`raw` 为保留键。`MeasuredJointNeighborhood` 与 `calibrated_transport.py` 是可复用的成本组件，尚未自动生成生产压缩策略。包复用默认关闭；压缩率不能用于折扣最坏临时空间或 Host 预留容量。

`tools/hyper_offload/compression_applicability.py` 提供有界真实激活诊断；带采样和校准的实验不计训练性能。`splitzip_basic_trial.py` 提供原生资格和小模型对照。`audit_results.py --pairs MANIFEST --protocol PROTOCOL` 要求显式冻结统计协议，拒绝复制运行和不匹配证据。

# HyperOffload 的 device 窗口联合调优

对于会重复运行的训练配置，可以把既有 device profile 交给 `ExplorationOffloadSession`。训练会重新捕获参考图，比较不同的重计算、常驻、卸载和搬运计划，再根据完整训练步的时间与容量确认最终策略。冻结后，训练继续使用选中的计划。省略 `device_window_profile` 时沿用原有探索入口。

## 训练入口

```python
import json
from pathlib import Path

profile = json.loads(Path("device-profile.json").read_text())
session = ExplorationOffloadSession(
    initial_policy, model=model, regions=regions, limits=limits,
    host_mb=host_mb, transfer_size_calibration=True,
    enable_compression=True, device_window_profile=profile,
)
```

模型适配器负责注册可切换且语义兼容的区域，并在完整训练步边界调用 `observe`。Qwen 示例在 `--hyper-offload-autotune --offload-tuning-method explore` 下接受 `--offload-device-window-profile device-profile.json`。首次运行需要先采集参考图和能暴露候选区域内部计算、存储版本的图，或提供等价的已验证观测。运行入口读取已经准备好的 profile。原始 profiler 中的任务类型、时钟原点、区域与硬件事件的对应关系由适配器提供。

## 生成窗口

在仓库根目录运行以下命令。输入应包含下面的图、区域和硬件事件字段。输出文件需要是新路径。

```bash
python -m tools.hyper_offload.device_window_profile observed-events.json device-profile.json
```

| 输入 | 含义 |
|---|---|
| `schema`, `anchor_signature`, `anchor_op_count` | 当前格式版本为 1。签名来自合并后的 phase report，算子数取自同一次捕获。 |
| `anchor_recompute`, `anchor_resident`, `resident_mb`, `anchor_window_scale` | 参考执行结构、受管激活预算和参考搬运窗口系数。 |
| `hardware_samples` | 每个 rank、每个 step 独立提供 `compute`、`dma`、`collectives` 三组设备起止区间，单位秒。DMA 包含两个方向和未分类的硬件搬运。一个样本的所有时间共享同一原点。 |
| `regions` | 每个可打开的参考检查点包含 `name`、`saved_seconds`、`workspace`、`native_bytes`、`first_op`、`last_op`、`native_until`。字节上界和算子位置来自测量及生命周期映射。 |
| 区域的 `samples` | 每个硬件样本对应一项，包含移除重计算后要扣除的 `removed_compute` 区间及 `packets` 列表。不可归属的版本应从候选排除。 |
| 包字段 | `key` 是参考版本的去重标识，`size` 是原始字节数。`ready`、`first_pressure`、`peak_pressure`、`prefetch`、`use` 是该样本中产生、需释放和反向需求的时间界限。`d2h`、`h2d` 是两方向实测或由尺寸校准估计的包时长。 |
| 可选 `native_groups` | 使用与区域相同的字段，另以 `replaces` 指明整棵子树内将撤销的参考检查点。完整子树的原始激活和工作集都要计入 phase 容量。 |
| 可选 `calendar_comparison`、`codec_origins` | 前者保存同图、同设备、同结构的原计划与双向重排样本，使较快方案成为待验证起点。后者是已测参考图上愿意再次尝试压缩的“产生算子位置、原始字节数”列表。运行时按新捕获的实际版本和新 codec 校准重新匹配。 |

导出工具先对硬件计算事件求并集，再扣除所有 DMA。它同时生成扣除集合通信的第二组区间。两组区间代表两种资源竞争假设。集体通信可以与 DMA 重叠，哪一种分配更快，由完整训练步决定。直接使用已有窗口时，分别传入 `samples` 和可选的 `collective_free_samples`。

## 从窗口到实际执行

1. **分配整个训练步。** 每个 rank 和样本独立使用一个 D2H/H2D 共享的时间表。算法扣除可能随重计算消失的计算区间，校验包的产生、释放、预取与反向使用约束。同一存储版本只记一次。不同压力期限、原生常驻和成组替换会生成多种完整配置。
2. **重建新图。** 改变重计算或原生常驻区域后，重新捕获该结构。参考版本标识只参与估算。实际峰值规划器重新选择跨越压力位置的存储区间，当前图的版本决定哪些节点可参与无损压缩。
3. **安排双向搬运。** 可选的 `JointPolicy.shared_transfer_calendar` 使用当前图的算子设备计时、定向包时长及内存安全的释放/预取范围，为两种方向分配实际提交位置。在窗口容纳不下包时，回到该包最早的合法位置，运行时事件保护正确性。延迟 D2H 的执行位置仍保留原始产生节点的 codec 身份。
4. **实测确认。** 规划的容量和耗时是预测。训练时重新检查完整步的设备与 Host 峰值，用各 rank 一致的观测值选择、确认或回退。超出步数或搜索时间上限后结束探索。跨模型复用需要重新采集匹配的新图和时间线。

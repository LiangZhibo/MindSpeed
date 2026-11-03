# HyperOffload 自动异步激活卸载

## 目标与范围

本特性将团队在 HyperParallel 中实现的动态图激活卸载流程移植到 MindSpeed。
无需依赖 HyperParallel，也不调用 SmartSwap、SwapLayerInput 或既有 optimizer swap 模块。
默认关闭，通过公共 context API 或 MindSpeed 特性开关启用。

首版覆盖单个 rank 内的激活 storage，参数和优化器状态仍由原训练组件管理。
它采用 eager 计算与卸载计划回放，不是 torch.compile、ACL Graph 或 NPUGraph 捕获。

## 执行流程

```text
首个完整前反向：TorchDispatchMode 记录访问，同时在线驱逐激活
    → ActivationTrace / OpGuide
    → GreedyResidencyPlanner 或 PeakResidencyPlanner
    → ResidencySchedule
后续完整前反向：继续执行 eager 算子，按 pre/post actions 搬运和释放
```

`ShadowTensor` 保存 shape、stride、offset、dtype 等逻辑信息，由 `PhysicalBuffer`
管理实际 Host / Device 存储。取用时重新解析设备视图，必要时按需回载。
调度动作是 `COPY_D2H`、`RELEASE_DEVICE`、`COPY_H2D`、`RELEASE_HOST`。
专用拷贝流和 event 保护异步访问及缓冲回收；回收 Host 缓冲时优先等待 H2D 完成事件。
写操作按 schema 的参数结构识别，包括 Tensor 列表及关键字 `out`；写入前等待旧 D2H 读完并使 Host 副本失效。
会话外创建的 view 继续共享物理缓冲和写入保护。不透明区域的输入按可能写入保守处理。
清理及最终释放时，仍存活的输出、梯度或 view 保有有效设备数据，之后由 Python 引用生命周期释放；
这部分保留数据会占用实际显存，不能计作已释放的物理内存。

第一轮使用在线启发式卸载，不要求先运行一个无卸载 step。峰值规划器对 storage
访问间隙、写入版本和别名生命周期建模，反复选择覆盖当前峰值的候选。
可选的搬运掩盖在驻留预算允许范围内调整释放和预取位置；目前不是基于完整时间模型的最优调度。

## 公共 API

```python
import torch
import torch_npu
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload.planning import PeakResidencyPlanner

torch.npu.set_device(0)
session = OffloadSession(OffloadConfig(
    max_resident_activation_mb=1024,
    max_offload_activation_mb=4096,
    planner=PeakResidencyPlanner(enable_communication_masking=True),
))

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with session:
        loss = model(batch)
        loss.backward()
    optimizer.step()
```

同一个 session 应覆盖相同范围的完整前反向，并跨训练迭代复用。优化器更新放在 context 外。
API 的默认规划器仍为 GreedyResidencyPlanner，以保持来源实现的行为；需要峰值规划时显式传入。
`skip_offload` 支持 partial、callable 对象及 nn.Module，并保留非可导输出和未使用梯度的原生语义。
其 backward 必须在创建该区域的同一次 session 内完成；不同 session 的 ShadowTensor 应先在会话外克隆。

## MindSpeed / Megatron 接入

沿用原有 `import mindspeed.megatron_adaptor` 入口，在启动参数中增加：

```shell
--hyper-offload \
--hyper-offload-device-mb 1024 \
--hyper-offload-host-mb 4096 \
--hyper-offload-planner peak \
--hyper-offload-communication-masking
```

特性通过 MindSpeedPatchesManager 包装 `megatron.training.training.get_forward_backward_func`。
一次 context 覆盖完整 schedule 的所有 microbatch 和 backward；不把每层或每个 microbatch
分别视为完整 step。evaluation / forward_only 调用直接执行原 schedule，不改变训练 session 的阶段。
该接入不捕获 optimizer.step，也不卸载其状态。同一 factory 返回相同训练函数、配置不变时复用原 wrapper/session，
重复查询不会每步重新 warmup；配置或训练函数变化时创建新 wrapper。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| --hyper-offload | 关闭 | 启用独立激活卸载 |
| --hyper-offload-device-mb | 1024 | 被跟踪激活的驻留预算，单位 MiB |
| --hyper-offload-host-mb | 65536 | 固定 Host 内存池上限，单位 MiB；按需申请 |
| --hyper-offload-planner | peak | CLI 使用 greedy 或 peak 规划器 |
| --hyper-offload-communication-masking | 关闭 | 仅 peak 支持，调整搬运与驻留位置 |

## 适用条件和限制

- warmup 与后续执行必须保持可见算子序列、输入标量参数、张量元数据及别名关系稳定。replay 在执行调度动作前检查算子身份和输入，在绑定输出前检查输出结构及别名，并在退出时检查完整算子数量；不一致会报错，不自动重规划。
- 复用计划时需保持梯度初始状态一致，例如每轮都使用 `zero_grad(set_to_none=True)`；叶子输入需要梯度时，也应一致地清理其 `.grad`。
- `@skip_offload` 可把函数视为不透明区域，内部不进行逐算子卸载；边界结构和内部峰值仍需由使用方控制。
- 支持 `transpose_`、`t_`、`squeeze_`、`unsqueeze_`、`as_strided_` 的包装张量元数据同步。`set_`、`resize_`、`resize_as_` 这类 storage 重绑或改容操作目前明确拒绝。`out=` 必须预先具有正确形状；若底层自动扩容，调用会报错，相关输出被标为不可再读取，应丢弃并重新创建。
- 不允许嵌套或重入 session。进入失败会恢复活动上下文；退出、规划及传输异常会清理状态。清理本身失败时保留原异常，并禁止复用该 session。
- 激活预算不是整卡 HBM 硬上限。参数、梯度、优化器、通信池、workspace、当前算子输出、延迟回收及分配器碎片均需额外预留。
- 如果当前工作集不可驱逐，warmup 会报错；规划候选耗尽也不代表预算已满足，需检查规划警告及实际峰值。
- Host 池耗尽且无可复用缓冲时会报错，不自动突破上限或回退 pageable 内存。
- 暂不支持与设备图捕获 / 编译模式、可变长度 schedule，以及其他接管激活存储的卸载特性叠加；CLI 对已知冲突显式报错。
- 验证包含独立模块、小模型完整数值与RNG检查，以及Qwen原生FSDP2双NPU的细粒度重计算／卸载组合。其他FSDP配置、EP / PP、多流特殊算子、FP8和其他大模型仍需单独验证，不据此声明支持。
- 基础 `OffloadSession` 不自动改变重计算策略；可选的有限多轮探索见下节。尚未实现计划持久化或参数 / 优化器卸载的联合选择。
- warmup 的逐算子计时包含同步，不能用它的整步耗时代表稳态回放性能。

## 有限多轮联合探索（显式启用）

`tuning.exploration_session.ExplorationOffloadSession` 在训练初期联合选择命名重计算区域、
原始卸载、搬运窗口、回放清缓存周期和可用的无损压缩节点。基础 session 和旧
`MeasuredOffloadSession` 的默认行为保持兼容。Qwen 原生 FSDP2 示例增加
`--hyper-offload-autotune --offload-tuning-method explore`；其默认预算为40步、11个候选，
可通过 `--offload-tuning-steps` / `--offload-tuning-candidates` 显式调整，探索会话最多接受60步。
`--offload-tuning-search-seconds` 单独设置搜索计算预算，默认180秒；它包含候选生成和
分布式等待，与完整训练步预算分开。搜索成本不作为稳定步时的替代指标。
训练步预算包含 warmup、候选测量、最终确认与回退，不只计算有效性能样本。
分布式探索在候选生成前后使用各 rank 搜索耗时的最大值判断时间预算，避免本地
时钟在上限附近造成不同选策分支；该同步和其计时同样在冻结后停止。

其他模型通过 `NamedRecomputeRegions` 注册可合法重计算的 module / callable、层级关系和冲突。
算法使用这些身份及观测到的结构、输入形状、storage版本、生命周期和成本；不读取模型名、
特定层号或固定张量尺寸阈值。结构相似只是提出实验的依据，每个改变执行结构的策略都重新
采集完整 trace，并通过整步数值、内存和时间验证。通用接口不等于未经验证的网络已受支持。

搜索先从整步前反向窗口和实测链路成本估算共享复制队列容量，再构造大卸载起点。
保留较大参考图，从全局预算的100%、75%、50%、25%分配细粒度区域，之后围绕实测较快点
细分预算，并交替进行时序、常驻和压缩调整；重复或不可行的策略不会进入实测。
既有峰值规划器决定哪些 storage 访问间隙需要卸载；队列估计扣除实际释放前和预取分配后的驻留，D2H/H2D共用复制流。
固定图的估计只用于组织候选和解释证据，不能证明新图的掩盖效果或全局最优。有限邻域耗尽、
步数耗尽和搜索时间耗尽均不等于已经收敛。

已测组合可以同时包含重计算和常驻子区域；扩展时分别比较保持常驻、将其改为细粒度重计算的
容量成本。已有常驻决策也能重新选择原始卸载或重计算，避免早期校准的一次选择锁死后续搜索。
缺少常驻区域的内部生命周期观测时，新的输入与原生 workspace 按整个训练步保守预留，
不会预先扣除推测可释放的旧常驻内存。每次变更仍需采集新图并实测。

集成方必须提供 `TuningLimits.device_peak_bytes`（整个设备分配峰值约束）和 Host 上限，
初始策略使用命名区域及 `codec_key='raw'`。一次 context 对应完整前反向，优化器更新在外部完成。
每个调优步结束后，将整步时间、整个设备分配峰值和 Host 池分配量在相同 process group 上取
max-rank，调用 `session.observe(StepObservation(...))`。所有 rank 都必须以相同次序参与观测；
具体集成可参照 Qwen 示例的 `observe_tuning`。开始下一步前缺少观测会报错。训练异常不会被
当作可忽略的失败性能样本，清理失败后也不会继续复用会话。

确认最终策略后，`session.stable` 为真，停止搜索、跨 rank 选策、区域及内存观测；稳定阶段继续
执行冻结的 eager 计划和正确性检查。报告应分别记录稳定整步时间，以及初期捕获、计划构建、
搜索、校准和确认的完整耗时，不用 warmup 的同步计时替代稳定性能。

### 选择性无损压缩

可用原生 NPU SplitZip 库时，通过 `--offload-compression-library` 或
`MINDSPEED_SPLITZIP_LIBRARY` 提供路径；`--offload-disable-compression` 禁用自动校准。
缺少库或设备能力会在 `compression_report` 中明确记录原因，继续原始卸载搜索。
该路径在初始 warmup 中收集有界的真实 BF16 激活副本，按实际尺寸分组做原始传输与编解码往返
测量，并校验逐位一致。字典来自本次真实值；候选只使用各 rank 共同支持的校准结果。

编码选择器绑定具体执行结构及传输节点，切换结构先恢复原始卸载。既考虑未掩盖的传输，也保留
压缩已掩盖数据以腾出后续传输容量的候选；最终是否启用由完整训练步实测决定。容量估计按
最坏包大小、临时空间及新增 Host 缓冲预留，不能用预期压缩率折扣显存。无足够样本或余量时
明确跳过；不支持的存储保留运行时 raw fallback。冻结为 raw 时关闭自动 codec 并释放其资源。
压缩可用、压缩率较好或单包传输较快，均不代表整个策略一定更快。

## 测试

CPU 单测可以在不安装 Megatron、torch_npu、HyperParallel 的情况下执行：

```shell
python -m pytest tests_extend/unit_tests/mindspeed/core/memory/hyper_offload -q
```

真实加速器集成测试：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest tests_extend/system_tests/hyper_offload -q
```

包括前反向精度、view / in-place、opaque region、warmup 在线驱逐、两种规划器、实际峰值显存和连续训练。
测试使用独立的基线模型，并在对称的梯度清理与同步条件下测量；NPU 显存统计直接使用 torch.npu。
无可用 NPU / CUDA 时集成测试跳过。CPU 自动化通过 GitHub Actions 执行。

## 来源与移植范围

保留原代码 Apache-2.0 版权与许可头，来源为：

- [HyperParallel issue 211](https://gitcode.com/mindspore/hyper-parallel/issues/211)
- [PR 814](https://gitcode.com/mindspore/hyper-parallel/pull/814)：基础执行与运行时
- [PR 986](https://gitcode.com/mindspore/hyper-parallel/pull/986)：峰值规划器
- [PR 1250](https://gitcode.com/mindspore/hyper-parallel/pull/1250)：Host 缓冲回收事件修复

具体来源提交与新增的 MindSpeed 适配范围见模块 README 和 Third_Party_Open_Source_Software_Notice。

完整组件覆盖与当前环境限制见[全组件调查报告](hyper-offload-component-audit.md)。

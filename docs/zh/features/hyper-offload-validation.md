# HyperOffload 首次移植验证记录

日期：2026-10-27。基线：Ascend-TechDev/MindSpeed `core_r0.18.0`，提交
`1135d097c6e4a3533bf4d22904b92bd904b11507`。目标：个人 fork 的 `hyperoffload` 分支。

## 执行环境和结果

| 验证 | 环境 | 结果 |
| --- | --- | --- |
| 单元测试 | Python 3.12、PyTorch 2.12.1，CPU | 264 passed |
| 实机集成测试 | Ascend 910B3、Python 3.10.22、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post2、CANN 9.1.1 | 20 passed，0 skipped |
| 无加速器时的集成测试收集 | CPU 环境 | 20 skipped，不误报加速器验证成功 |
| 语法与差异检查 | compileall、git diff --check | 通过 |

GitHub Actions 另外配置 Python 3.10 / 3.12 与 PyTorch 2.10.0 CPU 单测，结果以 PR checks 为准。
实机采用服务器已有的 torch / torch_npu 组合，不能据此声称已经验证目标依赖表中的全部版本组合。

## 覆盖范围

- 原来源实现的 API、IR、在线驱逐、两类规划器、算子 dispatch、storage 跟踪、ShadowTensor、物理缓冲及异常清理。
- Host 缓冲在 H2D 完成前不得复用的回归用例，包含 PR 1250 的事件选择修复。
- view、in-place、opaque region 及预 hook 替换 buffer 的数值行为。
- 真实 D2H / H2D、warmup 和 replay 的实际峰值显存，而非只检查模拟曲线。
- Greedy / Peak × FP32 / BF16 四种配置分别连续训练 10 步，每一步使用新输入；对比独立模型的 loss、梯度及带动量 SGD 更新后参数。
- 连续训练测试中，Host 池在前几步后保持稳定：FP32 为 16,157,696 bytes，BF16 为 6,769,664 bytes，均低于配置的 64 MiB 上限。
- MindSpeed 开关默认关闭、冲突参数拒绝、完整 schedule 作用域、evaluation 绕过和异常退出由单元测试验证。

FP32 连续训练的数值比较容差为 rtol=1e-4、atol=1e-6；BF16 为 rtol=0.02、atol=0.002。

## 部分实际显存观测

下表来自相同运行中的日志，单位 MiB。基线和卸载模型使用独立参数；测量前对称清理梯度并同步设备。
它们是测试网络的绝对 allocated 峰值，不是整卡 HBM 上限，也不代表真实大模型的收益比例。

| 用例阶段 | 无卸载基线 | 卸载 | 说明 |
| --- | ---: | ---: | --- |
| 1024 维、4 层、batch 8192，warmup | 256.04 | 164.04 | 测试 replay 前的首次执行 |
| 同配置，replay | 272.06 | 172.06 | 自动驻留计划执行 |
| 512 维、3 层、batch 4096，warmup | 54.02 | 56.03 | 初始策略没有降低此阶段峰值 |
| 同配置，replay | 58.03 | 46.02 | 规划后的峰值降低 |

在线启发式不保证所有用例的首次峰值都低于无卸载基线。
连续训练的 FP32 用例还会报告 3 MiB 激活预算低于约 6 MiB 的模拟不可卸载工作集；
该用例验证数值正确和资源复用，不表示预算已被满足。预算警告与实际峰值必须保留和检查。

## 本轮修正

1. 完成包路径迁移，去掉 HyperParallel 运行依赖，保留来源版权与许可说明。
2. 增加默认关闭的 MindSpeed 特性和完整前反向 schedule 适配。
3. 合入 Host 池回收时优先使用 H2D 完成事件的修复。
4. 基线模型改为深拷贝，避免来源测试复用同一组层和参数。
5. NPU 显存统计使用 torch.npu，避免当前环境 torch.accelerator 内存接口触发 allocator 断言。
6. 对称清理基线 / 卸载梯度，防止对照测量被另一模型的残留梯度污染。
7. 补充输入参数检查、已知执行模式冲突检查、重复训练及 GitHub CPU 自动化。

## 尚未验证

本轮不包含真实 Megatron 大模型训练、FSDP / EP / PP 多卡组合、FP8 或 checkpoint 联合使用。
CLI 包装器通过独立测试验证，但不以此代替完整 Megatron 训练验证。
参数 / 优化器卸载、任意动态结构、多轮自动重规划及设备图捕获属于后续工作。

## 正确性修复回归（首次移植后）

- CPU：291 passed；Ascend 910B3：32 passed、0 skipped。环境沿用上表，实机使用空闲物理卡 2。
- 新增 27 项 CPU / 12 项 NPU 回归，覆盖二次卸载前的原地写、会话外 view、Tensor-list 写入、
  关键字输出、跨 step 保留、Host 池复用、实际异步传输，以及执行计划不匹配时拒绝执行。
- 写入前等待 D2H 读完，废弃旧 Host 副本；Host 回收继续保留 H2D 完成事件保护。
- 物理缓冲跟踪全部存活别名，规划器保留集合包含执行器外创建的包装对象。
  清理时恢复仍存活的缓冲；最终释放也保护 replay 中新增保留的 loss，避免沿用 warmup 的 Python 引用寿命假设。
- replay 校验算子、输入标量参数、shape / dtype / stride / offset / storage 大小、
  pytree 结构、别名关系及逻辑激活绑定，并拒绝缩短或增加算子的 step；不包含自动重规划。
- 原测试的 warmup / replay batch 和梯度初始状态已对齐；另保留专门的执行变化拒绝用例。
  预 hook 替换 buffer 的用例原模型没有参数，原参数梯度比较为空；现使用相同随机种子，
  真实比较 loss 和输入梯度。
- 完整集成测试保留原有显存收益断言，以及 Greedy / Peak × FP32 / BF16 各 10 步训练比较。

保留输出的设备内存以及写入保护引入的同步属于正确性成本，本轮没有验证吞吐性能。
本轮也不扩展此前未验证的 Megatron / FSDP / EP / PP、大模型及重计算组合范围。

## 基础原子功能扩展回归

本轮在上一轮 291 CPU / 32 NPU 用例基础上，新增 63 CPU / 37 NPU 用例。
最终 CPU 354 passed；Ascend 910B3 69 passed、0 skipped，实机使用空闲物理卡 1。
首批 47 个定向用例在修复前为 22 failed、25 passed，剩余用例在定位过程中继续补入。

| 原子范围 | 验证与修复 |
| --- | --- |
| 原始字节及 view | FP32 / FP16 / BF16 / INT64，标量、空切片、非零 offset、转置、零 stride expand、步进切片；每种连续搬入搬出三次 |
| 原地元数据 | 修复 transpose_ / t_ / squeeze_ / unsqueeze_ / as_strided_ 后包装元数据未更新；NPU 同时比较三轮前反向与原生梯度 |
| storage 身份 | Shadow 使用逻辑 ID，并把 H2D 后的新物理 storage 重新映射到同一 ID；view / slice 不再错误地产生新 ID；大小按观测最大值记录 |
| Host 分配与等待 | 分配成功后再增加计数；预算满时可等待较大 bucket；等待失败保留池所有权；覆盖非法 size 与超出标准 bucket 的复用 |
| 传输事件 | 计算流 wait 不清除 H2D 完成事件；按当前 Host 分配的最近访问保护回收，覆盖旧 H2D 后又有新 D2H 的顺序 |
| 故障回收 | 注入分配、复制、event record、stream drain 失败；未完成复制不得失去 Host 所有权，必要时以整个拷贝流作为回收 fence |
| 会话状态 | 拒绝嵌套 / 重入；进入、退出、等待、profile、planner 异常后恢复；清理失败不掩盖原异常且禁止复用；reset 释放缓存输入 |
| 不支持的 storage 改动 | 显式拒绝 set_ / resize_ / resize_as_；自动扩容的 out= 报错且禁止继续读取，覆盖空输出和多输出 |
| 引用寿命 | 新增禁用循环 GC 的测试，确保解析出的原始 view 在 dispatch 返回后释放；已移除缓存递归闭包的引用环 |

原有实际显存下降断言、Greedy / Peak × FP32 / BF16 连续十步训练及参数 / 梯度比较均继续通过，
没有放宽显存或数值断言。扩容 out= 的保护属于明确拒绝不支持行为，不表示已经支持动态 storage 改容。
本轮没有扩展到真实 Megatron / FSDP 大模型、多卡、重计算组合或吞吐性能验证。

## 全组件调查（2026-10-28）

在 354 CPU / 69 NPU 基础上新增 237 CPU / 41 NPU 场景，最终 591 CPU / 110 NPU 全部通过。
组件矩阵、问题依据、覆盖率与后端受限场景见[全组件调查报告](hyper-offload-component-audit.md)。
核心包 CPU 语句覆盖率 97.87%，分支覆盖率 92.60%；不能据此替代实际大模型验证。

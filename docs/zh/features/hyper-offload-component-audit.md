# HyperOffload 全组件调查与测试矩阵

完成日期：2026-10-28。基线为个人仓库 `hyperoffload` 分支的 `8b6dd329`。
本轮逐一核查当前包的入口、执行器、IR、规划器、运行时、兼容层和 MindSpeed 开关，
在既有 354 项 CPU / 69 项 NPU 测试上新增 237 项 CPU / 41 项 NPU 场景。

## 最终结果

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| CPU 单元测试 | 591 passed | 含独立状态模型、故障注入、并发借用、资源销毁及配置组合 |
| Ascend 910B3 集成测试 | 110 passed，0 skipped | 空闲物理卡 2；真实拷贝、梯度、session 复用和原有显存断言 |
| 语法与差异检查 | 通过 | compileall、git diff --check |
| 核心包语句覆盖率 | 92.75% → 97.87% | CPU coverage.py 统计 |
| 核心包分支覆盖率 | 87.86% → 92.60% | 513 / 554 个分支被执行 |
| 核心包语句与分支综合覆盖率 | 91.72% → 96.71% | 不能当作 NPU 分支或端到端训练覆盖率 |

NPU 环境沿用既有 Python 3.10.22、PyTorch 2.12.0、torch_npu 2.12.0.post2、CANN 9.1.1。
覆盖率仅统计 `mindspeed.core.memory.hyper_offload`，包括假设备与故障注入；
MindSpeed feature 文件另有测试，但不在这一覆盖率分母中。

## 组件与场景对应关系

| 组件 | 扩展检查 | 结论 |
| --- | --- | --- |
| `api/config.py`、MindSpeed feature | 整数 MiB、布尔值 / NaN / inf / 字符串、关闭开关、`None` / `False` / `none` 图模式标记、冲突组合 | 预算类型明确校验；未启用图捕获的空值不再被误判为已启用 |
| `megatron.py` | 重复查询 factory、同一函数 / 配置复用、配置变化、关闭特性、显式和通用 kwargs 的 evaluation | 同一训练函数与配置复用同一个 wrapper/session；evaluation 不创建或推进训练 session |
| `api/session.py` | 进入 / 退出失败、前向中断、规划失败、重试、嵌套、清理失败 | 继续保留既有恢复与拒绝规则；真实设备验证普通和 opaque 异常后可重试 |
| `api/opaque.py` | partial、callable 对象、nn.Module、关键字常量、dict / tuple / list 混合输出 | 不再要求 callable 必须有 `__name__`；数值和梯度与原生计算一致 |
| `execution/base.py` 的 autograd 边界 | detach / 常量 / 混合可导输出、无用 NaN 分支、未使用输入、会话外及旧 step 反向 | 保留非可导标记和 `None` 梯度；旧图不能改写新记录；拒绝 opaque 深度下溢 |
| `execution/base.py` 的归属检查 | 外部 session 的输入、opaque 闭包输出、会话外显式 clone | 不允许外部逻辑 storage ID 混入当前计划；克隆后以普通张量作为输入可用 |
| `execution/tensor.py` | 共轭 / 负视图及组合标志、原地写穿透、空 storage、存活输出 | 值标志在构造和 resolve 时保留；CPU 验证复数共轭，NPU 验证负视图与梯度 |
| `execution/validation.py`、`schema.py` | 值标志、非 strided / meta 张量、既有 shape / dtype / stride / offset / alias、写参数绑定 | 值标志变化不再漏过 guard；继续拒绝不匹配的执行结构 |
| `warmup/executor.py`、`tracker.py` | 原有在线驱逐、别名 / SID、计时异常重置、完整训练重试 | 未放宽不可驱逐工作集的报错；中断计时不会污染下一轮 |
| `ir/trace.py`、`schedule.py`、`replay.py` | 非法 access / size / budget / duration、稀疏编号、同 op 拆分记录、零尺寸、空 trace | 无法解释的 IR 明确报错；合法旧式稀疏 / 拆分记录转为稠密副本，不修改输入 |
| Greedy 规划器 | 40 个固定种子，每组 8 个 storage、32 个 op、多次写、读写同 op、不同预算和保留集合 | 独立版本状态模型验证搬运 / 读写 / 释放顺序；重复 build 结果确定，输入不变 |
| Peak 规划器及 communication masking | 同一组 40 种子分别运行普通 Peak 和 masking；另有原有平台峰值 / 边界 gap 用例 | 三种策略合计 120 组随机场景通过；没有据此声明算法全局最优 |
| 规划器建模 / 选择 / 调度生成 / 日志 / 类型 | retained 生命周期、无访问尾部 op、有限权重、非法 IR、原有 helper 用例 | 保留 storage 计入直到整步末尾，修复后半步驻留量低估；拒绝非有限权重 |
| `runtime/residency.py` | 同 ID 换 storage、bind 各阶段失败、真实 D2H/H2D、空 storage、池销毁 | 旧 Host 快照失效；bind 不留下半成品表项；池销毁后输出仍能读取和修改 |
| `runtime/pinned_memory.py` | 重复 / 跨池 / 未归属回收、4 线程同时借用、既有 bucket / 事件 / 容量场景 | 记录实际所有权与借用状态；阻止两次申请拿到同一块活跃内存 |
| `runtime/timer.py`、`bandwidth.py` | 重复 start、失败 stop、零 / 负 / NaN / inf 时间、探测分配 OOM | 计时器可重置；无效带宽时间或探测 OOM 使用 1 GiB/s 保守值，其他传输错误继续报错 |
| `_patch_accelerator.py`、包导出 | 隔离 fake torch 下的 native / CPU / CUDA / NPU / 无统计接口分支 | 验证兼容层控制流且不修改真实 torch；不等同实际旧版本环境认证 |

## 关键问题与修复依据

1. **工厂重复查询导致每步重新 warmup。** 原测试只覆盖直接调用 `wrap_schedule`。
   新测试经过 factory 包装，并在 NPU 上验证四步训练、evaluation 穿插以及只创建一次真实 session。
2. **opaque 边界改变梯度语义。** 原先 dummy 输入会让 detach / 常量输出被误标为可导；
   自动补零梯度还会执行未使用的 NaN 分支。现在标记非可导输出并保留 `None` 梯度，
   同时验证有效输入梯度与原生结果一致、未使用输入的 `.grad` 仍为 `None`。
3. **值标志和快照错误。** 共轭 / 负视图的标志曾在包装或回载时丢失；
   同一 ID 换新 storage 仍可能恢复旧 Host 数据。定向数值用例覆盖并修复这两种情况。
4. **内存所有权与销毁。** 重复回收可导致活跃借用相互别名；存活输出还会强引用整个 Host 池。
   现在池验证借用状态，输出只弱引用池；仍需使用的实际 Host 数据由输出自身持有。
5. **保留集合建模不足。** retained storage 不能在最后一次算子访问后自动视为已释放。
   两个规划器都将其驻留量延续到整步末尾，并覆盖没有激活访问的尾部 op。
6. **错误路径污染。** bind 构造失败不再留下空表项，计时失败不会保留旧时间戳；
   外部 session 的张量和错误作用域的 opaque 反向会在影响当前计划前拒绝。

原有数值容差和实际显存下降断言保留；Greedy / Peak 的 FP32 / BF16 十步训练继续通过。
两个旧断言作了语义修正：opaque 深度下溢原先期待 `-1`，现在期待报错且保持 `0`；
访问了 storage 却缺少 size 的 trace 原先被忽略，现在明确报错。合法稀疏 IR 的兼容性保留。

## 环境边界与仍未认证的范围

- 独立格式探测的 ND / NCHW 样本搬运前后数值一致；NC1HWC0 / FRACTAL_NZ 样本在当前环境的
  原生格式转换阶段即报 `SetPrecisionMode` / `AclSetCompileopt` 错误，未进入卸载流程。
  不将这两种块格式列为通过，也没有修改共享 CANN 安装来绕过错误。
- 共轭复数测试在 CPU 完成；不能据此声称 910B 复数训练受支持。
- Megatron 适配测试使用替代训练 schedule 和真实 OffloadSession，不是完整 Megatron/FSDP 启动验证。
- 未认证多卡 FSDP / EP / PP、FP8 训练、重计算组合、多设备特殊流或吞吐性能。
- 跨 session 共享逻辑 ShadowTensor 不受支持；应在会话外 `.clone()` 后传入普通张量。
  opaque 反向必须在创建它的同一次 session 作用域内完成。
- storage 重绑 / 改容、自动扩容的 out=、任意动态执行变化仍遵循已有明确拒绝规则。
- 显存预算仍是规划使用的激活驻留预算，不是整卡 HBM 的物理硬上限。

## 复现入口

```shell
python -m pytest tests_extend/unit_tests/mindspeed/core/memory/hyper_offload -q
ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest tests_extend/system_tests/hyper_offload -q
```

核心包覆盖率可使用 coverage.py 的 branch 模式；请将临时报告放在仓库外。
新增单元用例位于 `tests_extend/unit_tests/mindspeed/core/memory/hyper_offload/components/`，
新增硬件组合用例位于 `tests_extend/system_tests/hyper_offload/test_components.py`。

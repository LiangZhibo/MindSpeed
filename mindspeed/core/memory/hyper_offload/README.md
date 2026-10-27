# MindSpeed HyperOffload

Independent eager activation offloading, ported from the team's HyperParallel implementation.
No HyperParallel installation is required. See [the feature guide](../../../../docs/zh/features/hyper-offload.md)
for API/CLI usage, memory-budget semantics, compatibility limits and test commands.

The runtime records model storage accesses while performing online eviction during the first
forward/backward pass. A pluggable planner emits residency actions for subsequent eager passes.
`ReplayExecutor` replays those actions; it is not a device-graph executor.

## Source snapshots

- HyperParallel PR 814: `278789316a91bab7ea215ead9e6236c1fd04a3e2`
- Peak planner PR 986: `6b6f9df79dc22bcdf0e5e6193f1b35c78d726eed`
- Host-buffer event fix PR 1250: `60e65471715e6bb34f500291f17ccc7d576bf001`

Original copyright and Apache-2.0 headers are retained. The port includes MindSpeed package
imports, an opt-in feature and Megatron schedule adapter, argument validation, the event fix,
and portable CPU / real accelerator tests. Baseline and offloaded models have independent
weights, and accelerator memory measurements clear gradients symmetrically.

The first release manages activations only. Parameter/optimizer offload, arbitrary changing
operator sequences, joint recomputation planning and graph capture are outside its support claim.

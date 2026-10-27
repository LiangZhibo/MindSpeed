# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Independent automatic activation offload, disabled by default."""

from mindspeed.features_manager.feature import MindSpeedFeature


class HyperOffloadFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("hyper-offload", optimization_level=2)

    def register_args(self, parser):
        group = parser.add_argument_group(title="HyperOffload")
        group.add_argument("--hyper-offload", action="store_true")
        group.add_argument("--hyper-offload-device-mb", type=int, default=1024)
        group.add_argument("--hyper-offload-host-mb", type=int, default=65536)
        group.add_argument("--hyper-offload-planner", choices=("greedy", "peak"), default="peak")
        group.add_argument("--hyper-offload-communication-masking", action="store_true")

    def validate_args(self, args):
        if not getattr(args, "hyper_offload", False):
            return
        if args.hyper_offload_device_mb < 0 or args.hyper_offload_host_mb <= 0:
            raise ValueError("HyperOffload requires device MiB >= 0 and host MiB > 0")
        if args.hyper_offload_communication_masking and args.hyper_offload_planner != "peak":
            raise ValueError("HyperOffload communication masking requires the peak planner")
        graph_enabled = (
            getattr(args, "cuda_graph_impl", "none") != "none"
            or getattr(args, "npu_graph_mode", "none") != "none"
            or any(getattr(args, key, False) for key in (
                "optimizer_cuda_graph", "enable_cuda_graph", "external_cuda_graph"
            ))
        )
        if graph_enabled:
            raise ValueError("HyperOffload currently supports eager schedules, not device graph capture")
        if getattr(args, "variable_seq_lengths", False):
            raise ValueError("HyperOffload requires a stable visible operator sequence and tensor shapes")
        for name in ("smart_swap", "swap_layer_input", "swap_attention", "compress_activation"):
            if getattr(args, name, False):
                raise ValueError(f"HyperOffload cannot share activation ownership with {name}")

    def register_patches(self, patch_manager, args):
        if not getattr(args, "hyper_offload", False):
            return
        from mindspeed.core.memory.hyper_offload.megatron import get_forward_backward_func_wrapper

        patch_manager.register_patch(
            "megatron.training.training.get_forward_backward_func",
            get_forward_backward_func_wrapper,
        )

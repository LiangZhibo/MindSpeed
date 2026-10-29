#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate training result JSON without requiring PyTorch on the host."""
import argparse
import json
import math
from pathlib import Path


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_result(result, rank, steps, correctness_mode="none"):
    expected_status = "one_step_complete" if steps == 1 else "steps_complete"
    if (result.get("rank") != rank or result.get("world_size") != 2
            or result.get("status") != expected_status or result.get("steps") != steps):
        raise ValueError(f"rank {rank}: incomplete or mismatched training result")
    rows = [dict(loss=result.get("loss"), probe_grad_max=result.get("probe_grad_max"),
                 parameter_update_max=result.get("parameter_update_max"))]
    extra = result.get("extra_step_results", [])
    if len(extra) != steps - 1 or [s.get("index") for s in extra] != list(range(2, steps + 1)):
        raise ValueError(f"rank {rank}: missing or duplicate step records")
    rows.extend(extra)
    for index, row in enumerate(rows, 1):
        number(row.get("loss"), f"rank {rank}, step {index} loss")
        for name in ("probe_grad_max", "parameter_update_max"):
            number(row.get(name), f"rank {rank}, step {index} {name}", positive=True)
    correctness = result.get("correctness", {})
    if correctness.get("mode", "none") != correctness_mode:
        raise ValueError(f"rank {rank}: unexpected correctness mode")
    if correctness_mode != "none":
        if (correctness.get("status") != "complete" or correctness.get("passed") is not True
                or correctness.get("checked_loss_steps") != steps):
            raise ValueError(f"rank {rank}: numerical validation did not complete")
        selected = correctness.get("selected_steps", [])
        if (not selected or len(set(selected)) != len(selected)
                or any(type(s) is not int or s < 1 or s > steps for s in selected)):
            raise ValueError(f"rank {rank}: invalid selected correctness steps")
        checks = correctness.get("checks", [])
        pairs = [(c.get("step"), c.get("kind")) for c in checks]
        expected = {(s, k) for s in selected for k in ("gradient", "parameter")}
        if len(pairs) != len(expected) or set(pairs) != expected:
            raise ValueError(f"rank {rank}: missing numerical tensor comparisons")
        for name in ("expected_tensors", "expected_elements"):
            number(correctness.get(name), name, positive=True)
        for check in checks:
            if (check.get("tensors") != correctness["expected_tensors"]
                    or check.get("elements") != correctness["expected_elements"]):
                raise ValueError(f"rank {rank}: incomplete tensor coverage")
            if (check.get("passed") is not True or check.get("outside_tolerance") != 0
                    or check.get("nonfinite") != 0):
                raise ValueError(f"rank {rank}: numerical comparison failed")
            for name in ("elements", "tensors"):
                number(check.get(name), name, positive=True)
            number(check.get("max_abs_difference"), "max_abs_difference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--correctness-mode", choices=("none", "record", "compare"), default="none")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    for rank in (0, 1):
        result = json.loads((args.directory / f"result_rank{rank}.json").read_text(encoding="utf-8"))
        validate_result(result, rank, args.steps, args.correctness_mode)
    description = "basic finite/nonzero checks" if args.correctness_mode == "none" else f"{args.correctness_mode} numerical validation"
    print(f"verified: both ranks completed {args.steps} steps and passed {description}")


if __name__ == "__main__":
    main()

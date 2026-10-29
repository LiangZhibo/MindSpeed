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
"""Opt-in, streaming numerical validation for the two-rank Qwen experiment.

Reference snapshots contain full local FSDP shards, never sampled values. Only
one parameter and one comparison chunk are materialized on CPU at a time.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def finite_scalar(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be nonzero and positive, got {value!r}")
    return value


def selected_steps(spec, total):
    steps = sorted({1, min(2, total), total}) if spec is None else [int(x) for x in spec.split(",")]
    if not steps or len(set(steps)) != len(steps) or any(s < 1 or s > total for s in steps):
        raise ValueError(f"correctness steps must be unique integers in 1..{total}")
    return sorted(steps)


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        cpu = local_tensor(parameter.detach()).cpu().contiguous()
        digest.update(json.dumps([name, list(parameter.shape), list(cpu.shape), str(cpu.dtype)]).encode())
        digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def compare_tensor(actual, expected, rtol, atol, chunk_elements=1048576):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("reference tensor shape or dtype differs")
    if chunk_elements <= 0:
        raise ValueError("comparison chunk size must be positive")
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    maximum, outside, nonfinite = 0.0, 0, 0
    for start in range(0, actual.numel(), chunk_elements):
        a = actual[start:start + chunk_elements].double()
        b = expected[start:start + chunk_elements].double()
        finite = torch.isfinite(a) & torch.isfinite(b)
        difference = (a - b).abs()
        nonfinite += int((~finite).sum())
        outside += int(((difference > atol + rtol * b.abs()) | ~finite).sum())
        if bool(finite.all()):
            maximum = max(maximum, float(difference.max()))
    return dict(elements=actual.numel(), outside_tolerance=outside,
                nonfinite=nonfinite, max_abs_difference=maximum, passed=outside == 0)


def check_chunk_loss(chunk_loss, calculate_lm_loss):
    """Check the *loaded* loss implementation against independent standard CE.

    CPU FP32 avoids allocating a full-vocabulary logits tensor on the NPU. A
    dedicated generator leaves the experiment's random state unchanged.
    """
    import torch.nn.functional as F
    generator = torch.Generator().manual_seed(7391)
    hidden = torch.randn(2, 17, 8, generator=generator, requires_grad=True)
    weight = torch.randn(19, 8, generator=generator, requires_grad=True)
    labels = torch.randint(19, (2, 17), generator=generator)
    expected = F.cross_entropy(F.linear(hidden, weight).reshape(-1, 19), labels.reshape(-1))
    expected_grad = torch.autograd.grad(expected * 0.37, (hidden, weight))
    denominator = torch.tensor(labels.numel(), dtype=torch.float32)
    kwargs = [dict(shift_labels=part, alpha=denominator, reduction="sum")
              for part in labels.split(6, dim=1)]
    actual = chunk_loss(hidden, weight, None, calculate_lm_loss, kwargs, 6)
    actual_grad = torch.autograd.grad(actual * 0.37, (hidden, weight))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return dict(passed=True, dtype="float32", device="cpu", rtol=1e-5, atol=1e-6)


class NumericalValidator:
    def __init__(self, model, inputs, directory, rank, mode, steps, contract, tolerances):
        if mode not in ("record", "compare"):
            raise ValueError("numerical validation requires record or compare mode")
        for tolerance in tolerances.values():
            for name in ("rtol", "atol"):
                value = finite_scalar(tolerance[name], name)
                if value < 0:
                    raise ValueError("tolerances must be nonnegative")
        self.model, self.mode, self.steps = model, mode, steps
        self.tolerances = tolerances
        self.root = Path(directory) / f"rank{rank}"
        self.losses, self.checks = [], []
        self.expected_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        self.expected_elements = sum(local_tensor(p).numel() for p in model.parameters() if p.requires_grad)
        if mode == "record":
            self.root.mkdir(parents=True, exist_ok=False)
            # DTensor.numel() is global: this estimates storage for *both* ranks.
            snapshot_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
            required = math.ceil(snapshot_bytes * 2 * len(steps) * 1.1)
            if shutil.disk_usage(self.root).free < required:
                raise ValueError(f"insufficient reference disk space: need approximately {required / 2**30:.1f} GiB free")
        self.contract = dict(contract, initial_parameters_sha256=tensor_digest(model),
                             input_sha256=hashlib.sha256(inputs.cpu().contiguous().numpy().tobytes()).hexdigest())
        self.manifest = self.root / "manifest.json"
        if mode == "record":
            self.reference = None
            atomic_json(self.manifest, self._manifest("writing"))
        else:
            self.reference = json.loads(self.manifest.read_text(encoding="utf-8"))
            if (self.reference.get("version") != 1 or self.reference.get("status") != "complete"
                    or self.reference.get("contract") != self.contract
                    or self.reference.get("selected_steps") != steps
                    or len(self.reference.get("losses", [])) != contract["total_steps"]):
                raise ValueError("reference is incomplete or its weights, inputs, settings or selected steps differ")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in self.reference["losses"]):
                raise ValueError("reference contains a non-finite loss")

    def _manifest(self, status):
        return dict(version=1, status=status, contract=self.contract,
                    selected_steps=self.steps, losses=self.losses)

    def _tensor_check(self, step, kind):
        checked, elements, outside, nonfinite, maximum = 0, 0, 0, 0, 0.0
        failures = []
        for index, (name, parameter) in enumerate(self.model.named_parameters()):
            if not parameter.requires_grad:
                continue
            tensor = parameter.grad if kind == "gradient" else parameter.detach()
            if tensor is None:
                raise ValueError(f"step {step}: missing gradient for {name}")
            local = local_tensor(tensor).detach()
            # Scan every tensor on every step, including steps without snapshots.
            if not bool(torch.isfinite(local).all()):
                raise ValueError(f"step {step}: non-finite {kind} in {name}")
            if step not in self.steps:
                continue
            cpu = local.cpu().contiguous()
            path = self.root / f"step{step:04d}" / kind / f"{index:04d}.pt"
            if self.mode == "record":
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".pt.tmp")
                torch.save(dict(name=name, tensor=cpu), temporary)
                temporary.replace(path)
                checked += 1
                elements += cpu.numel()
            else:
                data = torch.load(path, map_location="cpu", weights_only=True)
                if data["name"] != name:
                    raise ValueError(f"reference parameter name differs: {name}")
                comparison = compare_tensor(cpu, data["tensor"], **self.tolerances[kind])
                checked += 1
                elements += comparison["elements"]
                outside += comparison["outside_tolerance"]
                nonfinite += comparison["nonfinite"]
                maximum = max(maximum, comparison["max_abs_difference"])
                if not comparison["passed"]:
                    failures.append(name)
                del data
            del cpu
        if step in self.steps:
            check = dict(step=step, kind=kind, tensors=checked, elements=elements,
                         outside_tolerance=outside, nonfinite=nonfinite,
                         max_abs_difference=maximum, passed=not failures)
            self.checks.append(check)
            if failures:
                raise ValueError(f"step {step}: {kind} differs from reference in {len(failures)} tensors; "
                                 f"max_abs={maximum}, outside_tolerance={outside}; first={failures[0]}")

    def backward(self, step, loss):
        finite_scalar(loss, f"step {step} loss")
        if step != len(self.losses) + 1:
            raise ValueError("correctness steps must be consecutive")
        self.losses.append(loss)
        if self.reference is not None:
            reference = self.reference["losses"][step - 1]
            tolerance = self.tolerances["loss"]
            if not math.isclose(loss, reference, rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]):
                raise ValueError(f"step {step}: loss {loss} differs from reference {reference}")
        self._tensor_check(step, "gradient")

    def update(self, step):
        self._tensor_check(step, "parameter")

    def finish(self):
        expected = {(step, kind) for step in self.steps for kind in ("gradient", "parameter")}
        if (len(self.losses) != self.contract["total_steps"] or len(self.checks) != len(expected)
                or {(c["step"], c["kind"]) for c in self.checks} != expected
                or not all(c["passed"] and c["tensors"] == self.expected_tensors
                           and c["elements"] == self.expected_elements for c in self.checks)):
            raise ValueError("incomplete numerical validation")
        if self.mode == "record":
            atomic_json(self.manifest, self._manifest("complete"))
        return self.summary("complete")

    def summary(self, status):
        return dict(mode=self.mode, status=status, passed=status == "complete",
                    reference_dir=str(self.root.parent), selected_steps=self.steps,
                    tolerances=self.tolerances, checked_loss_steps=len(self.losses),
                    expected_tensors=self.expected_tensors, expected_elements=self.expected_elements,
                    checks=self.checks)

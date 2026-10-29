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
"""CPU regression tests for the Qwen example's production validation helpers."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/qwen25_32b_fsdp2_a2"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


correctness = load("correctness")
verify = load("verify_results")
TOLERANCES = {name: dict(rtol=0.0, atol=0.0) for name in ("loss", "gradient", "parameter")}


def healthy(rank=0):
    return dict(rank=rank, world_size=2, status="steps_complete", steps=3,
                loss=1.0, probe_grad_max=0.1, parameter_update_max=0.01,
                extra_step_results=[dict(index=i, loss=1.0 / i, probe_grad_max=0.1,
                                         parameter_update_max=0.01) for i in (2, 3)])


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("name", ["loss", "probe_grad_max", "parameter_update_max"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, True])
def test_reject_nonfinite_or_invalid_step(step, name, value):
    result = healthy()
    row = result if step == 1 else result["extra_step_results"][step - 2]
    row[name] = value
    with pytest.raises(ValueError):
        verify.validate_result(result, 0, 3)


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("name", ["probe_grad_max", "parameter_update_max"])
def test_reject_zero_probe(step, name):
    result = healthy()
    row = result if step == 1 else result["extra_step_results"][step - 2]
    row[name] = 0.0
    with pytest.raises(ValueError):
        verify.validate_result(result, 0, 3)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(steps=2),
    lambda r: r.update(status="failed"),
    lambda r: r.update(rank=1),
    lambda r: r["extra_step_results"].pop(),
    lambda r: r["extra_step_results"][1].update(index=2),
])
def test_incomplete_results_fail(mutate):
    result = healthy()
    mutate(result)
    with pytest.raises(ValueError):
        verify.validate_result(result, 0, 3)


def test_cli_rejects_later_nan_on_either_rank(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"result_rank{rank}.json").write_text(json.dumps(healthy(rank)))
    command = [sys.executable, str(EXAMPLE / "verify_results.py"), str(tmp_path), "--steps", "3"]
    assert subprocess.run(command, capture_output=True).returncode == 0
    result = healthy(1)
    result["extra_step_results"][1]["loss"] = float("nan")
    (tmp_path / "result_rank1.json").write_text(json.dumps(result))
    assert subprocess.run(command, capture_output=True).returncode != 0


@pytest.fixture
def experiment():
    generator = torch.Generator().manual_seed(314)
    model = torch.nn.Linear(5000, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator) * 0.001)
    inputs = torch.randn(2, 5000, generator=generator)
    return model, inputs


def validator(model, inputs, directory, mode, total=3, steps=None):
    return correctness.NumericalValidator(model, inputs, directory, 0, mode, steps or [1, 2, 3],
                                          dict(total_steps=total, world_size=2, optimizer="SGD"), TOLERANCES)


def train(model, inputs, validation, mutate=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        if mutate:
            mutate(step, model)
        validation.backward(step, float(loss.item()))
        optimizer.step()
        validation.update(step)
    return validation.finish()


def test_full_reference_roundtrip_and_host_verifier(tmp_path, experiment):
    model, inputs = experiment
    reference = validator(copy.deepcopy(model), inputs, tmp_path, "record")
    train(reference.model, inputs, reference)
    candidate = validator(copy.deepcopy(model), inputs, tmp_path, "compare")
    report = train(candidate.model, inputs, candidate)
    assert report["passed"] and len(report["checks"]) == 6
    assert all(c["elements"] == 5000 for c in report["checks"])
    result = healthy()
    result["correctness"] = report
    verify.validate_result(result, 0, 3, "compare")
    for mutate in (
        lambda r: r["correctness"].update(passed=False),
        lambda r: r["correctness"]["checks"].pop(),
        lambda r: r["correctness"]["checks"][0].update(outside_tolerance=1),
        lambda r: r["correctness"]["checks"][0].update(elements=1),
    ):
        bad = copy.deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            verify.validate_result(bad, 0, 3, "compare")


def test_difference_outside_old_4096_element_probe_fails(tmp_path, experiment):
    model, inputs = experiment
    reference = validator(copy.deepcopy(model), inputs, tmp_path, "record")
    train(reference.model, inputs, reference)
    candidate = validator(copy.deepcopy(model), inputs, tmp_path, "compare")
    def corrupt(step, network):
        network.weight.grad.reshape(-1)[4999] += 1.0
    with pytest.raises(ValueError, match="gradient differs"):
        train(candidate.model, inputs, candidate, corrupt)


@pytest.mark.parametrize("change", ["input", "weight", "settings", "incomplete"])
def test_reject_mismatched_reference(tmp_path, experiment, change):
    model, inputs = experiment
    reference = validator(copy.deepcopy(model), inputs, tmp_path, "record")
    train(reference.model, inputs, reference)
    if change == "input": inputs = inputs + 1
    if change == "weight":
        with torch.no_grad(): model.weight[0, 4999] += 0.1
    if change == "incomplete":
        path = tmp_path / "rank0/manifest.json"
        data = json.loads(path.read_text())
        data["status"] = "writing"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reference is incomplete"):
        validator(model, inputs, tmp_path, "compare", total=4 if change == "settings" else 3)


def test_scan_non_snapshot_step_and_reject_partial_record(tmp_path, experiment):
    model, inputs = experiment
    validation = validator(model, inputs, tmp_path, "record", steps=[1, 3])
    def corrupt(step, network):
        if step == 2: network.weight.grad.reshape(-1)[4999] = float("nan")
    with pytest.raises(ValueError, match="non-finite gradient"):
        train(model, inputs, validation, corrupt)
    assert json.loads((tmp_path / "rank0/manifest.json").read_text())["status"] == "writing"


def test_updated_parameters_and_losses_are_compared(tmp_path, experiment):
    model, inputs = experiment
    reference = validator(copy.deepcopy(model), inputs, tmp_path, "record")
    train(reference.model, inputs, reference)
    candidate = validator(copy.deepcopy(model), inputs, tmp_path, "compare")
    loss = candidate.model(inputs).square().mean()
    loss.backward()
    with pytest.raises(ValueError, match="loss .* differs"):
        candidate.backward(1, float(loss.item()) + 0.1)
    candidate = validator(copy.deepcopy(model), inputs, tmp_path, "compare")
    optimizer = torch.optim.SGD(candidate.model.parameters(), lr=0.01)
    loss = candidate.model(inputs).square().mean()
    loss.backward()
    candidate.backward(1, float(loss.item()))
    optimizer.step()
    with torch.no_grad(): candidate.model.weight[0, 4999] += 1
    with pytest.raises(ValueError, match="parameter differs"):
        candidate.update(1)


def test_chunked_comparison_handles_tail_and_nonfinite():
    a = torch.arange(17, dtype=torch.float32)
    assert correctness.compare_tensor(a, a.clone(), 0, 0, chunk_elements=6)["passed"]
    b = a.clone()
    b[-1] = float("nan")
    report = correctness.compare_tensor(a, b, 0, 0, chunk_elements=6)
    assert not report["passed"] and report["nonfinite"] == 1


def test_loss_self_check_rejects_wrong_implementation_and_preserves_rng():
    def calculate(hidden, weight, bias, shift_labels, alpha, reduction):
        logits = torch.nn.functional.linear(hidden, weight)
        return (torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                  shift_labels.reshape(-1), reduction=reduction) / alpha, logits)
    def chunked(hidden, weight, bias, loss, kwargs, size):
        return sum(loss(h, weight, bias, **kw)[0] for h, kw in zip(hidden.split(size, dim=1), kwargs))
    state = torch.random.get_rng_state()
    assert correctness.check_chunk_loss(chunked, calculate)["passed"]
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(AssertionError):
        correctness.check_chunk_loss(lambda *args: 1.1 * chunked(*args), calculate)


@pytest.mark.parametrize("spec", ["0", "4", "1,1", "", "x"])
def test_reject_invalid_snapshot_steps(spec):
    with pytest.raises(ValueError): correctness.selected_steps(spec, 3)


def test_default_steps_cover_warmup_and_replay():
    assert correctness.selected_steps(None, 10) == [1, 2, 10]
    assert correctness.selected_steps(None, 1) == [1]


def test_reference_capture_checks_disk_space(tmp_path, experiment, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(correctness.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="insufficient reference disk space"):
        validator(*experiment, tmp_path, "record")


@pytest.mark.parametrize("invalid", [False, True])
def test_host_launcher_propagates_verification_failure(tmp_path, invalid):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for rank in (0, 1):
        result = healthy(rank)
        result["steps"] = 10
        result["extra_step_results"].extend(dict(index=i, loss=0.5, probe_grad_max=0.1,
                                                 parameter_update_max=0.01) for i in range(4, 11))
        if invalid and rank == 1:
            result["extra_step_results"][-1]["loss"] = float("nan")
        (fixtures / f"result_rank{rank}.json").write_text(json.dumps(result))
    source = (EXAMPLE / "deploy_and_reproduce.sh").read_text()
    function = source[source.index("run_case() {"):source.index('case "$MODE" in')]
    script = """set -euo pipefail
 die() { printf '%s\n' "$*" >&2; exit 1; }
 choose_devices() { :; }
 date() { printf 'fixed-time\\n'; }
 pgrep() { return 1; }
 docker() {
     if [[ "$*" == *"bash -s --"* ]]; then
         cp "$FIXTURES"/result_rank*.json "$out/"
         cat >/dev/null
     fi
 }
""" + function + "\nrun_case test 0 1\n"
    env = dict(os.environ, PROFILE_DIR=str(tmp_path / "profile"), LOG_DIR=str(tmp_path / "logs"),
               FIXTURES=str(fixtures), MODEL_DIR=str(tmp_path / "model"), EXPERIMENT_DIR=str(EXAMPLE),
               CONTAINER_NAME="fake", RUN_ID="fixed", NPU_DEVICES="4,5", BRANCH="test",
               MASTER_PORT="29500", RUN_UID="1010", RUN_GID="1010", USER_HOME=str(tmp_path))
    completed = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert (completed.returncode != 0) == invalid, completed.stdout + completed.stderr
    output = tmp_path / "profile/qwen25-32b-fsdp2-test-fixed"
    status = dict(line.split("=", 1) for line in (output / "exit.txt").read_text().splitlines())
    assert status["training_exit_code"] == "0"
    assert (int(status["verification_exit_code"]) != 0) == invalid
    assert (int(status["exit_code"]) != 0) == invalid

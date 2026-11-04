from copy import deepcopy

import pytest

from mindspeed.core.memory.hyper_offload.tuning.device_windows import DeviceWindowProfile
from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.device_window_search import DeviceWindowNeighborhood


def workload():
    # Labels are intentionally opaque. Both regions compete for one D2H slot.
    sample = {
        "removed_compute": [4.0, 5.0],
        "packets": [
            dict(
                key="a",
                size=8,
                ready=0.0,
                first_pressure=0.4,
                peak_pressure=1.0,
                prefetch=1.0,
                use=2.0,
                d2h=0.6,
                h2d=0.2,
            )
        ],
    }
    one = dict(
        name="blue",
        saved_seconds=0.3,
        workspace=1,
        native_bytes=0,
        first_op=0,
        last_op=0,
        native_until=2,
        samples=[sample],
    )
    two = deepcopy(one)
    two.update(name="orange", saved_seconds=0.2)
    two["samples"][0]["packets"][0]["key"] = "b"
    return dict(
        schema=1,
        anchor_signature="1" * 64,
        anchor_op_count=3,
        resident_mb=1,
        anchor_recompute=["blue", "orange"],
        anchor_resident=[],
        anchor_window_scale=1.0,
        samples=[[[0.0, 2.0]]],
        regions=[one, two],
    )


def phase():
    return dict(
        trace_signature="1" * 64,
        forward_end_op=0,
        truncated=False,
        capacity_summary=dict(width=1, op_count=3, peak_bytes=[2, 2, 2]),
    )


def test_shared_slots_and_first_pressure():
    out = DeviceWindowProfile(workload()).portfolio(phase(), device_limit=10, host_headroom=16)
    assert any(r.offload == ("blue",) for r in out)
    assert all(len(r.offload) <= 1 for r in out)
    assert all(r.mode != "copy_pressure_0.0" for r in out)


def test_removed_compute_and_actual_capacity():
    d = workload()
    d["regions"][0]["samples"][0]["removed_compute"] = [0.0, 1.0]
    out = DeviceWindowProfile(d).portfolio(phase(), device_limit=10, host_headroom=16)
    assert out and all("blue" not in r.offload for r in out)
    assert not DeviceWindowProfile(d).portfolio(phase(), device_limit=2, host_headroom=16)
    assert not DeviceWindowProfile(d).portfolio(phase(), device_limit=10, host_headroom=0)


def test_reject_stale_graph_and_invalid_intervals():
    d = workload()
    p = phase()
    p["trace_signature"] = "2" * 64
    with pytest.raises(ValueError, match="anchor graph"):
        DeviceWindowProfile(d).portfolio(p, device_limit=10, host_headroom=16)
    d["samples"] = [[[0, 1], [0.2, 2]]]
    with pytest.raises(ValueError, match="disjoint"):
        DeviceWindowProfile(d)


def test_version_deduplication_and_native_lifetime():
    d = workload()
    d["regions"][1]["samples"][0]["packets"][0]["key"] = "a"
    out = DeviceWindowProfile(d).portfolio(phase(), device_limit=10, host_headroom=8)
    assert any(len(r.offload) == 2 and r.gross_bytes == 8 for r in out)
    d["regions"][0]["native_bytes"] = 9
    assert all(
        "blue" not in r.resident
        for r in DeviceWindowProfile(d).portfolio(phase(), device_limit=10, host_headroom=8)
    )


def test_composite_parent_and_remainder_share_the_same_peak_limit():
    d = workload()
    d["regions"][0]["name"] = "engine.blue"
    d["anchor_recompute"] = ["engine.blue", "orange"]
    group = dict(d["regions"][0], name="engine", replaces=["engine.blue"], native_bytes=2)
    d["native_groups"] = [group]
    d["collective_free_samples"] = [[[0.0, 0.7], [1.0, 1.3]]]
    profile = DeviceWindowProfile(d)
    out = profile.portfolio(phase(), device_limit=10, host_headroom=16)
    assert any(p.resident == ("engine",) and p.replaced_checkpoints == ("engine.blue",) for p in out)
    assert any(p.resident == ("engine",) and p.offload == ("orange",) for p in out)
    limited = profile.portfolio(phase(), device_limit=4, host_headroom=16)
    assert all("engine" not in p.resident for p in limited)
    policy = JointPolicy(0, 1, recompute_regions=("engine.blue", "orange"))
    evidence = dict(phase_report=phase(), device_limit_bytes=10, host_limit_bytes=16, conflicts=[])
    neighborhood = DeviceWindowNeighborhood(lambda _: evidence, profile)
    first = Trial(policy, samples=[1.0], device_peak_bytes=10)
    option = neighborhood([first])
    move = neighborhood([first, Trial(option, samples=[1.0], device_peak_bytes=10)])
    assert move.resident_regions == ("engine",) and move.recompute_regions == ("orange",)
    d["native_groups"][0]["replaces"] = ["orange"]
    with pytest.raises(ValueError, match="subtree"):
        DeviceWindowProfile(d)


def test_reject_foreign_collective_windows_and_choose_only_matched_schedule():
    d = workload()
    d["calendar_comparison"] = dict(
        trace_signature=d["anchor_signature"], original=[1.0, 1.1], shared=[1.3, 1.2]
    )
    assert not DeviceWindowProfile(d).prefer_shared_calendar
    d["calendar_comparison"]["shared"] = [0.8, 0.9]
    assert DeviceWindowProfile(d).prefer_shared_calendar
    d["collective_free_samples"] = [[[0.1, 2.5]]]
    with pytest.raises(ValueError, match="subset"):
        DeviceWindowProfile(d)

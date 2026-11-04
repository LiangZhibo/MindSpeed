"""Build a graph-bound HyperOffload profile from normalized device event intervals.

Input times are seconds in each rank/step's own clock. The framework adapter
supplies the graph, region lifetimes and version-specific directional costs.
This utility unions overlapping kernel events and subtracts all hardware DMA.
"""
import argparse
import json
from pathlib import Path

from mindspeed.core.memory.hyper_offload.tuning.device_windows import DeviceWindowProfile, _Calendar, _interval


def merge(intervals):
    ordered = sorted(_interval(row) for row in intervals)
    result = []
    for start, end in ordered:
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result



def from_hardware_events(data):
    """Return the validated input consumed by ExplorationOffloadSession."""
    data = dict(data)
    if "samples" in data or "collective_free_samples" in data:
        raise ValueError("provide raw events or ready-made windows, never both")
    data["samples"], data["collective_free_samples"] = [], []
    for sample in data.pop("hardware_samples"):
        free = _Calendar(merge(sample["compute"]), merge(sample["dma"])).slots
        data["samples"].append(free)
        data["collective_free_samples"].append(_Calendar(free, merge(sample["collectives"])).slots)
    DeviceWindowProfile(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = from_hardware_events(json.loads(args.input.read_text()))
    with args.output.open("x") as destination:
        json.dump(data, destination, separators=(",", ":"), allow_nan=False)
        destination.write("\n")


if __name__ == "__main__":
    main()

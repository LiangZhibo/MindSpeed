#!/usr/bin/env python3
"""Run a command on one NUMA node's CPUs with local-preferred memory."""

import argparse
import ctypes
import os
import platform
from pathlib import Path


def parse_cpu_list(value):
    cpus = set()
    for part in value.strip().split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        else:
            cpus.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return cpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--memory-node", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command is required")
    if platform.machine() != "aarch64":
        raise RuntimeError("NUMA syscall numbers in this launcher require aarch64")
    memory_node = args.node if args.memory_node is None else args.memory_node
    if not Path(f"/sys/devices/system/node/node{memory_node}/meminfo").is_file():
        raise ValueError(f"NUMA memory node {memory_node} does not exist")

    cpulist_path = Path(f"/sys/devices/system/node/node{args.node}/cpulist")
    node_cpus = parse_cpu_list(cpulist_path.read_text())
    allowed_cpus = os.sched_getaffinity(0)
    cpus = node_cpus & allowed_cpus
    if not cpus:
        raise RuntimeError(f"no permitted CPUs on NUMA node {args.node}")
    os.sched_setaffinity(0, cpus)

    # aarch64 asm-generic: set_mempolicy=237, get_mempolicy=236.
    # MPOL_PREFERRED=1 tries the device-local node first and permits fallback.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    mask = ctypes.c_ulong(1 << memory_node)
    maxnode = ctypes.sizeof(mask) * 8
    if libc.syscall(237, 1, ctypes.byref(mask), maxnode) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), "set_mempolicy")
    policy = ctypes.c_int()
    actual_mask = ctypes.c_ulong()
    if libc.syscall(236, ctypes.byref(policy), ctypes.byref(actual_mask), maxnode,
                    ctypes.c_void_p(), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), "get_mempolicy")
    if policy.value != 1 or not actual_mask.value & (1 << memory_node):
        raise RuntimeError(f"NUMA memory policy mismatch: mode={policy.value}, mask={actual_mask.value:#x}")

    print(f"numa_cpu_node={args.node} numa_cpu_list={','.join(map(str, sorted(cpus)))} "
          f"numa_memory_policy=preferred numa_memory_node={memory_node}", flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()

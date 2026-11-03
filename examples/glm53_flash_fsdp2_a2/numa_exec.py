#!/usr/bin/env python3
"""Bind the launcher to one NUMA node, then replace it with the training command."""

import ctypes
import os
import sys
from pathlib import Path


def parse_cpulist(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            cpus.update(range(first, last + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("usage: numa_exec.py NODE -- COMMAND [ARGS...]")

    node = int(sys.argv[1])
    cpulist_path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    cpus = parse_cpulist(cpulist_path.read_text())
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    os.sched_setaffinity(0, cpus)

    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    numa.numa_allocate_nodemask.restype = ctypes.c_void_p
    numa.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    numa.numa_set_membind.argtypes = [ctypes.c_void_p]
    numa.numa_set_membind.restype = None
    numa.numa_bitmask_free.argtypes = [ctypes.c_void_p]

    mask = numa.numa_allocate_nodemask()
    if not mask:
        raise OSError(ctypes.get_errno(), "numa_allocate_nodemask failed")
    try:
        numa.numa_bitmask_setbit(mask, node)
        numa.numa_set_membind(mask)
    finally:
        numa.numa_bitmask_free(mask)

    command = sys.argv[3:]
    print(f"[glm53] NUMA node={node}, CPUs={cpulist_path.read_text().strip()}", flush=True)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()

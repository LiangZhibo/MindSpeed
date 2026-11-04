#!/usr/bin/env python3
"""Summarize per-rank baseline and HyperOffload step measurements."""

import argparse
import json
from pathlib import Path
import statistics


def read_metrics(directory: Path) -> dict[int, list[dict]]:
    result = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("rank*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            result[int(path.stem.removeprefix("rank"))] = records
    return result


def group_summary(directory: Path, exclude_warmup: bool) -> dict:
    by_rank = read_metrics(directory)
    if not by_rank:
        return {"status": "no_metrics", "directory": str(directory)}
    all_records = [item for records in by_rank.values() for item in records]
    stable_records = [item for item in all_records if not exclude_warmup or item["iteration"] > 1]
    result = {
        "status": "measured",
        "directory": str(directory),
        "model_shape": all_records[0].get("model_shape", "unknown shape"),
        "ranks": sorted(by_rank),
        "records": len(all_records),
        "steady_records_excluding_iteration_1": len(stable_records),
        "iteration_seconds_mean": statistics.fmean(item["iteration_seconds"] for item in stable_records)
        if stable_records else None,
        "iteration_seconds_median": statistics.median(item["iteration_seconds"] for item in stable_records)
        if stable_records else None,
        "peak_allocated_gib": max(item["npu_peak_allocated_bytes"] for item in all_records) / 1024**3,
        "peak_reserved_gib": max(item["npu_peak_reserved_bytes"] for item in all_records) / 1024**3,
        "per_rank": {},
    }
    for rank, records in by_rank.items():
        steady = [item for item in records if not exclude_warmup or item["iteration"] > 1]
        result["per_rank"][str(rank)] = {
            "records": len(records),
            "iteration_seconds_mean_steady": statistics.fmean(item["iteration_seconds"] for item in steady)
            if steady else None,
            "peak_allocated_gib": max(item["npu_peak_allocated_bytes"] for item in records) / 1024**3,
            "peak_reserved_gib": max(item["npu_peak_reserved_bytes"] for item in records) / 1024**3,
        }
    transfers = [item.get("hyperoffload_transfers", {}) for item in stable_records]
    result["steady_hyperoffload_transfers"] = {
        key: sum(item.get(key, 0) for item in transfers)
        for key in ("d2h_calls", "d2h_bytes", "h2d_calls", "h2d_bytes")
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hyperoffload-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = group_summary(args.baseline_dir, exclude_warmup=True)
    hyperoffload = group_summary(args.hyperoffload_dir, exclude_warmup=True)
    speedup = None
    if baseline.get("iteration_seconds_mean") and hyperoffload.get("iteration_seconds_mean"):
        speedup = baseline["iteration_seconds_mean"] / hyperoffload["iteration_seconds_mean"]
    report = {
        "model": (
            f"GLM-5.3-Flash proportional random-init proxy "
            f"{(baseline.get('model_shape') or hyperoffload.get('model_shape') or 'unknown shape')}"
        ),
        "primary_comparison": "strict full-recompute baseline (stash disabled) vs partial recompute + HyperOffload (stash disabled)",
        "warmup_policy": "iteration 1 excluded from mean/median; peaks include all recorded iterations",
        "baseline": baseline,
        "hyperoffload": hyperoffload,
        "baseline_to_hyperoffload_speedup": speedup,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

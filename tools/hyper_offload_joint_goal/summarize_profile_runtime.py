"""Locate allocator/runtime waits in recorded host call scopes.

This is temporal association, not proof of the cause of a wait. Inclusive
durations may overlap and must not be added to estimate step latency.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def start(event):
    return float(event['ts'])


def end(event):
    return start(event) + float(event['dur'])


parser = argparse.ArgumentParser()
parser.add_argument('profile', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ranks = []
for folder in sorted(args.profile.glob('*/ASCEND_PROFILER_OUTPUT')):
    api = list(csv.DictReader((folder / 'api_statistic.csv').open()))
    api.sort(key=lambda row: float(row['Time(us)']), reverse=True)
    data = json.loads((folder / 'trace_view.json').read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    spans = [event for event in events if event.get('ph') == 'X' and 'dur' in event]
    cpu_by_thread = defaultdict(list)
    runtime_by_thread = defaultdict(list)
    for event in spans:
        if event.get('cat') in ('cpu_op', 'user_annotation') or event['name'].startswith('HyperOffload::'):
            cpu_by_thread[event['tid']].append((start(event), end(event), float(event['dur']), event))
        if event['name'].split('@')[-1].startswith('aclrt'):
            runtime_by_thread[event['tid']].append(event)
    for thread_events in runtime_by_thread.values():
        thread_events.sort(key=start)
    runtime_starts = {thread: [start(event) for event in values]
                      for thread, values in runtime_by_thread.items()}
    # Ascend places CPU operators and runtime APIs on separate virtual process
    # tracks while retaining their real host thread IDs within this rank file.
    syncs = [event for event in spans if event['name'].split('@')[-1] == 'aclrtSynchronizeDeviceWithTimeout']
    associations, totals = [], Counter()
    for event in syncs:
        left, right = start(event), end(event)
        containers = [row[3] for row in sorted((row for row in cpu_by_thread[event['tid']]
                                               if row[0] <= left and row[1] >= right),
                                              key=lambda row: row[2])]
        seconds = float(event['dur']) / 1e6
        totals[containers[0]['name'] if containers else 'no_cpu_op_container'] += seconds
        associations.append(dict(seconds=seconds, thread=event['tid'], timestamp_us=event['ts'],
                                 cpu_containers=[dict(name=cpu['name'], seconds=float(cpu['dur']) / 1e6)
                                                 for cpu in containers[:8]]))
    rows = []
    for step in (event for event in spans if event['name'].startswith('ProfilerStep#')):
        within = [event for event in spans if start(step) <= start(event) < end(step)]
        gathers = Counter(event['name'] for event in within if event['name'].startswith('FSDP::all_gather ('))
        allocators = Counter()
        counts = Counter()
        for event in within:
            name = event['name'].split('@')[-1]
            if name.startswith('aclrt'):
                allocators[name] += float(event['dur']) / 1e6
                counts[name] += 1
        tag_totals, tag_counts = Counter(), Counter()
        tag_runtime, tag_runtime_counts = defaultdict(Counter), defaultdict(Counter)
        for tag in (event for event in within if event['name'].startswith('HyperOffload::')):
            name, thread = tag['name'], tag['tid']
            tag_totals[name] += float(tag['dur']) / 1e6
            tag_counts[name] += 1
            starts = runtime_starts.get(thread, [])
            runtime = runtime_by_thread.get(thread, [])
            left, right = bisect_left(starts, start(tag)), bisect_left(starts, end(tag))
            for api_event in runtime[left:right]:
                if end(api_event) <= end(tag):
                    api_name = api_event['name'].split('@')[-1]
                    tag_runtime[name][api_name] += float(api_event['dur']) / 1e6
                    tag_runtime_counts[name][api_name] += 1
        rows.append(dict(step=step['name'], fsdp_all_gather_scopes=sum(gathers.values()),
                         all_gather_scopes=dict(gathers), runtime_top=[
                             dict(name=name, inclusive_seconds=seconds, count=counts[name])
                             for name, seconds in allocators.most_common(12)],
                         hyperoffload_scopes=[
                             dict(name=name, inclusive_seconds=seconds, count=tag_counts[name],
                                  runtime=[dict(name=api_name, inclusive_seconds=api_seconds,
                                                count=tag_runtime_counts[name][api_name])
                                           for api_name, api_seconds in tag_runtime[name].most_common(10)])
                             for name, seconds in tag_totals.most_common()]))
    ranks.append(dict(profile=folder.parent.name, steps=rows,
                      aggregated_api_top=[dict(name=row['API Name'], level=row['Level'],
                                               inclusive_seconds=float(row['Time(us)']) / 1e6,
                                               count=int(row['Count'])) for row in api[:20]],
                      device_sync_seconds_by_innermost_cpu=dict(totals),
                      longest_device_syncs=sorted(associations, key=lambda row: row['seconds'], reverse=True)[:20]))
    del data, events, spans, cpu_by_thread, syncs
report = dict(caveat='Instrumented diagnostics only. Inclusive times overlap. Same-thread temporal containment '
              'associates a wait with a call site but is not causal critical-path proof. ProfilerStep scopes '
              'also contain setup and control outside the benchmark timer.', ranks=ranks)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps([dict(profile=rank['profile'],
                       steps=[dict(step=step['step'], gathers=step['fsdp_all_gather_scopes']) for step in rank['steps']],
                       device_sync=rank['device_sync_seconds_by_innermost_cpu']) for rank in ranks], indent=2))

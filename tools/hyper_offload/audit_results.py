"""Recompute run, matched-overhead and paired evidence without accepting screens.

Only explicit complete step windows are scored. Missing evidence remains
unverified; a successful script invocation is not itself goal acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def indexed_steps(result, first, last):
    values=result['extra_step_results']
    rows={x['index']:x for x in values}
    if len(rows)!=len(values) or any(i not in rows for i in range(first,last+1)):
        raise ValueError('duplicate or missing explicit training step')
    if any(not math.isfinite(rows[i]['elapsed_seconds']) or rows[i]['elapsed_seconds']<=0
           for i in range(first,last+1)):
        raise ValueError('invalid complete-step duration')
    return rows


def stable_summary(ranks, first=13, last=36):
    indexed=[indexed_steps(result,first,last) for result in ranks]
    values=[max(rows[i]['elapsed_seconds'] for rows in indexed) for i in range(first,last+1)]
    return dict(first_step=first,last_step=last,samples=len(values),maxrank_times=values,
                median_seconds=statistics.median(values),minimum_seconds=min(values),maximum_seconds=max(values))


def first12_wall(ranks):
    values=[]
    for result in ranks:
        rows=indexed_steps(result,2,12)
        if 'session_setup_seconds' not in result or 'cumulative_step_wall_seconds' not in rows[12]:
            return None
        values.append(result['session_setup_seconds']+rows[12]['cumulative_step_wall_seconds'])
    if any(not math.isfinite(value) or value<=0 for value in values):
        raise ValueError('invalid first12 wall duration')
    return max(values)


def argument(launch,name):
    args=launch['argv']
    return args[args.index(name)+1] if name in args else None


def workload(path,launch):
    config=launch['config']
    keys=('model','layers','seq_length','model_init_seed','allocator_fraction','deterministic','device_mb')
    key={k:config.get(k) for k in keys}
    key['layers']=config.get('layers',48);key['seq_length']=config.get('seq_length',4096)
    model_config=config.get('model_config')
    if model_config:
        # Resolve the frozen harness file, including older explicit references.
        remote=Path(model_config)
        tag=remote.parent.parent.name
        local=path.parents[2]/tag/'configs'/remote.name
        if not local.is_file():
            raise ValueError('missing frozen model configuration: '+str(local))
        key['model_config_sha256']=hashlib.sha256(local.read_bytes()).hexdigest()
        runtime=launch.get('model_config_snapshot')
        if runtime is not None and (runtime['sha256']!=key['model_config_sha256'] or runtime['values']!=read(local)):
            raise ValueError('runtime model configuration differs from frozen harness')
        key.pop('model')  # Pretrained path is unused by random-config runs.
    key.update(torch=launch['torch'],torch_npu=launch['torch_npu'],
               device_peak_mb=argument(launch,'--offload-tuning-device-peak-mb'),
               host_mb=argument(launch,'--offload-host-mb'))
    return key


def inspect_run(path):
    path=Path(path).resolve()
    state=read(path/'status.json')
    if state['status']!='complete' or state.get('interference') or state.get('foreign'):
        raise ValueError('incomplete or interfered run: '+str(path))
    before=read(path/'source-before.json');after=read(path/'source-after.json')
    if before!=after or before.get('status')!='verified':
        raise ValueError('source provenance changed')
    launches=[read(path/f'launch_rank{rank}.json') for rank in range(2)]
    if not all(x['performance_usable'] for x in launches):
        raise ValueError('diagnostic/correctness instrumentation cannot score performance')
    ranks=[read(path/f'result_rank{rank}.json') for rank in range(2)]
    last=launches[0]['config']['steps']
    if last<13 or any(d['steps']!=last or d['status']!='steps_complete' for d in ranks):
        raise ValueError('incomplete requested training run')
    if launches[0]['config']!=launches[1]['config']:
        raise ValueError('rank launch configurations differ')
    summaries=[]
    for rank,(result,launch) in enumerate(zip(ranks,launches)):
        tuning=result['joint_tuning'];frozen=tuning['frozen_at_step']
        direct_fixed=launch['config'].get('direct_fixed_policy',False)
        minimum_freeze=0 if direct_fixed else 1
        if direct_fixed and (frozen!=0 or not tuning.get('direct_fixed_policy') or tuning['trials']
                             or tuning['search_seconds'] or tuning['observation_and_consensus_seconds']
                             or not launch['config'].get('fixed_policy')
                             or launch['config']['fixed_policy']!=tuning['policy']):
            raise ValueError('direct fixed control performed tuning or changed its supplied policy')
        rows=indexed_steps(result,2,last)
        if (tuning['phase']!='stable' or isinstance(frozen,bool) or not isinstance(frozen,int)
                or not minimum_freeze<=frozen<=12 or tuning['steps']!=frozen):
            raise ValueError('tuning did not freeze within the hard budget')
        if any(rows[i]['tuning_phase']!='stable' or rows[i]['offload_mode']!='replay'
               or rows[i]['joint_policy']!=tuning['policy'] for i in range(13,last+1)):
            raise ValueError('scored window is not the fixed final replay policy')
        peak=max(result['first_step_npu_peak_allocated_gib'],*(x['npu_peak_allocated_gib'] for x in rows.values()))
        reserved=max(result['first_step_npu_peak_reserved_gib'],*(x['npu_peak_reserved_gib'] for x in rows.values()))
        host=max([result['host_pool_allocated_gib']*2**30,*(t['host_peak_bytes'] for t in tuning['trials'])])
        peak_limit=argument(launch,'--offload-tuning-device-peak-mb')
        host_limit=argument(launch,'--offload-host-mb')
        proposed=path/f'measured_proposal_rank{rank}.json'
        proposal=read(proposed) if proposed.exists() else {}
        phase_file=path/f'phase_memory_rank{rank}.json'
        phase=read(phase_file) if phase_file.exists() else {}
        runtime_file=path/f'runtime_audit_rank{rank}.json'
        runtime=read(runtime_file) if runtime_file.exists() else None
        runtime_ok=(None if runtime is None else runtime['status']=='passed'
                    and runtime['freeze_step']==frozen and runtime['completed_steps']==last
                    and all(runtime['counters_unchanged'].values()) and all(runtime['observers'].values())
                    and runtime['frozen_counters']==runtime['final_counters'])
        summaries.append(dict(rank=rank,frozen_at_step=frozen,policy=tuning['policy'],
            peak_allocated_gib=peak,peak_reserved_gib=reserved,host_pool_peak_gib=host/2**30,
            frozen_runtime_audit_pass=runtime_ok,
            allocator_capacity_pass=None if peak_limit is None else peak*2**30<=int(peak_limit)*2**20,
            host_pool_capacity_pass=None if host_limit is None else host<=int(host_limit)*2**20,
            pure_proposal_seconds=proposal.get('proposal_compute_seconds'),
            proposal_consensus_seconds=proposal.get('proposal_consensus_seconds'),
            controller_search_inclusive_seconds=tuning['search_seconds'],
            search_reference_seconds=.05,search_reference_met=None if 'proposal_compute_seconds' not in proposal else proposal['proposal_compute_seconds']<=.05,
            search_runtime_is_hard_gate=False,
            plan_seconds=tuning['plan_seconds'],plan_budget_pass=tuning['plan_seconds']<=2.,
            switch_seconds=tuning['switch_seconds'],drain_seconds=tuning['switch_drain_seconds'],
            apply_regions_seconds=tuning['apply_regions_seconds'],codec_prepare_seconds=tuning['codec_prepare_seconds'],
            observation_and_consensus_seconds=tuning['observation_and_consensus_seconds'],
            warmup_context_seconds=tuning['warmup_context_seconds'],warmup_finish_seconds=tuning['warmup_finish_seconds'],
            phase_observer_seconds=phase.get('observation_seconds'),timing_events=tuning.get('timing_events'),
            confirmation_target=tuning.get('confirmation_target'),confirmation_fallback=tuning.get('confirmation_fallback')))
    if summaries[0]['policy']!=summaries[1]['policy']:
        raise ValueError('rank final policies differ')
    physical=[]
    for line in (path/'device-memory.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('code')==0 and 'memory_usage' in row:physical.append(row['memory_usage'])
    return dict(path=str(path),source_sha256=before['archive_sha256'],workload=workload(path,launches[0]),
                started=state['started'],finished=state['finished'],
                execution_identity=[state['child_pid'],state['child_start_ticks'],state['started'],state['finished']],
                stable=stable_summary(ranks,last=last),ranks=summaries,
                first12_wall_seconds=first12_wall(ranks),physical_sampled_peak_mib=max(physical) if physical else None,
                trace_capture=launches[0]['config'].get('capture_trace',False),
                transfer_size_calibration=launches[0]['config'].get('transfer_size_calibration',False),
                direct_fixed_policy=launches[0]['config'].get('direct_fixed_policy',False),confirmation=False)


def compare_final_state(a,b):
    for rank in range(2):
        def tensors(path):
            audit=read(Path(path)/f'full_audit_rank{rank}.json')
            if audit['status']!='complete' or audit['rank']!=rank or not audit['tensors']:
                raise ValueError('incomplete final state audit')
            rows=audit['tensors']
            if any(not x[k]['finite'] for x in rows for k in ('parameter','gradient')):
                raise ValueError('nonfinite final state')
            values={x['name']:{k:x[k] for k in ('parameter','gradient')} for x in rows}
            if len(values)!=len(rows):raise ValueError('duplicate parameter identity')
            return values
        if tensors(a)!=tensors(b):return False
    return True



def compare_step_probes(a,b):
    """Compare every recorded scalar probe, not full gradients at every step."""
    keys=('loss','probe_grad_max','parameter_update_max')
    for rank in range(2):
        def probes(path):
            result=read(Path(path)/f'result_rank{rank}.json')
            extra=result['extra_step_results']
            rows=[dict(index=1,**{key:result[key] for key in keys}),*extra]
            indices=[row['index'] for row in rows]
            if len(set(indices))!=len(indices) or set(indices)!=set(range(1,result['steps']+1)):
                raise ValueError('missing or duplicate numerical probe step')
            if any(not math.isfinite(row[key]) for row in rows for key in keys):
                raise ValueError('nonfinite numerical probe')
            return {row['index']:tuple(row[key] for key in keys) for row in rows}
        if probes(a)!=probes(b):return False
    return True


def compare_runs(candidate,reference,*,fixed_final=False):
    a,b=inspect_run(candidate),inspect_run(reference)
    if a['source_sha256']!=b['source_sha256'] or a['workload']!=b['workload']:
        raise ValueError('comparison requires matching source and frozen workload/capacities')
    if a['stable']['last_step']!=b['stable']['last_step']:
        raise ValueError('comparison requires matching complete training lengths')
    if fixed_final and not b.get('direct_fixed_policy',False):
        raise ValueError('overhead control must bypass tuning observations and decision consensus')
    if fixed_final and a['ranks'][0]['policy']!=b['ranks'][0]['policy']:
        raise ValueError('overhead control must execute the identical final policy')
    overhead=None
    if fixed_final and a['first12_wall_seconds'] is not None and b['first12_wall_seconds'] is not None:
        if a['trace_capture']!=b['trace_capture']:raise ValueError('trace instrumentation differs')
        if a.get('transfer_size_calibration',False)!=b.get('transfer_size_calibration',False):raise ValueError('transfer calibration differs')
        overhead=a['first12_wall_seconds']/b['first12_wall_seconds']-1
    return dict(candidate=a,reference=b,stable_ratio=a['stable']['median_seconds']/b['stable']['median_seconds'],
                final_state_exact=compare_final_state(candidate,reference),step_probes_exact=compare_step_probes(candidate,reference),
                first12_extra_fraction=overhead,
                first12_budget_pass=None if overhead is None else overhead<=.1,confirmation=False)


def paired_summary(manifest, goal_protocol=None):
    pairs=manifest['pairs'];order=manifest['order']
    if goal_protocol is None:
        goal_protocol=manifest.get('protocol')
    if not isinstance(goal_protocol,dict):
        raise ValueError('paired confirmation requires an explicit frozen protocol')
    protocol=goal_protocol['statistics']
    if len(pairs)!=protocol['final_pairs'] or order!=protocol['order']:
        raise ValueError('pair count/order differs from the frozen protocol')
    rows=[];identities=set();executions=set();ordered={}
    for i,pair in enumerate(pairs,1):
        row=compare_runs(pair['candidate'],pair['baseline']);rows.append(row)
        for label,role in (('A','reference'),('B','candidate')):
            item=row[role];identity=Path(item['path']).resolve()
            if identity in identities:raise ValueError('reused run is not independent evidence')
            execution=tuple(item['execution_identity'])
            if execution in executions:raise ValueError('copied run is not independent evidence')
            executions.add(execution)
            identities.add(identity);ordered[f'{label}{i}']=item['started']
            if any(r['allocator_capacity_pass'] is not True or r['host_pool_capacity_pass'] is not True for r in item['ranks']):
                raise ValueError('capacity constraints not qualified')
            if any(r.get('frozen_runtime_audit_pass') is not True for r in item['ranks']):
                raise ValueError('confirmation requires frozen runtime counter evidence')
        if (row['candidate']['stable']['last_step']!=protocol['steps_per_run'] or not row['final_state_exact']
                or not row['step_probes_exact']):
            raise ValueError('confirmation requires the frozen step count, exact final state and every-step probes')
    if set(order)!=set(ordered) or [ordered[k] for k in order]!=sorted(ordered.values()):
        raise ValueError('actual independent run order differs from frozen protocol')
    source={row['candidate']['source_sha256'] for row in rows}
    if len(source)!=1:raise ValueError('confirmation source changed across pairs')
    if any(row['candidate']['workload']!=rows[0]['candidate']['workload'] for row in rows[1:]):
        raise ValueError('confirmation workload changed across pairs')
    geometric=lambda values:math.exp(statistics.mean(math.log(x) for x in values))
    speed=geometric([r['candidate']['stable']['median_seconds'] for r in rows])
    ratio=geometric([r['stable_ratio'] for r in rows])
    configured_target=goal_protocol.get('target',{}).get('stable_seconds')
    primary=manifest.get('primary_seconds_target_applies',configured_target is not None)
    if not isinstance(primary,bool):
        raise ValueError('primary target applicability must be boolean')
    target=configured_target if primary else None
    regression=goal_protocol.get('generalization',{}).get('regression_target')
    return dict(pairs=rows,candidate_geomean_seconds=speed,paired_geomean_ratio=ratio,
                primary_seconds_target_applies=primary,stable_seconds_target=target,
                stable_seconds_target_pass=None if target is None else speed<=target,
                matched_regression_target=regression,
                matched_regression_target_pass=None if regression is None else ratio<=1+regression,
                confirmation=True,
                scope='stable speed, frozen runtime counters, every-step scalar probes and exact final state; other goal requirements require their own audit')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate',nargs='?');parser.add_argument('--reference')
    parser.add_argument('--fixed-final',action='store_true');parser.add_argument('--pairs');parser.add_argument('--output')
    parser.add_argument('--protocol',help='Frozen protocol JSON; alternatively embed protocol in the pairs manifest')
    args=parser.parse_args()
    result=(paired_summary(read(args.pairs),read(args.protocol) if args.protocol else None) if args.pairs else
            compare_runs(args.candidate,args.reference,fixed_final=args.fixed_final) if args.reference else inspect_run(args.candidate))
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(payload)
    else:print(payload)

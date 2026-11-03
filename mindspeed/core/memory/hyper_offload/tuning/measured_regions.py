"""Bounded fine-region proposals using warmup costs, with opaque model IDs.

Adapters certify legal regions. This search never reads a model family, layer
count, semantic suffix or tensor-size threshold. Estimates order at most two
experiments; full-step measurements and capacity checks choose the winner.
"""
from __future__ import annotations

from dataclasses import replace
import math
from statistics import median

from .trace_regions import region_costs
from .phase_capacity import PhaseCapacity
from ..ir.transfer import transfer_duration_ms, validate_transfer_samples


def _freeze(value):
    if isinstance(value, tuple):
        try:
            hash(value)
            return value
        except TypeError:
            pass
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(x) for x in value)
    return value


def _queue_window_scale(costs, pool_bytes):
    """Finite queue estimate from byte-weighted dominant activation payloads."""
    if not pool_bytes:
        return 1.0
    # Shared/nested region evidence must not multiply the same transfer.
    versions = {(v.storage_id,v.produced_at):v.nbytes for c in costs for v in c.saved_versions}
    sizes = sorted(versions.values())
    half, cumulative = sum(sizes)/2, 0
    for size in sizes:
        cumulative += size
        if cumulative > half and size > 0:
            return max(1.0,pool_bytes/size)
    return 1.0


def measured_trim_interval(initial, samples, minimum_relative_gain, timing_noise_multiplier):
    """Propose the smallest nontrivial cadence only from observed trim cost.

    The estimated saved time orders a full-step experiment. It is not a speed
    claim and does not discount allocator or physical capacity requirements.
    """
    base = initial.policy
    result = dict(selected_interval=base.trim_replay_interval, samples_seconds=list(samples),
                  estimated_saved_seconds=0.0, required_gain_seconds=None)
    if (not base.trim_before_replay or base.trim_replay_interval != 1 or len(samples) < 2
            or not math.isfinite(initial.score) or initial.score <= 0
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 <= x <= initial.score for x in samples)):
        return result
    center = median(samples)
    spread = median(abs(x-center) for x in samples)
    saved = center / 2
    margin = initial.score * minimum_relative_gain + timing_noise_multiplier * spread
    result.update(estimated_saved_seconds=saved, required_gain_seconds=margin)
    if saved > margin:
        result['selected_interval'] = 2
    return result


class MeasuredRegionNeighborhood:
    """One analysis, then a finite raw-only fine cut and coarse replacement.

    Phase evidence bounds activation lifetimes and an estimated transfer queue.
    A midpoint of the admitted expansion range is measured first. One bounded
    refinement follows its measured direction, jointly changing the compute cut
    and finite overlap window. When measured trim cost warrants an experiment,
    the default second slot compares cadence on the successful first cut. The
    optional joint refinement combines the next admitted cut with that cadence
    and evaluates their complete-step effect within the same two-candidate bound.
    Without phase evidence a conservative fine cut
    is used. Full-step selection is required; unknown structure stays intact.
    """
    def __init__(self, evidence_provider, *, minimum_relative_gain=0.005, timing_noise_multiplier=1.0,
                 joint_refinement=False):
        if (not math.isfinite(minimum_relative_gain) or not 0 <= minimum_relative_gain < 1
                or not math.isfinite(timing_noise_multiplier) or timing_noise_multiplier < 0):
            raise ValueError("invalid measured refinement timing margin")
        if not isinstance(joint_refinement, bool):
            raise ValueError('joint_refinement must be boolean')
        self.joint_refinement = joint_refinement
        self.provider = evidence_provider
        self.minimum_relative_gain = minimum_relative_gain
        self.timing_noise_multiplier = timing_noise_multiplier
        self._pending = None
        self._refinement = None
        self.diagnostics = {}
        self._trim_interval = None
        self._continuation_done = False

    def _candidate(self, base, **changes):
        return replace(base, **changes)

    def __call__(self, trials):
        if not trials or not trials[0].feasible:
            return None
        if self._pending is None:
            self._pending = self._build(trials[0])
        tested = {t.policy for t in trials}
        if self._pending and not self._continuation_done:
            measured = next((t for t in trials if t.policy == self._pending[0]), None)
            if measured is not None:
                self._continuation_done = True
                initial = trials[0]
                finite = math.isfinite(measured.score) and math.isfinite(initial.score)
                margin = (self.minimum_relative_gain * initial.score
                          + self.timing_noise_multiplier * (initial.spread + measured.spread)) if finite else math.inf
                cadence_warranted = (self._trim_interval is not None
                    and self._trim_interval != measured.policy.trim_replay_interval
                    and finite and initial.score - measured.score > margin)
                if self.joint_refinement and self._refinement is not None:
                    # Spend the second slot on the next capacity-admitted cut,
                    # optionally combining the measured cadence opportunity.
                    # Whole-cycle measurement/confirmation evaluates the joint
                    # policy; no isolated cadence benefit is inferred from it.
                    previous_count = len(self._pending)
                    self._refine(initial, measured)
                    if cadence_warranted:
                        if len(self._pending) > previous_count:
                            self._pending[-1] = replace(self._pending[-1],
                                trim_replay_interval=self._trim_interval)
                            self.diagnostics['refinement'].update(axis='regions_and_replay_trim',
                                comparison='joint cut/window/cadence; complete execution cycles')
                        else:
                            self._pending.append(replace(measured.policy,
                                trim_replay_interval=self._trim_interval))
                            self.diagnostics['refinement'].update(axis='replay_trim',
                                comparison='no further admitted cut; same measured cut/window')
                    self.diagnostics['candidate_count'] = len(self._pending)
                elif cadence_warranted:
                    # Establish the raw cut before changing its cadence. A cut's
                    # recomputation gain must not conceal a slower allocator
                    # cadence; compare the second candidate to this exact cut.
                    self._pending = [self._pending[0], replace(measured.policy,
                                       trim_replay_interval=self._trim_interval)]
                    self._refinement = None
                    self.diagnostics['refinement'] = dict(axis='replay_trim',
                        comparison='same measured cut/window; complete execution cycles',
                        initial_gain_seconds=initial.score-measured.score)
                    self.diagnostics['candidate_count'] = 2
                elif self._refinement is not None:
                    self._refine(initial, measured)
        return next((p for p in self._pending if p not in tested), None)

    def _refine(self, initial, measured):
        base, path, midpoint, window_scale = self._refinement
        self._refinement = None  # At most one further experiment, even on failure.
        finite = math.isfinite(measured.score) and math.isfinite(initial.score)
        margin = (self.minimum_relative_gain * initial.score
                  + self.timing_noise_multiplier * (initial.spread + measured.spread)) if finite else None
        improve = finite and initial.score - measured.score > margin
        count = midpoint + 1
        # Search the remaining half in the measured direction. A non-monotonic
        # objective may still have a missed optimum; full-step selection and
        # independent confirmation protect the incumbent, not global optimality.
        next_count = (count + len(path) + 1) // 2 if improve else count // 2
        detail = dict(first_expansions=count, admitted_expansions=len(path),
                      measured_gain_seconds=initial.score-measured.score if finite else None,
                      required_gain_seconds=margin, direction='increase' if improve else 'decrease',
                      next_expansions=next_count)
        self.diagnostics['refinement'] = detail
        if not 0 < next_count <= len(path) or next_count == count:
            return
        recompute, resident = path[next_count-1]
        candidate = self._candidate(base,recompute_regions=recompute,resident_regions=resident,
                            transfer_window_scale=window_scale if base.masking else None)
        if candidate != base and candidate not in self._pending:
            self._pending.append(candidate)
        self.diagnostics['candidate_count'] = len(self._pending)

    def _build(self, initial):
        data = self.provider()
        if data is None:
            return []
        base = initial.policy
        if base.recompute_regions is None or base.codec_key != 'raw':
            return []
        trim = measured_trim_interval(initial, data.get('replay_trim_samples_seconds', ()),
                                      self.minimum_relative_gain, self.timing_noise_multiplier)
        self._trim_interval = trim['selected_interval']
        trace, report = data['trace'], data['report']
        observed_names = [r['name'] for r in report['regions']]
        if len(observed_names) != len(set(observed_names)):
            self.diagnostics = dict(skipped='shared or repeated region calls require aggregate lifetime evidence')
            return []
        costs = region_costs(trace, report)
        if not costs:
            return []
        groups = {}
        def signature(cost):
            return _freeze((cost.structure,cost.input_shapes,cost.saved_bytes,cost.created_bytes,
                            cost.input_boundary_bytes,cost.output_boundary_bytes))
        signatures = {c.name:signature(c) for c in costs}
        for cost in costs:
            groups.setdefault(signatures[cost.name],[]).append(cost.compute_ms)
        medians = {key:median(values) for key,values in groups.items()}
        windows = {}
        for c in costs:
            windows.setdefault(signatures[c.name], []).append(c)
        windows = {key:(median(c.forward_save_window_ms for c in values),
                        median(c.backward_save_window_ms for c in values),
                        max(c.working_set_bytes for c in values)) for key,values in windows.items()}
        costs = tuple(replace(c,compute_ms=medians[signatures[c.name]],
                              forward_save_window_ms=windows[signatures[c.name]][0],
                              backward_save_window_ms=windows[signatures[c.name]][1],
                              working_set_bytes=windows[signatures[c.name]][2]) for c in costs)
        names = frozenset(data['names'])
        conflicts = tuple(frozenset(x) for x in data.get('conflicts', ()))
        movable_inputs = frozenset(data.get('offloaded_checkpoint_inputs', ()))
        capacity = max(0, data['device_limit_bytes'] - initial.device_peak_bytes)
        host_capacity = max(0, data['host_limit_bytes'] - initial.host_peak_bytes)
        phase = None
        phase_error = None
        phase_report = data.get('phase_report')
        if phase_report is not None:
            try:
                phase = PhaseCapacity(phase_report, op_count=len(trace.ops), limit_bytes=data['device_limit_bytes'])
            except (ValueError, KeyError, TypeError) as error:
                phase_error = str(error)
        window_scale = _queue_window_scale(costs,trace.memory_limit_bytes) if phase is not None else 1.0
        transfer_samples = getattr(trace,'transfer_cost_samples',())
        validate_transfer_samples(transfer_samples)
        def transfer_ms(versions, direction):
            bandwidth = trace.d2h_bandwidth_gbps if direction=='d2h' else trace.h2d_bandwidth_gbps
            return sum(transfer_duration_ms(v.nbytes,bandwidth,transfer_samples,direction) for v in versions)
        chosen, resident = set(base.recompute_regions), set(base.resident_regions)
        used = 0
        decisions = []
        # Build ancestry once instead of scanning every selected dotted name
        # for every leaf. Only identity relations are cached, not decisions.
        related = {name:set() for name in names | chosen | resident}
        for name in related:
            parts = name.split('.')
            for index in range(1,len(parts)):
                parent = '.'.join(parts[:index])
                if parent in related:
                    related[parent].add(name)
                    related[name].add(parent)
        blocked = chosen | resident
        for name in chosen | resident:
            blocked.update(related[name])

        def compatible(name, recompute, keep):
            selected = recompute | keep
            if name not in names or name in blocked:
                return False
            combined = selected | {name}
            return not any(pair <= combined for pair in conflicts)

        # Leaf refers to registered observed hierarchy, not operator names.
        ordered = sorted((c for c in costs if c.saved_bytes > 0),key=lambda c:c.name)
        leaves = [c for i,c in enumerate(ordered) if i+1 == len(ordered)
                  or not ordered[i+1].name.startswith(c.name + '.')]
        scored = []
        for cost in leaves:
            if cost.saved_bytes <= 0:
                continue
            transfer = transfer_ms(cost.saved_versions,'d2h') + transfer_ms(cost.saved_versions,'h2d')
            mode = 'recompute' if cost.compute_ms < transfer else 'resident'
            extra = (0 if cost.name in movable_inputs else cost.input_boundary_bytes) if mode == 'recompute' else cost.saved_bytes + cost.input_boundary_bytes + cost.output_boundary_bytes
            # Gain is an upper-bound screening estimate: hidden DMA and dispatch
            # overhead are resolved by the real complete-step measurement.
            gain = transfer - cost.compute_ms if mode == 'recompute' else transfer
            scored.append((gain / max(1, extra), gain, cost, mode, extra))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].first_op))
        for _, gain, cost, mode, extra in scored:
            if gain <= 0 or not compatible(cost.name, chosen, resident):
                continue
            if phase is None:
                if used + extra > capacity:
                    continue
            else:
                last_read = max(v.last_backward_read or v.first_backward_read for v in cost.saved_versions)
                if not phase.try_reserve([(cost.first_op, last_read, extra, None)]):
                    continue
            (chosen if mode == 'recompute' else resident).add(cost.name)
            blocked.add(cost.name)
            blocked.update(related[cost.name])
            used += extra
            decisions.append(dict(name=cost.name, mode=mode, compute_ms=cost.compute_ms,
                                  saved_bytes=cost.saved_bytes, additional_device_bound=extra))
        fine_recompute,fine_resident = tuple(sorted(chosen)),tuple(sorted(resident))
        proposals = []
        # Only remove a coarse checkpoint when a visible structurally matching
        # region supplies its costs. Without phase evidence, reserve all created
        # storage. With it, reserve bounded workspace and native lifetimes.
        visible = {_freeze((c.structure, c.input_shapes)): c for c in costs}
        by_name = {c.name: c for c in costs}
        rows = {r['name']: r for r in report['regions'] if r['complete']}
        expanded = []
        expansion_policies = []
        # Shorter observed activation lifetimes consume less phase capacity.
        # This ordering uses execution positions, never semantic layer IDs.
        def order(name):
            observed = (phase_report or {}).get('regions', {}).get(name, ())
            if len(observed) == 1 and len(observed[0]['backward_ops']) == 1:
                row = observed[0]
                return row['backward_ops'][0] - row['forward_op']
            return len(trace.ops) if hasattr(trace, 'ops') else 0
        for old in sorted(base.recompute_regions, key=order):
            row = rows.get(old)
            if row is None:
                continue
            exemplar = visible.get(_freeze((row['structure'], row['input_shapes'])))
            if exemplar is None or exemplar.name == old:
                continue
            local_recompute = [n for n in chosen if n.startswith(exemplar.name + '.')]
            local_resident = [n for n in resident if n.startswith(exemplar.name + '.')]
            mapping = {n: old + n[len(exemplar.name):] for n in local_recompute + local_resident}
            if not mapping or not set(mapping.values()) <= names:
                continue
            children = [by_name[n] for n in local_recompute + local_resident]
            saved = {(v.storage_id,v.produced_at):v for c in children for v in c.saved_versions}
            remaining_versions = tuple(v for v in exemplar.saved_versions if (v.storage_id,v.produced_at) not in saved)
            if phase is None:
                transfer_cost = transfer_ms(remaining_versions,'d2h') + transfer_ms(remaining_versions,'h2d')
            else:
                # Each expanded coarse scope is disjoint. Its own forward and
                # backward compute budgets can be spent once, not per packet.
                transfer_cost = max(0.0, transfer_ms(remaining_versions,'d2h') - exemplar.forward_save_window_ms)
                transfer_cost += max(0.0, transfer_ms(remaining_versions,'h2d') - exemplar.backward_save_window_ms)
            benefit = exemplar.compute_ms - sum(by_name[n].compute_ms for n in local_recompute) - transfer_cost
            bound = exemplar.created_bytes + exemplar.input_boundary_bytes + exemplar.output_boundary_bytes
            host_bound = exemplar.created_bytes + exemplar.output_boundary_bytes
            if benefit <= 0 or host_bound > host_capacity:
                continue
            if phase is None and used + bound > capacity:
                continue
            candidate_recompute = chosen.difference((old,)).union(mapping[n] for n in local_recompute)
            candidate_resident = resident.union(mapping[n] for n in local_resident)
            if any(pair <= candidate_recompute | candidate_resident for pair in conflicts):
                continue
            combined_names = sorted(candidate_recompute | candidate_resident)
            if any(right.startswith(left + '.') for left,right in zip(combined_names,combined_names[1:])):
                continue
            if phase is not None:
                observed = phase_report.get('regions', {}).get(old, ())
                if len(observed) != 1 or len(observed[0]['backward_ops']) != 1:
                    continue
                first = observed[0]['forward_op']
                last = observed[0]['backward_ops'][0]
                if (not 0 <= first <= last < len(trace.ops)
                        or not trace.ops[first].name.endswith('_fwd')
                        or not trace.ops[last].name.endswith('_bwd')
                        or trace.memory_limit_bytes is None):
                    continue
                native_bound = sum(0 if mapping[n] in movable_inputs else by_name[n].input_boundary_bytes
                                   for n in local_recompute)
                native_bound += sum(by_name[n].saved_bytes + by_name[n].input_boundary_bytes
                                    + by_name[n].output_boundary_bytes for n in local_resident)
                # One shared managed pool plus a full op's inputs/outputs.
                # Native checkpoint/resident allocations add by lifetime.
                # Changed structures still require warmup and capacity checks.
                workspace_bound = trace.memory_limit_bytes + (exemplar.working_set_bytes or exemplar.created_bytes)
                if not phase.try_reserve([(first,last,workspace_bound,'expanded_raw_pool'),
                                          (first,last,native_bound,None)]):
                    continue
                bound = native_bound
            chosen, resident = candidate_recompute, candidate_resident
            used += bound
            host_capacity -= host_bound
            expanded.append(dict(name=old, exemplar=exemplar.name, avoided_estimate_ms=benefit,
                                 additional_device_bound=bound, additional_host_bound=host_bound,
                                 exposed_transfer_estimate_ms=transfer_cost))
            expansion_policies.append((tuple(sorted(chosen)),tuple(sorted(resident))))
        if not (expanded and phase is not None):
            fine = self._candidate(base,recompute_regions=fine_recompute,resident_regions=fine_resident,
                           transfer_window_scale=window_scale if base.masking else None)
            if fine != base:
                proposals.append(fine)
        if expanded:
            if phase is not None:
                midpoint = (len(expansion_policies)-1)//2
                middle_recompute,middle_resident = expansion_policies[midpoint]
                proposals = [self._candidate(base,recompute_regions=middle_recompute,resident_regions=middle_resident,transfer_window_scale=window_scale if base.masking else None)]
                self._refinement = (base,expansion_policies,midpoint,window_scale)
            else:
                candidate = self._candidate(base,recompute_regions=tuple(sorted(chosen)),resident_regions=tuple(sorted(resident)),
                                    transfer_window_scale=None)
                if candidate != base and candidate not in proposals:
                    proposals.append(candidate)
        self.diagnostics = dict(joint_refinement=self.joint_refinement, visible_regions=len(costs), decisions=decisions, expanded=expanded,
                                replay_trim=trim,
                                device_headroom_bytes=capacity, additional_device_bound=used,
                                estimated_transfer_scope='region-local aggregate overlap for expansions; full-step measurement decides',
                                calibrated_transfer_window_scale=window_scale,
                                phase_capacity_used=phase is not None, phase_capacity_error=phase_error,
                                phase_estimated_peak_bytes=phase.estimated_peak_bytes if phase is not None else None,
                                candidate_count=len(proposals))
        return proposals

"""Finite structural exploration interleaved with transport refinements.

Names are opaque adapter identities. Large moves batch legal fine-region
changes; a matching visible scope supplies an estimate for an opaque scope,
never a cached storage ID or a proof of equivalent execution. Every new cut
must collect a fresh trace and pass complete-step capacity/timing admission.

Unlike the legacy neighborhood, a full raw expansion is considered even when
its total DMA cost exceeds saved compute. Local, overlapping windows are not
added up as a transferable budget. Queue forecasts apply only to their own
captured graph and order refinements, not unobserved structural admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from statistics import median

from .measured_regions import MeasuredRegionNeighborhood, _freeze, _queue_window_scale, measured_trim_interval
from .phase_capacity import PhaseCapacity
from .trace_regions import region_costs
from ..ir.transfer import transfer_duration_ms


@dataclass(frozen=True)
class RegionTemplate:
    signature: tuple
    compute_ms: float
    saved_bytes: int
    created_bytes: int
    input_bytes: int
    output_bytes: int
    working_set_bytes: int
    # Relative certified descendants, recompute cost, native input reservation.
    cheap_cut: tuple[tuple[str, float, int], ...] = ()
    # Relative name, recompute time, input reservation, total transfer time,
    # conservative native reservation for keeping this child resident.
    child_observations: tuple[tuple[str, float, int, float, int], ...] = ()


@dataclass(frozen=True)
class Expansion:
    name: str
    children: tuple[str, ...]
    resident_children: tuple[str, ...]
    first_op: int
    last_op: int
    saved_compute_ms: float
    host_bytes: int
    native_bytes: int
    workspace_bytes: int


@dataclass(frozen=True)
class Proposal:
    policy: object
    parent: object
    family: str
    estimated_saved_compute_ms: float
    changed_regions: tuple[str, ...]
    capacity_peak_bytes: int | None = None
    additional_host_bound: int = 0
    structural_mode: str = ''
    refinement_axis: str = ''


class ExplorationNeighborhood:
    """Bounded beam of distinct measured structures, with finite local fronts.

    The controller owns the hard training-step budget and independent final
    confirmation. Exhausting this finite neighborhood is not global
    convergence. Templates pool compute estimates by structure; region
    positions and lifetime windows always remain local to the current graph.
    """
    def __init__(self, evidence_provider, *, beam_width=3, minimum_relative_gain=.005,
                 timing_noise_multiplier=1.0):
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 4:
            raise ValueError('beam_width must be an integer from one to four')
        self.provider, self.beam_width = evidence_provider, beam_width
        self.minimum_relative_gain, self.noise = minimum_relative_gain, timing_noise_multiplier
        # Validate margins through the unchanged legacy seed implementation.
        self._seed = MeasuredRegionNeighborhood(lambda: self._initial_data,
            minimum_relative_gain=minimum_relative_gain, timing_noise_multiplier=timing_noise_multiplier)
        self._initial_data = None
        self._seed_done = False
        self._templates = {}
        self._costs = {}
        self._invalid_graphs = set()
        self._fronts = {}
        self._family_counts = {}
        self._axis_counts = {}
        self._proposed = []
        self._metadata = {}
        self.diagnostics = dict(proposals=self._proposed, graphs=[], exclusions=[],
            convergence='not_established', estimate_scope='proposal ordering; full-step measurement decides')

    def retire_evidence(self):
        """Drop graph-bearing analysis state when the controller freezes."""
        self._initial_data = None
        self._costs.clear()
        self._invalid_graphs.clear()
        self._templates.clear()
        self._fronts.clear()
        self._seed._refinement = None

    @staticmethod
    def _signature(structure, shapes):
        return _freeze((structure, shapes))

    def _ingest(self, data, key):
        if key in self._costs:
            return self._costs[key]
        report, trace = data['report'], data['trace']
        names = [r['name'] for r in report['regions']]
        if len(names) != len(set(names)) or report.get('truncated'):
            self.diagnostics['exclusions'].append(dict(reason='repeated_or_incomplete_regions'))
            self._invalid_graphs.add(key)
            self._costs[key] = ()
            return ()
        costs = region_costs(trace, report)
        compute_groups = {}
        for cost in costs:
            compute_groups.setdefault(self._signature(cost.structure, cost.input_shapes), []).append(cost.compute_ms)
        compute_medians = {key:median(values) for key,values in compute_groups.items()}
        costs = tuple(replace(cost,compute_ms=compute_medians[self._signature(cost.structure,cost.input_shapes)])
                      for cost in costs)
        for c in costs:
            descendants = sorted((x for x in costs if x.name.startswith(c.name + '.') and x.saved_bytes),
                                 key=lambda x: x.name)
            leaves = [x for i, x in enumerate(descendants) if i + 1 == len(descendants)
                      or not descendants[i + 1].name.startswith(x.name + '.')]
            cheap, child_observations = [], []
            for leaf in leaves:
                transfer = sum(transfer_duration_ms(v.nbytes, bandwidth,
                    trace.transfer_cost_samples, direction) for v in leaf.saved_versions
                    for bandwidth, direction in ((trace.d2h_bandwidth_gbps, 'd2h'),
                                                  (trace.h2d_bandwidth_gbps, 'h2d')))
                child_observations.append((leaf.name[len(c.name):],leaf.compute_ms,
                                           leaf.input_boundary_bytes,transfer,
                                           leaf.saved_bytes+leaf.input_boundary_bytes+leaf.output_boundary_bytes))
                # This supplies one alternative bracket, never rejects the
                # full-raw alternative on the basis of total transfer time.
                if 0 < leaf.compute_ms < transfer:
                    cheap.append((leaf.name[len(c.name):], leaf.compute_ms, leaf.input_boundary_bytes))
            signature = self._signature(c.structure, c.input_shapes)
            template = RegionTemplate(signature, c.compute_ms, c.saved_bytes, c.created_bytes,
                c.input_boundary_bytes, c.output_boundary_bytes, c.working_set_bytes, tuple(cheap),
                tuple(child_observations))
            values = self._templates.setdefault(signature, [])
            # Finite recent structural evidence, independent of layer count.
            values.append(template)
            del values[:-64]
        self._costs[key] = costs
        self.diagnostics['graphs'].append(dict(visible_regions=len(costs),
            observed_regions=len(report['regions']), ops=len(trace.ops),
            queue=data.get('queue_summary'), evidence='captured candidate graph'))
        return costs

    def _template(self, row):
        values = self._templates.get(self._signature(row['structure'], row['input_shapes']), ())
        if not values:
            return None
        # Cost noise on one child must not erase every fine alternative. Only
        # common observed descendants are inferred; each child's robust cost
        # decision is independent. Lifetimes are never pooled by structure.
        indexed = [{row[0]:row for row in value.child_observations} for value in values]
        common = set(indexed[0]).intersection(*(set(rows) for rows in indexed[1:]))
        children, observations = [], []
        for suffix in sorted(common):
            observed = [rows[suffix] for rows in indexed]
            compute = median(row[1] for row in observed)
            observations.append((suffix,compute,max(row[2] for row in observed),min(row[3] for row in observed),
                                 max(row[4] for row in observed)))
            if 0 < compute < min(row[3] for row in observed):
                children.append((suffix,compute,max(row[2] for row in observed)))
        return replace(values[0], compute_ms=median(v.compute_ms for v in values),
            saved_bytes=max(v.saved_bytes for v in values), created_bytes=max(v.created_bytes for v in values),
            input_bytes=max(v.input_bytes for v in values), output_bytes=max(v.output_bytes for v in values),
            working_set_bytes=max(v.working_set_bytes for v in values), cheap_cut=tuple(children),
            child_observations=tuple(observations))

    def _retained_cut(self, row, trial, rows, template):
        """Propagate an already measured fine cut across equivalent scopes.

        An isolated transfer/compute threshold can drop a child that makes a
        whole measured policy fast (for example by avoiding dispatch or queue
        pressure). Preserve that combination as another bracket. Equivalence
        only proposes a fresh execution; it never copies storage lifetimes.
        """
        selected = set(trial.policy.recompute_regions)
        resident = set(trial.policy.resident_regions)
        signature = self._signature(row['structure'], row['input_shapes'])
        recipes = {}
        costs = {suffix:(compute,native,keep) for suffix,compute,native,_,keep in template.child_observations}
        for peer in rows.values():
            name = peer['name']
            if (name in selected or name in resident
                    or self._signature(peer['structure'],peer['input_shapes']) != signature):
                continue
            suffixes = tuple(sorted(value[len(name):] for value in selected if value.startswith(name+'.')))
            kept = tuple(sorted(value[len(name):] for value in resident if value.startswith(name+'.')))
            if not (suffixes or kept) or not set(suffixes+kept) <= costs.keys():
                continue
            recipe=(suffixes,kept)
            recipes[recipe] = recipes.get(recipe,0)+1
        if not recipes:
            return (),()
        recipe = min(recipes,key=lambda value:(-recipes[value],sum(costs[suffix][0] for suffix in value[0]),value))
        return (tuple((suffix,*costs[suffix][:2]) for suffix in recipe[0]),
                tuple((suffix,costs[suffix][2]) for suffix in recipe[1]))

    def _expansions(self, trial, data, cheap, *, retained=False, checkpoint_kept=False):
        trace, report = data['trace'], data['report']
        phase = data.get('phase_report') or {}
        rows = {r['name']: r for r in report['regions'] if r['complete']}
        names = set(data['names'])
        movable = set(data.get('offloaded_checkpoint_inputs', ()))
        result = []
        for name in trial.policy.recompute_regions:
            row = rows.get(name)
            observations = phase.get('regions', {}).get(name, ())
            if row is None or len(observations) != 1 or len(observations[0]['backward_ops']) != 1:
                continue
            template = self._template(row)
            if template is None:
                continue
            first, last = observations[0]['forward_op'], observations[0]['backward_ops'][0]
            if not (0 <= first <= last < len(trace.ops) and trace.memory_limit_bytes is not None
                    and trace.ops[first].name.endswith('_fwd') and trace.ops[last].name.endswith('_bwd')):
                continue
            cut,kept = self._retained_cut(row,trial,rows,template) if retained else (template.cheap_cut if cheap else (),())
            if checkpoint_kept:
                if not kept:
                    continue
                observed = {suffix:(compute,native) for suffix,compute,native,_,_ in template.child_observations}
                cut = (*cut, *((suffix,*observed[suffix]) for suffix,_ in kept))
                kept = ()
            if (cheap or retained) and not (cut or kept):
                continue
            children = tuple(name + suffix for suffix, _, _ in cut)
            kept_children = tuple(name + suffix for suffix,_ in kept)
            if not set(children+kept_children) <= names:
                continue
            saved = template.compute_ms - sum(ms for _, ms, _ in cut)
            if saved <= 0:
                continue
            native = sum(size for (suffix, _, size) in cut if name + suffix not in movable)
            native += sum(size for _,size in kept)
            result.append(Expansion(name, children, kept_children, first, last, saved,
                template.created_bytes + template.output_bytes, native,
                trace.memory_limit_bytes + max(template.working_set_bytes, template.input_bytes + template.output_bytes)))
        return result

    @staticmethod
    def _legal(recompute, resident, conflicts):
        selected = recompute | resident
        for name in selected:
            parts = name.split('.')
            if any('.'.join(parts[:i]) in selected for i in range(1, len(parts))):
                return False
        return not any(set(pair) <= selected for pair in conflicts)

    def _expansion_front(self, trial, data, cheap, order, *, retained=False, checkpoint_kept=False):
        phase_report = data.get('phase_report')
        key = trial.policy.structure_key
        if phase_report is None or key not in self._costs or key in self._invalid_graphs:
            return []
        try:
            capacity = PhaseCapacity(phase_report, op_count=len(data['trace'].ops),
                                     limit_bytes=data['device_limit_bytes'])
        except (ValueError, KeyError, TypeError) as error:
            self.diagnostics['exclusions'].append(dict(reason='phase_capacity', detail=str(error)))
            return []
        changes = self._expansions(trial, data, cheap, retained=retained,checkpoint_kept=checkpoint_kept)
        if order == 'early':
            changes.sort(key=lambda x: (-(x.last_op - x.first_op), x.first_op, -x.saved_compute_ms))
        else:
            changes.sort(key=lambda x: (-x.saved_compute_ms / max(1, x.host_bytes), x.first_op))
        recompute, resident = set(trial.policy.recompute_regions), set(trial.policy.resident_regions)
        host_left = max(0, data['host_limit_bytes'] - trial.host_peak_bytes)
        paths, accepted, densities, saved, added_host = [], [], [], 0.0, 0
        # A useful measured cut may hide every registered region's interior.
        # Historical structural templates can still propose its expansion;
        # current phase lifetimes/capacity remain mandatory. Do not mistake
        # an opaque-only graph for missing or invalid observations.
        scale = trial.policy.transfer_window_scale
        if scale is None and self._costs[key]:
            scale = _queue_window_scale(self._costs[key], data['trace'].memory_limit_bytes)
        for change in changes:
            proposed = recompute.difference((change.name,)).union(change.children)
            proposed_resident = resident.union(change.resident_children)
            if not self._legal(proposed, proposed_resident, data.get('conflicts', ())):
                continue
            if change.host_bytes > host_left or not capacity.try_reserve([
                    (change.first_op, change.last_op, change.workspace_bytes, 'expanded_managed_pool'),
                    (change.first_op, change.last_op, change.native_bytes, None)]):
                continue
            recompute = proposed
            resident = proposed_resident
            host_left -= change.host_bytes
            accepted.append(change.name)
            densities.append(change.saved_compute_ms / max(1,change.host_bytes))
            saved += change.saved_compute_ms
            added_host += change.host_bytes
            policy = replace(trial.policy, recompute_regions=tuple(sorted(recompute)),
                             resident_regions=tuple(sorted(resident)),
                             transfer_window_scale=scale if trial.policy.masking else None, codec_key='raw')
            mode = 'checkpoint' if checkpoint_kept else 'retained' if retained else 'cheap' if cheap else 'raw'
            paths.append(Proposal(policy, trial.policy, 'expand_'+mode,
                                  saved, tuple(accepted), capacity.estimated_peak_bytes,added_host,
                                  mode))
        if not paths:
            return []
        # Test the admitted endpoint, a large midpoint, and a single local
        # move. The endpoint is not silently replaced by a conservative half.
        points = sorted({len(paths) - 1, (len(paths) - 1) // 2, 0}, reverse=True)
        result = []
        if not cheap and not retained and order == 'value' and len(paths) > 1:
            knee = max(range(len(paths)-1),key=lambda i:densities[i]/max(densities[i+1],1e-30))
            if densities[knee] > densities[knee+1]*(1.+self.minimum_relative_gain):
                result.append(replace(paths[knee],family='expand_value'))
        for i in points:
            proposal = paths[i]
            if i == 0 and len(paths) > 1:
                proposal = replace(proposal, family='refine_regions')
            result.append(proposal)
        return result

    def _refinements(self, trial, data):
        policy, result = trial.policy, []
        # Destination allocator stream fences exist in both allocation modes.
        # Test release and allocation separately and together, in both
        # directions: a new graph can reverse an old graph's best setting.
        result.append(Proposal(replace(policy,defer_h2d_release=not policy.defer_h2d_release),
                               policy,'schedule',0.,(),refinement_axis='defer_release'))
        result.append(Proposal(replace(policy,prefetch_on_copy_stream=not policy.prefetch_on_copy_stream),
                               policy,'schedule',0.,(),refinement_axis='prefetch_allocation'))
        result.append(Proposal(replace(policy,prefetch_on_copy_stream=not policy.prefetch_on_copy_stream,
                                       defer_h2d_release=not policy.defer_h2d_release),
                               policy,'schedule',0.,(),refinement_axis='release_and_allocation'))
        if policy.masking:
            scale = policy.transfer_window_scale
            if scale is None:
                scale = _queue_window_scale(self._costs[policy.structure_key], data['trace'].memory_limit_bytes)
            # Both directions and unrestricted deadlines are finite measured
            # alternatives; the forecast does not certify runtime hiding.
            for axis,value in (('window_half',scale/2),('window_double',scale*2),('window_unlimited',None)):
                if value != policy.transfer_window_scale:
                    result.append(Proposal(replace(policy, transfer_window_scale=value), policy, 'schedule', 0., (),
                                           refinement_axis=axis))
            if policy.limit_prefetch_window:
                result.append(Proposal(replace(policy, limit_prefetch_window=False), policy, 'schedule', 0., (),
                                       refinement_axis='prefetch_window_unlimited'))
        trim = measured_trim_interval(trial, data.get('replay_trim_samples_seconds', ()),
                                     self.minimum_relative_gain, self.noise)
        if trim['selected_interval'] != policy.trim_replay_interval:
            result.append(Proposal(replace(policy, trim_replay_interval=trim['selected_interval']),
                                   policy, 'trim', 1000 * trim['estimated_saved_seconds'], ()))
        # Codec candidates are registered only after actual local calibration
        # and cross-rank common support; unavailable profiles remain explicit.
        for key in data.get('codec_candidates', ()):
            if key != policy.codec_key:
                result.append(Proposal(replace(policy, codec_key=key), policy, 'compression', 0., ()))
        return result

    def _resident_front(self, trial, data, *, recompute):
        """Reconsider keep decisions instead of inheriting them forever.

        Resident scope internals have no measured local backward lifetime.
        Reserve new checkpoint inputs across the ENTIRE current step and all
        created native workspace for one recomputation. No old resident bytes
        are credited away. Raw expansion instead reserves a managed pool and
        an operator working set. A fresh graph and complete step still decide.
        """
        key = trial.policy.structure_key
        if (key in self._invalid_graphs or data.get('phase_report') is None
                or not recompute and data['trace'].memory_limit_bytes is None):
            return []
        try:
            capacity = PhaseCapacity(data['phase_report'],op_count=len(data['trace'].ops),
                                     limit_bytes=data['device_limit_bytes'])
        except (ValueError,KeyError,TypeError):
            return []
        rows = {row['name']:row for row in data['report']['regions'] if row['complete']}
        choices=[]
        for name in trial.policy.resident_regions:
            row=rows.get(name)
            template=self._template(row) if row is not None else None
            if template is not None and name in data['names']:
                choices.append((name,template,row['first_op'] if 'first_op' in row else 0))
        choices.sort(key=lambda row:(-row[1].saved_bytes/max(1,row[1].input_bytes),row[2]))
        selected,kept=set(trial.policy.recompute_regions),set(trial.policy.resident_regions)
        movable=set(data.get('offloaded_checkpoint_inputs',()))
        host_left=max(0,data['host_limit_bytes']-trial.host_peak_bytes)
        accepted,paths,extra_compute,added_host=[],[],0.,0
        last=len(data['trace'].ops)-1
        for name,template,_ in choices:
            new_selected=selected|{name} if recompute else selected
            new_kept=kept-{name}
            if not self._legal(new_selected,new_kept,data.get('conflicts',())):
                continue
            if recompute:
                native=0 if name in movable else template.input_bytes
                host=template.input_bytes if name in movable else 0
                workspace=max(template.working_set_bytes,
                              template.created_bytes+template.input_bytes+template.output_bytes)
            else:
                native=0;host=template.created_bytes+template.output_bytes
                workspace=(data['trace'].memory_limit_bytes or 0)+max(template.working_set_bytes,
                    template.input_bytes+template.output_bytes)
            if host>host_left or not capacity.try_reserve([
                    (0,last,native,None),(0,last,workspace,'resident_mode_workspace')]):
                continue
            selected,kept=new_selected,new_kept
            host_left-=host;added_host+=host
            extra_compute+=template.compute_ms if recompute else 0.
            accepted.append(name)
            policy=replace(trial.policy,recompute_regions=tuple(sorted(selected)),
                           resident_regions=tuple(sorted(kept)),codec_key='raw')
            family='resident_recompute' if recompute else 'resident_raw'
            paths.append(Proposal(policy,trial.policy,family,-extra_compute,tuple(accepted),
                                  capacity.estimated_peak_bytes,added_host,family))
        points=sorted({len(paths)-1,(len(paths)-1)//2,0},reverse=True) if paths else []
        return [replace(paths[i],family='refine_regions') if i==0 and len(paths)>1 else paths[i] for i in points]

    def __call__(self, trials):
        if not trials or not trials[0].feasible:
            return None
        available = []
        for trial in trials:
            if not math.isfinite(trial.score):
                continue
            data = self.provider(trial.policy)
            if data is not None:
                self._ingest(data, trial.policy.structure_key)
                available.append((trial, data))
        if not self._seed_done:
            self._seed_done = True
            self._initial_data = self.provider(trials[0].policy)
            if self._initial_data is not None:
                seed = self._seed((trials[0],))
                if seed is not None and seed not in {t.policy for t in trials}:
                    return self._record(Proposal(seed, trials[0].policy, 'legacy_seed', 0., ()))
        # Distinct structures preserve alternatives instead of allowing many
        # timing knobs on one graph to consume all beam positions.
        beam, seen = [], set()
        for trial, data in sorted(available, key=lambda x: x[0].score):
            if trial.policy.structure_key in seen:
                continue
            beam.append((trial, data)); seen.add(trial.policy.structure_key)
            if len(beam) >= self.beam_width:
                break
        tested = {t.policy for t in trials}
        candidates = {}
        for trial, data in beam:
            if trial.policy not in self._fronts:
                front = []
                for recompute in (True,False):
                    front.extend(self._resident_front(trial,data,recompute=recompute))
                for order in ('early','value'):
                    front.extend(self._expansion_front(trial,data,False,order,retained=True))
                    front.extend(self._expansion_front(trial,data,False,order,retained=True,checkpoint_kept=True))
                for cheap in (False, True):
                    for order in ('early', 'value'):
                        front.extend(self._expansion_front(trial, data, cheap, order))
                self._fronts[trial.policy] = front
            refinements=self._refinements(trial,data)
            if beam and trial.score > beam[0][0].score*(1.+self.minimum_relative_gain) + self.noise*(trial.spread+beam[0][0].spread):
                # A tiny cache cadence saving on a clearly losing parent must
                # not consume a slot intended for structural exploration.
                refinements=[p for p in refinements if p.family!='trim']
            for p in self._fronts[trial.policy] + refinements:
                if p.policy not in tested:
                    candidates.setdefault(p.policy, (p, trial.score))
        # Interleave broad structural moves with local and transport feedback.
        # Family fairness prevents an optimistic compute estimate from using
        # every experiment on the same cut or one transport-only axis.
        families = ('resident_recompute', 'expand_retained', 'expand_checkpoint', 'resident_raw', 'expand_value', 'expand_cheap', 'schedule', 'expand_raw', 'refine_regions', 'compression', 'trim')
        present = {p.family for p, _ in candidates.values()}
        family = min((f for f in families if f in present),
                     key=lambda f: (self._family_counts.get(f, 0), families.index(f)), default=None)
        if family is None:
            self.diagnostics['frontier_status'] = 'finite_neighborhood_exhausted_not_global_convergence'
            return None
        pool = [(p, score) for p, score in candidates.values() if p.family == family]
        if family == 'schedule':
            # A new winning trim cadence must not restart the same knob's
            # exploration and starve other synchronization/queue axes.
            count = min(self._axis_counts.get(p.refinement_axis,0) for p,_ in pool)
            pool = [(p,score) for p,score in pool if self._axis_counts.get(p.refinement_axis,0)==count]
        scores={trial.policy:trial.score for trial in trials}
        def predict(item):
            proposal,parent_score=item
            residuals=[]
            for trial in trials:
                old=self._metadata.get(trial.policy)
                if (old is None or not proposal.structural_mode or old.structural_mode!=proposal.structural_mode
                        or old.additional_host_bound<=0 or not math.isfinite(trial.score)
                        or not math.isfinite(scores.get(old.parent,math.inf))):
                    continue
                residuals.append(max(0.,trial.score-scores[old.parent]+old.estimated_saved_compute_ms/1000)
                                 /old.additional_host_bound)
            # Empirical residual includes dispatch, DMA exposure and model
            # error. This byte-normalized heuristic is not causal attribution.
            penalty=(median(residuals)*proposal.additional_host_bound) if residuals else 0.
            return parent_score-proposal.estimated_saved_compute_ms/1000+penalty
        if family == 'resident_recompute':
            # The immediate extra compute cannot price the capacity released
            # for a later expansion. Test a broad admitted endpoint first.
            chosen,parent_score=min(pool,key=lambda item:(item[1],-len(item[0].changed_regions)))
        else:
            chosen, parent_score = min(pool, key=lambda item:(predict(item),-len(item[0].changed_regions)))
        self.diagnostics['last_predicted_seconds']=predict((chosen,parent_score))
        self.diagnostics['remaining_frontier'] = len(candidates) - 1
        return self._record(chosen)

    def _record(self, proposal):
        self._metadata[proposal.policy]=proposal
        self._family_counts[proposal.family] = self._family_counts.get(proposal.family, 0) + 1
        if proposal.refinement_axis:
            self._axis_counts[proposal.refinement_axis] = self._axis_counts.get(proposal.refinement_axis,0)+1
        self._proposed.append(dict(family=proposal.family, changed_regions=proposal.changed_regions,
            parent_policy=asdict(proposal.parent),
            refinement_axis=proposal.refinement_axis,
            estimated_saved_compute_ms=proposal.estimated_saved_compute_ms,
            estimated_capacity_peak_bytes=proposal.capacity_peak_bytes,
            additional_host_bound=proposal.additional_host_bound,structural_mode=proposal.structural_mode,
            recompute_regions=proposal.policy.recompute_regions,
            resident_regions=proposal.policy.resident_regions, codec=proposal.policy.codec_key,
            transport_window_scale=proposal.policy.transfer_window_scale,
            prefetch_on_copy_stream=proposal.policy.prefetch_on_copy_stream,
            defer_h2d_release=proposal.policy.defer_h2d_release,
            requires_fresh_trace=proposal.parent.structure_key != proposal.policy.structure_key))
        return proposal.policy

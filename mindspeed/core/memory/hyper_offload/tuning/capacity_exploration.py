"""Global copy-budget exploration, followed by fine-region allocation.

The initial move opens a large capacity-admitted graph. Even a slow expanded
graph remains useful as a source of visible storage-version and compute costs.
Subsequent candidates allocate a WHOLE-STEP copy budget; they do not walk the
previous winning cut one region at a time. Changed cuts always require a new
actual capture and the ordinary peak planner chooses their runtime transfers.
"""
from dataclasses import asdict, replace
import math

from .exploration_regions import ExplorationNeighborhood, Proposal
from .global_capacity import estimate_forward_backward_capacity
from .measured_regions import _queue_window_scale
from .peak_windows import PeakWindowModel
from .phase_capacity import PhaseCapacity
from ..ir.transfer import transfer_duration_ms


class CapacityFirstNeighborhood(ExplorationNeighborhood):
    """Open the graph, bracket its global copy budget, then refine the winner.

    The shared windows are estimated BEFORE assigning regions. Reference node
    IDs are used only to price candidates, never installed in another graph's
    runtime. Missing structural evidence never means zero cost. A finite
    budget can end exploration without establishing convergence.
    """
    def __init__(self, evidence_provider, **kwargs):
        super().__init__(evidence_provider, **kwargs)
        self._seed_done = True  # The old measured fine cut is not our seed.
        self._capacity_seed_done = False
        self._references = {}
        self._budget_policies = {}
        self._attempted_fractions = {}
        self._reopen_proposals = {}
        self._reopened_structures = set()
        self._observed_scopes = set()
        self._round = 0
        self.diagnostics.update(algorithm='global_capacity_first', global_budgets=[],
                                allocations=[], initial_move='large_offload_seed')

    def retire_evidence(self):
        super().retire_evidence()
        self._references.clear()
        self._budget_policies.clear()
        self._attempted_fractions.clear()
        self._reopen_proposals.clear()
        self._reopened_structures.clear()
        self._observed_scopes.clear()

    def _ingest(self, data, key):
        known=key in self._costs
        costs=super()._ingest(data,key)
        if not known:
            self._observed_scopes.update((c.name,self._signature(c.structure,c.input_shapes)) for c in costs)
        return costs

    def _capacity(self, data, *, removed=0.):
        phase = data.get('phase_report')
        if phase is None:
            return None
        try:
            return estimate_forward_backward_capacity(data['trace'], phase['forward_end_op'],
                host_limit_bytes=data['host_limit_bytes'], removed_backward_compute_ms=removed)
        except (ValueError, TypeError, KeyError) as error:
            self.diagnostics['exclusions'].append(dict(reason='global_capacity', detail=str(error)))
            return None

    @staticmethod
    def _load(capacity, dms, hms):
        def fraction(value, limit):
            return value/limit if limit else (0. if value == 0 else math.inf)
        return max(fraction(dms, capacity.d2h_available_ms),
                   fraction(hms, capacity.h2d_available_ms),
                   fraction(dms+hms, capacity.shared_server_ms))

    def _large_seed(self, trial, data, *, parents_only=False):
        """Spend one global budget on a broad raw expansion, not a fine seed."""
        if trial.policy.structure_key in self._invalid_graphs:
            return None
        changes = self._expansions(trial, data, False)
        if parents_only:
            # A capacity-limited initial expansion must not permanently lock
            # its remaining opaque parents. A later global cut can free copy
            # capacity for opening all of these scopes in another large move.
            rows={row['name']:row for row in data['report']['regions']}
            changes = [c for c in changes if any(n.startswith(c.name+'.') for n in data['names'])
                and (c.name,self._signature(rows[c.name]['structure'],rows[c.name]['input_shapes']))
                not in self._observed_scopes]
        if not changes:
            return None
        capacity = self._capacity(data, removed=sum(c.saved_compute_ms for c in changes))
        if capacity is None:
            return None
        try:
            memory = PhaseCapacity(data['phase_report'], op_count=len(data['trace'].ops),
                                   limit_bytes=data['device_limit_bytes'])
        except (ValueError, TypeError, KeyError):
            return None
        reference = self._references.get(trial.policy.structure_key)
        if reference is not None:
            packets = tuple(reference['packets'].values())
        else:
            model = data.get('queue_model') or PeakWindowModel(data['trace'])
            packets = model.default_selection()
        dms, hms = sum(p.d2h_ms for p in packets), sum(p.h2d_ms for p in packets)
        volume = sum(p.candidate.size_bytes for p in packets)
        rows = {r['name']:r for r in data['report']['regions']}
        known = [(c, self._template(rows[c.name])) for c in changes]
        # For equal value, early long-lived activations retain the chance to
        # use a broad window. This is based on positions, never model names.
        known.sort(key=lambda item: (-item[0].saved_compute_ms/max(1,item[1].saved_bytes), item[0].first_op))
        selected, changed, extra_host, saved = set(trial.policy.recompute_regions), [], 0, 0.
        for change, template in known:
            added = template.saved_bytes
            if added <= 0:
                continue
            down = transfer_duration_ms(added, data['trace'].d2h_bandwidth_gbps,
                                        data['trace'].transfer_cost_samples, 'd2h')
            up = transfer_duration_ms(added, data['trace'].h2d_bandwidth_gbps,
                                      data['trace'].transfer_cost_samples, 'h2d')
            if (self._load(capacity,dms+down,hms+up)>1.
                    or trial.host_peak_bytes+extra_host+change.host_bytes>data['host_limit_bytes']):
                continue
            if not memory.try_reserve([(change.first_op,change.last_op,change.workspace_bytes,'global_seed_workspace')]):
                continue
            selected.remove(change.name)
            changed.append(change.name)
            extra_host += change.host_bytes
            saved += change.saved_compute_ms
            dms += down; hms += up; volume += added
        stage = 'reopen_opaque_parents' if parents_only else 'initial_seed'
        self.diagnostics['global_budgets'].append(dict(stage=stage,capacity=asdict(capacity),
            estimated_transfer_bytes=volume, changed_regions=len(changed),
            estimated_d2h_ms=dms, estimated_h2d_ms=hms))
        if not changed:
            return None
        scale = trial.policy.transfer_window_scale
        if scale is None and self._costs[trial.policy.structure_key]:
            scale = _queue_window_scale(self._costs[trial.policy.structure_key],data['trace'].memory_limit_bytes)
        policy = replace(trial.policy,recompute_regions=tuple(sorted(selected)),codec_key='raw',
                         transfer_window_scale=scale if trial.policy.masking else None)
        family = 'global_capacity_reopen' if parents_only else 'global_capacity_seed'
        return Proposal(policy,trial.policy,family,saved,tuple(changed),
                        memory.estimated_peak_bytes,extra_host,stage)

    def _reference(self, trial, data):
        key = trial.policy.structure_key
        if key in self._references:
            return self._references[key]
        capacity = self._capacity(data)
        costs = self._costs.get(key, ())
        if capacity is None or not costs or key in self._invalid_graphs:
            self._references[key] = None
            return None
        # Global capacity already exists before the concrete packet/region
        # allocation below. Packet latencies consume that shared time budget.
        model = data.get('queue_model') or PeakWindowModel(data['trace'])
        packets = model.default_selection()
        packet_map = {(p.candidate.storage_id,p.candidate.copy_start.op_id):p for p in packets}
        if len(packet_map)!=len(packets):
            self._references[key] = None
            return None
        base_selected = set(trial.policy.recompute_regions)
        base_kept = set(trial.policy.resident_regions)
        rows = {r['name']:r for r in data['report']['regions']}
        candidates = []
        movable = set(data.get('offloaded_checkpoint_inputs',()))
        for cost in costs:
            if cost.name not in data['names'] or cost.name in base_selected|base_kept:
                continue
            descendants = {name for name in base_selected if name.startswith(cost.name+'.')}
            if not self._legal((base_selected-descendants)|{cost.name},base_kept,data.get('conflicts',())):
                continue
            # The runtime may snapshot after an alias write. Match the saved
            # version's interval rather than demanding equal write indices.
            # region_costs already excludes ambiguous mutations. The first
            # backward read bounds the forward-produced version here.
            covered = frozenset(key for key in packet_map for v in cost.saved_versions
                if key[0]==v.storage_id and v.produced_at<=key[1]<v.first_backward_read)
            if not covered:
                continue
            extra = {}
            if cost.name in movable:
                # Newly required checkpoint inputs absent from this raw plan
                # consume additional shared queue time and Host reservation.
                for sid in rows[cost.name].get('input_storage_ids',()):
                    if sid in data['trace'].retained_sids:
                        continue
                    matches = [p for (storage,op),p in packet_map.items() if storage==sid and op<=cost.first_op
                               and p.candidate.end.op_id>data['phase_report']['forward_end_op']]
                    if len(matches)!=1:
                        extra[sid] = data['trace'].storage_sizes[sid]
            gain = sum(packet_map[k].d2h_ms+packet_map[k].h2d_ms for k in covered)
            gain -= sum(transfer_duration_ms(size,data['trace'].d2h_bandwidth_gbps,data['trace'].transfer_cost_samples,'d2h')
                        +transfer_duration_ms(size,data['trace'].h2d_bandwidth_gbps,data['trace'].transfer_cost_samples,'h2d')
                        for size in extra.values())
            if gain>0:
                candidates.append((cost,covered,extra,gain))
        # First use the finest certified regions. Ancestors are lower-budget
        # alternatives after that frontier, not replacements for fine search.
        leaves = [row for row in candidates if not any(other[0].name.startswith(row[0].name+'.') for other in candidates)]
        parents = [row for row in candidates if row not in leaves]
        order = lambda row:(row[0].compute_ms/row[3], -row[0].first_op)
        leaves.sort(key=order); parents.sort(key=order)
        reference = dict(trial=trial,data=data,capacity=capacity,packets=packet_map,choices=leaves+parents,
                         base_selected=base_selected,base_kept=base_kept,costs={c.name:c for c in costs})
        self._references[key] = reference
        self.diagnostics['global_budgets'].append(dict(stage='observed_reference',capacity=asdict(capacity),
            reference_recompute=list(trial.policy.recompute_regions), visible_regions=len(costs),
            peak_crossing_packets=len(packets), original_bytes=sum(p.candidate.size_bytes for p in packets),
            finest_choices=len(leaves), ancestor_choices=len(parents)))
        return reference

    def _admit_cut(self, reference, selected):
        """Joint capacity admission; swaps do not retain old synthetic bounds."""
        data = reference['data']
        try:
            memory = PhaseCapacity(data['phase_report'],op_count=len(data['trace'].ops),
                                   limit_bytes=data['device_limit_bytes'])
        except (ValueError,TypeError,KeyError):
            return None
        extra_host = 0
        movable = set(data.get('offloaded_checkpoint_inputs',()))
        bounds = []
        for name in selected-reference['base_selected']:
            c = reference['costs'][name]
            last = max(v.last_backward_read or v.first_backward_read for v in c.saved_versions)
            native = 0 if name in movable else c.input_boundary_bytes
            if name in movable:
                extra_host += c.input_boundary_bytes
            bounds.extend(((c.first_op,last,native,None),
                           (c.first_op,last,max(c.working_set_bytes,c.created_bytes+c.input_boundary_bytes+c.output_boundary_bytes),
                            'global_checkpoint_workspace')))
        if reference['trial'].host_peak_bytes+extra_host>data['host_limit_bytes'] or not memory.try_reserve(bounds):
            return None
        return memory.estimated_peak_bytes,extra_host

    def _allocate(self, reference, fraction):
        selected = set(reference['base_selected'])
        chosen = {}
        packets, data = reference['packets'], reference['data']
        capacity = reference['capacity']
        # A cut is selected against one shared whole-step budget; no per-node
        # independently spendable slack is introduced here.
        for choice in [None,*reference['choices']]:
            if choice is not None:
                cost, _, _, _ = choice
                descendants = {name for name in selected if name.startswith(cost.name+'.')}
                proposed = (selected-descendants)|{cost.name}
                if not self._legal(proposed,reference['base_kept'],data.get('conflicts',())):
                    continue
                # One unplaceable input or workspace must not poison every
                # later combination. Rebuild bounds before accepting a move.
                if self._admit_cut(reference,proposed) is None:
                    continue
                for name in descendants:
                    chosen.pop(name,None)
                selected = proposed
                chosen[cost.name] = choice
            removed = set().union(*(row[1] for row in chosen.values())) if chosen else set()
            extra = {}
            for row in chosen.values():
                extra.update(row[2])
            remaining = [p for key,p in packets.items() if key not in removed]
            volume = sum(p.candidate.size_bytes for p in remaining)+sum(extra.values())
            dms = sum(p.d2h_ms for p in remaining)+sum(transfer_duration_ms(size,data['trace'].d2h_bandwidth_gbps,
                data['trace'].transfer_cost_samples,'d2h') for size in extra.values())
            hms = sum(p.h2d_ms for p in remaining)+sum(transfer_duration_ms(size,data['trace'].h2d_bandwidth_gbps,
                data['trace'].transfer_cost_samples,'h2d') for size in extra.values())
            load = self._load(capacity,dms,hms)
            if load>fraction or volume>data['host_limit_bytes']:
                continue
            admitted = self._admit_cut(reference,selected)
            if admitted is None:
                continue
            parent = reference['trial'].policy
            policy = replace(parent,recompute_regions=tuple(sorted(selected)),codec_key='raw')
            return Proposal(policy,parent,'global_budget_cut',-sum(row[0].compute_ms for row in chosen.values()),
                tuple(sorted(chosen)),admitted[0],admitted[1],'global_budget'), dict(
                    fraction=fraction, realized_load=load, estimated_transfer_bytes=volume,
                    estimated_d2h_ms=dms,estimated_h2d_ms=hms,
                    capacity_original_bytes=capacity.roundtrip_original_bytes,
                    reference_packet_count=len(packets),remaining_reference_packets=len(remaining),
                    removed_peak_versions=len(removed),extra_input_bytes=sum(extra.values()),
                    scope='reference-graph version demand; actual changed graph is recaptured and replanned')
        return None

    def _fractions(self, reference, trials):
        key = reference['trial'].policy.structure_key
        used = self._attempted_fractions.setdefault(key,set())
        for value in (1.,.75,.5,.25):
            if value not in used:
                yield value
        measured = [(self._budget_policies[t.policy][1]['fraction'],t.score) for t in trials
                    if t.policy in self._budget_policies and self._budget_policies[t.policy][0]==key and math.isfinite(t.score)]
        packets = reference['packets'].values()
        base_load = self._load(reference['capacity'],sum(p.d2h_ms for p in packets),
                               sum(p.h2d_ms for p in packets))
        # The observed reference itself is a measured budget point, even
        # when it came from a broad reopening rather than a prior budget cut.
        if 0.<base_load<=1.:
            measured.extend((base_load,t.score) for t in trials
                            if t.policy==reference['trial'].policy and math.isfinite(t.score))
        if measured:
            best = min(measured,key=lambda row:row[1])[0]
            ordered = sorted(used|{0.,1.}|{row[0] for row in measured})
            neighbors = ([max(x for x in ordered if x<best)] if any(x<best for x in ordered) else [])
            neighbors += ([min(x for x in ordered if x>best)] if any(x>best for x in ordered) else [])
            for neighbor in neighbors:
                middle = (best+neighbor)/2
                if middle not in used and abs(best-neighbor)>.03125:
                    yield middle

    def _reference_front(self, references, trials):
        """Keep wide demand evidence, the winner and newly exposed scopes."""
        if not references:
            return []
        indexed = {r['trial'].policy.structure_key:r for r in references}
        widest = max(indexed.values(),key=lambda r:(sum(p.candidate.size_bytes for p in r['packets'].values()),
                                                     len(r['choices'])))
        fastest = min((t for t in trials if math.isfinite(t.score)),key=lambda t:t.score)
        best_reference=indexed.get(fastest.policy.structure_key)
        if best_reference is None:
            # An all-checkpoint winning graph can have no visible cost rows.
            # Preserve the reference that priced its global cut, instead of
            # treating the fastest still-visible (but slower) graph as best.
            basis=next((value[0] for policy,value in self._budget_policies.items()
                        if policy.structure_key==fastest.policy.structure_key and value[0] in indexed),None)
            best_reference=indexed.get(basis,widest)
        reopened = next((indexed[t.policy.structure_key] for t in reversed(trials)
            if t.policy.structure_key in indexed and t.policy in self._metadata
            and self._metadata[t.policy].family=='global_capacity_reopen'),None)
        result=[]
        for ref in (reopened,widest,best_reference):
            if ref is not None and all(ref is not other for other in result):
                result.append(ref)
        if self._round<=3:
            return [widest,*[r for r in result if r is not widest]]
        counts = {key:sum(value[0]==key for value in self._budget_policies.values()) for key in indexed}
        return sorted(result,key=lambda r:counts[r['trial'].policy.structure_key])

    def _residency_refinements(self, trial, data):
        """Batch keep choices under current full-device capacity, no credits.

        This follows the broad copy-budget sweep. Only visible certified raw
        scopes with measured saved versions can become resident. Keeping a
        scope reserves all created storage, not merely its offloaded subset.
        Existing resident scopes can also return to raw or recomputation.
        """
        result = []
        for recompute in (False, True):
            result.extend(replace(p,family='residency',refinement_axis=p.family)
                for p in self._resident_front(trial,data,recompute=recompute))
        result.extend(self._opaque_keep_front(trial,data))
        key = trial.policy.structure_key
        if key in self._invalid_graphs or data.get('phase_report') is None:
            return result
        try:
            memory = PhaseCapacity(data['phase_report'],op_count=len(data['trace'].ops),
                                   limit_bytes=data['device_limit_bytes'])
        except (ValueError,TypeError,KeyError):
            return result
        selected,kept = set(trial.policy.recompute_regions),set(trial.policy.resident_regions)
        costs = [c for c in self._costs.get(key,()) if c.saved_versions and c.name in data['names']
                 and c.name not in selected|kept and self._legal(selected,kept|{c.name},data.get('conflicts',()))]
        leaves = [c for c in costs if not any(o.name.startswith(c.name+'.') for o in costs)]
        leaves.sort(key=lambda c:(-c.saved_bytes/max(1,c.created_bytes+c.input_boundary_bytes+c.output_boundary_bytes),
                                  c.first_op))
        accepted,paths = [],[]
        for cost in leaves:
            last = max(v.last_backward_read or v.first_backward_read for v in cost.saved_versions)
            native = cost.created_bytes+cost.input_boundary_bytes+cost.output_boundary_bytes
            if not self._legal(selected,kept|{cost.name},data.get('conflicts',())):
                continue
            if not memory.try_reserve([(cost.first_op,last,native,None)]):
                continue
            kept.add(cost.name);accepted.append(cost.name)
            paths.append(Proposal(replace(trial.policy,resident_regions=tuple(sorted(kept)),codec_key='raw'),
                trial.policy,'residency',0.,tuple(accepted),memory.estimated_peak_bytes,
                structural_mode='visible_resident_batch',refinement_axis='raw_to_resident'))
        if paths:
            result.extend(paths[i] for i in sorted({len(paths)-1,(len(paths)-1)//2},reverse=True))
        return result

    def _opaque_keep_front(self, trial, data):
        """Allow a measured checkpoint to become resident after a global cut.

        Its interior is opaque in the current graph, so require an observed
        structural template and exact current opaque boundaries. Charge all
        created/input/output storage across that lifetime, without subtracting
        the previous checkpoint's reservation or discounting temporary saves.
        """
        key=trial.policy.structure_key
        if key in self._invalid_graphs or data.get('phase_report') is None:
            return []
        try:
            memory=PhaseCapacity(data['phase_report'],op_count=len(data['trace'].ops),
                                 limit_bytes=data['device_limit_bytes'])
        except (ValueError,TypeError,KeyError):
            return []
        rows={row['name']:row for row in data['report']['regions'] if row['complete']}
        choices=[]
        for expansion in self._expansions(trial,data,False):
            template=self._template(rows[expansion.name])
            native=template.created_bytes+template.input_bytes+template.output_bytes
            if native>0:
                choices.append((expansion,template,native))
        choices.sort(key=lambda item:(-item[1].compute_ms/item[2],item[0].first_op))
        selected,kept=set(trial.policy.recompute_regions),set(trial.policy.resident_regions)
        accepted,paths,saved=[],[],0.
        for expansion,template,native in choices:
            proposed=selected-{expansion.name}
            if not self._legal(proposed,kept|{expansion.name},data.get('conflicts',())):
                continue
            if not memory.try_reserve([(expansion.first_op,expansion.last_op,native,None)]):
                continue
            selected=proposed;kept.add(expansion.name);accepted.append(expansion.name)
            saved+=template.compute_ms
            paths.append(Proposal(replace(trial.policy,recompute_regions=tuple(sorted(selected)),
                resident_regions=tuple(sorted(kept)),codec_key='raw'),trial.policy,'residency',saved,
                tuple(accepted),memory.estimated_peak_bytes,structural_mode='opaque_checkpoint_to_resident',
                refinement_axis='recompute_to_resident'))
        return [paths[i] for i in sorted({len(paths)-1,(len(paths)-1)//2},reverse=True)] if paths else []

    def __call__(self, trials):
        if not trials or not trials[0].feasible:
            return None
        available = []
        for trial in trials:
            if not math.isfinite(trial.score):
                continue
            data = self.provider(trial.policy)
            if data is not None:
                self._ingest(data,trial.policy.structure_key)
                available.append((trial,data))
        if not self._capacity_seed_done:
            self._capacity_seed_done = True
            data = self.provider(trials[0].policy)
            if data is not None:
                proposal = self._large_seed(trials[0],data)
                if proposal is not None:
                    return self._record(proposal)
        tested = {t.policy for t in trials}
        references = [r for trial,data in available if (r:=self._reference(trial,data)) is not None]
        self._round += 1
        structural = None
        for reference in self._reference_front(references,trials):
            key = reference['trial'].policy.structure_key
            for fraction in self._fractions(reference,trials):
                self._attempted_fractions[key].add(fraction)
                allocation = self._allocate(reference,fraction)
                if allocation is not None and allocation[0].policy not in tested:
                    structural = allocation
                    break
            if structural is not None:
                break
        # Revisit opaque parents once broad budget choices have been measured.
        # Their initial retention was a capacity decision, not a frozen rule.
        if self._round>=4 and (structural is None or self._round%2):
            for trial,data in sorted(available,key=lambda item:item[0].score):
                key=trial.policy.structure_key
                if key in self._reopened_structures:
                    continue
                if trial.policy not in self._reopen_proposals:
                    self._reopen_proposals[trial.policy]=self._large_seed(trial,data,parents_only=True)
                proposal=self._reopen_proposals[trial.policy]
                if proposal is not None and proposal.policy not in tested:
                    if structural is not None:
                        self._attempted_fractions[reference['trial'].policy.structure_key].discard(structural[1]['fraction'])
                    self._reopened_structures.add(key)
                    return self._record(proposal)
        refinements = []
        for trial,data in sorted(available,key=lambda item:item[0].score)[:self.beam_width]:
            refinements.extend(p for p in self._refinements(trial,data) if p.policy not in tested)
            if trial.policy not in self._fronts:
                self._fronts[trial.policy] = self._residency_refinements(trial,data)
            refinements.extend(p for p in self._fronts[trial.policy] if p.policy not in tested)
        # Broad global budgets come first. Thereafter alternate budget
        # bracketing and measured transport/codec refinements.
        if structural is not None and (self._round<=4 or self._round%2):
            proposal,details = structural
            key = reference['trial'].policy.structure_key
            self._budget_policies[proposal.policy] = (key,details)
            self.diagnostics['allocations'].append(details)
            result = self._record(proposal)
            self._proposed[-1].update(global_budget=details)
            return result
        if refinements:
            order = ('compression','trim','schedule','residency')
            chosen = min(refinements,key=lambda p:(self._family_counts.get(p.family,0),
                order.index(p.family),self._axis_counts.get(p.refinement_axis,0),
                next(t.score for t in trials if t.policy==p.parent)))
            # An allocated but deferred structural point is not a tested point.
            if structural is not None:
                self._attempted_fractions[reference['trial'].policy.structure_key].discard(structural[1]['fraction'])
            return self._record(chosen)
        if structural is not None:
            proposal,details = structural
            self._budget_policies[proposal.policy] = (reference['trial'].policy.structure_key,details)
            self.diagnostics['allocations'].append(details)
            result = self._record(proposal)
            self._proposed[-1].update(global_budget=details)
            return result
        self.diagnostics['frontier_status'] = 'global_budget_frontier_exhausted_not_global_convergence'
        return None

"""Direct fixed-policy experiment control; no production tuning API changes.

The policy is supplied before training (freeze step 0). The ordinary first
warmup still collects a trace and builds its replay plan. After that warmup,
retain only the fixed executor, exactly as after automatic tuning freezes.
The training entry sees ``stable`` immediately and skips tuning observations
and distributed decision consensus throughout this control.
"""
from copy import deepcopy
from dataclasses import asdict


def make_direct_fixed(controller):
    if controller.steps != 0 or controller.phase != 'warmup':
        raise ValueError('direct fixed control must be configured before training')
    controller.phase = 'stable'
    controller.frozen_at_step = 0
    controller.stop_reason = 'experiment policy supplied before training'


def finish_direct_warmup(session):
    controller = session.controller
    if (not controller.stable or controller.steps != 0 or controller.frozen_at_step != 0
            or session.mode != 'replay' or session.trace_collections != 1 or session.plan_builds != 1):
        raise ValueError('direct fixed control requires exactly one ordinary completed warmup')
    if controller.trials or controller.search_seconds or session.policy_switches:
        raise ValueError('direct fixed control unexpectedly performed tuning')
    session._plans.clear()
    session._traces.clear()
    session.planner = session.planner.planner
    fields = ('switch_seconds', 'switch_drain_seconds', 'apply_regions_seconds', 'codec_prepare_seconds',
              'warmup_context_seconds', 'warmup_finish_seconds', 'plan_seconds', 'plan_builds',
              'trace_collections', 'post_offload_trim_calls', 'post_offload_trim_seconds')
    result = {key: getattr(session, key) for key in fields}
    result.update(observation_and_consensus_seconds=0.0, timing_events=deepcopy(session.timing_events),
                  cache_reclaim_stats=dict(session.cache_reclaim_stats),
                  cache_reclaim_observation_enabled=session.record_cache_reclaim,
                  timing_event_count=len(session.timing_events), proposal_compute_seconds=None,
                  proposal_consensus_seconds=None, phase_sample_count=0, region_observation_count=0,
                  phase='stable', steps=0, frozen_at_step=0, policy=asdict(controller.policy),
                  search_seconds=0.0, confirmation_target=0, confirmation_fallback=False,
                  selection_log=[], trials=[], direct_fixed_policy=True,
                  snapshot_boundary='after first trace/plan; policy fixed before training')
    return result

"""Index qualified screens without confusing trial samples with final policies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent


def policy_summary(policy):
    def group(names):
        return dict(Counter(name.split('.', 1)[1] if '.' in name else 'layer' for name in names))
    return dict(recompute_count=policy['recompute_count'],
                recompute_regions=group(policy.get('recompute_regions', [])),
                resident_regions=group(policy.get('resident_regions', [])),
                resident_mb=policy['resident_mb'], masking=policy['masking'],
                transfer_window_scale=policy['transfer_window_scale'],
                limit_prefetch_window=policy.get('limit_prefetch_window', True),
                prefetch_on_copy_stream=policy.get('prefetch_on_copy_stream', False),
                trim_before_replay=policy.get('trim_before_replay', True),
                defer_h2d_release=policy.get('defer_h2d_release', False),
                trim_after_offload=policy.get('trim_after_offload', False))


rows = []
for path in sorted(root.glob('r*/*screen*summary.json')):
    value = json.loads(path.read_text())
    if value.get('status') not in ('screen_audit_passed', 'screen_memory_ineligible'):
        continue
    tuning = value.get('tuning')
    if not tuning:
        continue
    assert len(tuning) == 2 and tuning[0]['policy'] == tuning[1]['policy']
    current = tuning[0]
    trials = [dict(policy=policy_summary(trial['policy']),
                   tuning_samples_seconds=trial['samples'],
                   tuning_median_seconds=statistics.median(trial['samples']) if trial['samples'] else None,
                   feasible=trial['feasible'],
                   allocated_peak_gib=trial['device_peak_bytes'] / 1024**3)
              for trial in current['trials']]
    rows.append(dict(evidence=str(path.relative_to(root)), evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     status=value['status'], memory_mode=value.get('memory_mode', 'baseline-peak'),
                     baseline_seconds=value['baseline_seconds'], candidate_seconds=value['candidate_seconds'],
                     reduction_percent=value['reduction_percent'], first_step=value['first_step'],
                     samples=value['samples'], selected_policy=policy_summary(current['policy']),
                     frozen_at_step=current['frozen_at_step'], search_seconds=current['search_seconds'],
                     plan_seconds=current['plan_seconds'], memory=value.get('memory'),
                     legacy_missing_memory_report='memory' not in value,
                     trials=trials, independent_repeat_confirmation=False))
report = dict(caveat='These are separate screens, often against an older baseline run. Their apparent gains '
              'are not paired independent confirmations. The stable metric belongs to selected_policy; '
              'rejected policies have only tuning samples. Different memory eligibility rules remain explicit.',
              independent_confirmation='repeats20261101a/summary.json', screens=rows)
(root / 'EXPERIMENT_INDEX.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(screens=len(rows), output=str(root / 'EXPERIMENT_INDEX.json'))))

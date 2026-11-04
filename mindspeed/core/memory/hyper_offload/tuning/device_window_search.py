"""Finite measured search over profile-derived, distinct execution structures."""

from dataclasses import asdict, replace
import math

from .exploration_regions import ExplorationNeighborhood, Proposal


class DeviceWindowNeighborhood(ExplorationNeighborhood):
    def __init__(self, provider, profile, **kwargs):
        super().__init__(provider, **kwargs)
        self.profile = profile
        self.structures = None
        self.diagnostics.update(
            algorithm="device_window_portfolio", signature=profile.signature, structure_portfolio=[]
        )

    def retire_evidence(self):
        super().retire_evidence()
        self.profile = None
        self.structures = None

    def __call__(self, trials):
        anchor = trials[0]
        data = self.provider(anchor.policy)
        if self.structures is None:
            raw = self.profile.portfolio(
                data["phase_report"],
                device_limit=min(data["device_limit_bytes"], anchor.device_peak_bytes),
                host_headroom=max(0, data["host_limit_bytes"] - anchor.host_peak_bytes),
            )
            self.structures = []
            for row in raw:
                chosen = (
                    set(anchor.policy.recompute_regions)
                    - set(row.offload)
                    - set(row.resident)
                    - set(row.replaced_checkpoints)
                )
                kept = set(anchor.policy.resident_regions) | set(row.resident)
                if not self._legal(chosen, kept, data["conflicts"]):
                    continue
                policy = replace(
                    anchor.policy,
                    recompute_regions=tuple(sorted(chosen)),
                    resident_regions=tuple(sorted(kept)),
                    codec_key="raw",
                )
                self.structures.append((row, policy))
            self.diagnostics["structure_portfolio"] = [asdict(row) for row, _ in self.structures]
        tested = {t.policy for t in trials}
        codec_tested = {p.structure_key for p in tested if p.codec_key != "raw"}
        actions = []
        for trial in sorted((t for t in trials if math.isfinite(t.score)), key=lambda t: t.score):
            observed = self.provider(trial.policy)
            for key in observed.get("codec_candidates", ()):
                if key != trial.policy.codec_key:
                    actions.append(("codec", trial, replace(trial.policy, codec_key=key)))
        # Distinct global alternatives compete before repeated local settings.
        # Alternate a new structure and a graph-specific codec/scheduling move.
        by_mode = {}
        for row, policy in self.structures:
            by_mode.setdefault(row.mode, []).append((row, policy))
        structures = [
            items[i]
            for i in range(max(map(len, by_mode.values()), default=0))
            for items in by_mode.values()
            if i < len(items)
        ]
        broad = next(((row, p) for row, p in structures if p not in tested), None)
        # Compare joint scheduling options on the fastest observed structures,
        # including moving the H2D allocator and release together. Do not infer
        # that more lookahead alone is faster.
        for trial, data in self._structural_beam(
            [(t, self.provider(t.policy)) for t in trials if math.isfinite(t.score)]
        ):
            p = trial.policy
            actions.append(
                (
                    "shared_calendar",
                    trial,
                    replace(p, shared_transfer_calendar=not p.shared_transfer_calendar),
                )
            )
            if not p.prefetch_on_copy_stream or not p.defer_h2d_release:
                actions.append(
                    (
                        "async_replay",
                        trial,
                        replace(p, prefetch_on_copy_stream=True, defer_h2d_release=True),
                    )
                )
            if p.limit_prefetch_window:
                actions.append(("open_lookahead", trial, replace(p, limit_prefetch_window=False)))
            if p.trim_before_replay and p.trim_replay_interval == 1:
                actions.append(("periodic_replay", trial, replace(p, trim_replay_interval=2)))
        actions = [a for a in actions if a[2] not in tested]
        if actions and (broad is None or len(trials) % 2 == 1):
            # Prefer actual joint placements and immutable codecs over a series
            # of single timing flags. Give each family a measured comparison.
            priority = ("codec", "shared_calendar", "async_replay", "open_lookahead", "periodic_replay")
            # Before revisiting an old cut's timing knobs, pair a freshly
            # captured cut with one codec scoped to its own storage versions.
            first_codec = [a for a in actions if a[0] == "codec" and a[2].structure_key not in codec_tested]
            axis, parent, p = min(
                first_codec or actions,
                key=lambda a: (self._axis_counts.get(a[0], 0), priority.index(a[0]), a[1].score),
            )
            return self._record(
                Proposal(
                    p,
                    parent.policy,
                    "window_option",
                    0.0,
                    (),
                    refinement_axis=axis,
                    structural_mode="window_option",
                )
            )
        if broad:
            row, p = broad
            return self._record(
                Proposal(
                    p,
                    anchor.policy,
                    "window_structure",
                    1000 * row.saved_seconds,
                    row.offload + row.resident,
                    additional_host_bound=row.gross_bytes,
                    structural_mode=row.mode,
                )
            )
        return None

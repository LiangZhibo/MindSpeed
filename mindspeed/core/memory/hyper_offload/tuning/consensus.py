"""Integer metadata for comparing the complete policy across ranks."""

import hashlib

from .controller import JointPolicy


def policy_fingerprint(policy: JointPolicy) -> tuple[int, ...]:
    """Return all 256 digest bits as signed int64 words for exact transport.

    The frozen dataclass representation covers every policy field, including
    exact named regions and floating-point window values. Keeping the digest
    in integers avoids losing bits on devices that do not support float64.
    """
    digest = hashlib.sha256(repr(policy).encode('utf-8')).digest()
    return tuple(int.from_bytes(digest[i : i + 8], 'big', signed=True) for i in range(0, len(digest), 8))


def decision_fingerprint(controller) -> tuple[int, ...]:
    """Encode the complete bounded-controller decision for rank agreement.

    Keep state encoding with the policy metadata so callers cannot omit a new
    transition such as timing-confirmation fallback. Unknown states fail closed.
    """
    phases = ('warmup', 'measure', 'restore', 'confirm', 'stable', 'fallback')
    reference = controller.confirmation_reference_index
    reference_policy = controller.trials[reference].policy if controller.trials else controller.policy
    return (controller.steps, phases.index(controller.phase), *policy_fingerprint(controller.policy),
            controller.confirmation_target, reference, *policy_fingerprint(reference_policy))

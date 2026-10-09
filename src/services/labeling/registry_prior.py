"""Optional registry prior for VLM item labeling (#61 item 2).

A prompt pack with ``registry_prior_top_k > 0`` tells the VLM which registry
classes are best established in the project (ranked by validated-item count)
and which new-class proposals are well supported. The list is a hint: the reply parser
still accepts any answer, and the lock rule and test holdout are untouched
because only the prompt text changes.

A live COCO oracle run (#193) showed the first version made labeling worse: with
no validated item the ranking was alphabetical, and pending proposal names fed back
into the prompt so a noise term reinforced itself. So the prior now (1) never lists a
denylisted name, (2) lists only classes that have validated items and is skipped
entirely when none do, and (3) lists a pending proposal only once at least
``MIN_PENDING_SUPPORT`` crops carry it.

Class identity is the name; no index crosses this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.labeling.vlm_prompts import proposal_denied


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


#: Hard ceiling on ``registry_prior_top_k``, enforced by pack validation.
MAX_REGISTRY_PRIOR_TOP_K = 50

#: A pending proposal name is fed back into the prompt only when at least this many
#: crops carry it. A name proposed once or twice is more likely a VLM one-off than a
#: class; three independent proposals is the smallest count that is not a coincidence
#: of one batch. Fixed (not a pack field) so a pack cannot reopen the feedback loop.
MIN_PENDING_SUPPORT = 3


@dataclass(frozen=True)
class RegistryPrior:
    """Candidate names for one VLM labeling run, already ordered."""

    classes: tuple[str, ...]
    pending: tuple[str, ...]

    def prompt_text(self) -> str:
        parts = [f'Most established registry classes: {", ".join(self.classes)}.']
        if self.pending:
            parts.append(f'Pending proposed classes: {", ".join(self.pending)}.')
        parts.append(
            'Treat these as a prior, not a constraint: answer with any registry class or '
            'a new class if the crop clearly belongs elsewhere.'
        )
        return ' '.join(parts)


def rank_registry_prior(
    registry_names: Collection[str],
    validated_counts: Mapping[str, int],
    pending_counts: Mapping[str, int],
    *,
    top_k: int,
    denylist: Sequence[str] = (),
) -> RegistryPrior | None:
    """Top ``top_k`` registry names by validated count, ties by name.

    Only registry classes with at least one validated item are ranked, and
    ``None`` is returned when there are none: ranking zero-count classes can only
    be alphabetical, which primes the VLM with noise. Denylisted names are never
    listed, whether registry or pending. A pending proposal is listed only with
    ``MIN_PENDING_SUPPORT`` crops behind it; one that already is a registry name is
    dropped (a duplicate candidate). ``None`` also when ``top_k`` is not positive
    or the registry is empty.
    """
    if top_k <= 0 or not registry_names:
        return None
    known = set(registry_names)
    ranked = sorted(
        (
            n
            for n in known
            if int(validated_counts.get(n, 0)) > 0 and not proposal_denied(n, denylist)
        ),
        key=lambda n: (-int(validated_counts[n]), n),
    )[:top_k]
    if not ranked:
        return None
    pending = sorted(
        (
            n
            for n, count in pending_counts.items()
            if n
            and n not in known
            and int(count) >= MIN_PENDING_SUPPORT
            and not proposal_denied(n, denylist)
        ),
        key=lambda n: (-int(pending_counts[n]), n),
    )[:top_k]
    return RegistryPrior(classes=tuple(ranked), pending=tuple(pending))

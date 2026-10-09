"""Optional registry prior for VLM item labeling (#61 item 2).

A prompt pack with ``registry_prior_top_k > 0`` tells the VLM which registry
classes are best established in the project (ranked by validated-item count)
and which new-class proposals are pending. The list is a hint: the reply parser
still accepts any answer, and the lock rule and test holdout are untouched
because only the prompt text changes.

Class identity is the name; no index crosses this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


#: Hard ceiling on ``registry_prior_top_k``, enforced by pack validation.
MAX_REGISTRY_PRIOR_TOP_K = 50


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
) -> RegistryPrior | None:
    """Top ``top_k`` registry names by validated count, ties by name.

    Pending proposals that already are registry names are dropped (they would
    be a duplicate candidate). ``None`` when ``top_k`` is not positive or the
    registry is empty: there is nothing to prime with.
    """
    if top_k <= 0 or not registry_names:
        return None
    known = set(registry_names)
    classes = sorted(known, key=lambda n: (-int(validated_counts.get(n, 0)), n))[:top_k]
    pending = sorted(
        (n for n in pending_counts if n and n not in known),
        key=lambda n: (-int(pending_counts[n]), n),
    )[:top_k]
    return RegistryPrior(classes=tuple(classes), pending=tuple(pending))

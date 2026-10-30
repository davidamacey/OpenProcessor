"""A run's sampled answers read as a label proposal for the scope.

Computed only from the answers the VLM gave on the sampled items, never from
unsampled ones. A locked item's answer counts as a vote too, which makes it a
check against the human label; its class is never written.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.services.curation.assist.proposal_models import (
    AttributeSummary,
    AttributeValueCount,
    ClassVotes,
    NewClassVotes,
    Proposal,
)


MIN_ANSWERED = 3
"""Fewer answers than this never pass the gate, however unanimous."""
TOP_VALUES = 10


@dataclass(frozen=True)
class ClassAnswer:
    """What the VLM said about one sampled item."""

    class_name: str | None
    """The class it named: a registry name when it resolved, else its own words."""
    proposed: str = ''
    """The slug it proposed when nothing fit (`__new__`); not a vote for a class."""
    attributes: dict[str, str] = field(default_factory=dict)
    crop_id: str = ''
    """The item that was asked (the auto-label loop keeps answers per crop)."""


def compute_proposal(
    answers: list[ClassAnswer], *, min_agreement: float, registry_names: frozenset[str]
) -> Proposal:
    """Votes per named class, agreement = top votes over answered, and the gate."""
    votes = Counter(a.class_name for a in answers if a.class_name)
    new = Counter(a.proposed for a in answers if a.proposed and not a.class_name)
    answered = sum(votes.values()) + sum(new.values())
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, top_votes = ranked[0] if ranked else (None, 0)
    agreement = (top_votes / answered) if answered else 0.0
    passes = answered >= MIN_ANSWERED and agreement >= min_agreement
    chosen = top_name if passes else None
    return Proposal(
        class_name=chosen,
        in_registry=chosen is not None and chosen in registry_names,
        agreement=round(agreement, 4),
        answered=answered,
        min_agreement=min_agreement,
        passes_gate=passes,
        distribution=[ClassVotes(class_name=n, votes=v) for n, v in ranked],
        new_class_proposals=[
            NewClassVotes(name=n, votes=v)
            for n, v in sorted(new.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        attribute_summary=_attribute_summary(answers),
    )


def _attribute_summary(answers: list[ClassAnswer]) -> list[AttributeSummary]:
    by_key: dict[str, Counter[str]] = {}
    for a in answers:
        for key, value in a.attributes.items():
            by_key.setdefault(key, Counter())[value] += 1
    return [
        AttributeSummary(
            key=key,
            values=[
                AttributeValueCount(value=v, count=c)
                for v, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_VALUES]
            ],
            answered=sum(counts.values()),
        )
        for key, counts in sorted(by_key.items())
    ]


__all__ = ['ClassAnswer', 'compute_proposal']

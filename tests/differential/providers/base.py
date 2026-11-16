"""What a fixtures provider is: the parameter values both seeded stacks hold.

Generated requests carry random ids, so most of them meet a 404 on both stacks and prove little. A provider
asks the two stacks over HTTP which values of a family's path and query parameters really exist, and keeps
the values present in both; the runner then adds one case per value with that parameter fixed and
everything else generated. A family opts in with ``fixtures = ["<name>"]`` in ``families.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx


class ProviderError(RuntimeError):
    """A stack did not answer a provider's question, so the family's fixture cases cannot be built."""


@dataclass(frozen=True)
class Fixtures:
    """Parameter name to the values present in both stacks, plus what only one of them holds."""

    values: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Provider(Protocol):
    name: str

    def collect(self, reference: httpx.Client, candidate: httpx.Client) -> Fixtures: ...


def present_in_both(parameter: str, reference: Iterable[str], candidate: Iterable[str]) -> Fixtures:
    """The sorted values both stacks hold; the others become notes (the seeding differs)."""
    a, b = set(reference), set(candidate)
    notes = [f'{parameter} {v!r} is only in the reference stack' for v in sorted(a - b)]
    notes += [f'{parameter} {v!r} is only in the candidate stack' for v in sorted(b - a)]
    return Fixtures({parameter: sorted(a & b)}, notes)

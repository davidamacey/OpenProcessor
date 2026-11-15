"""The differential test: today's code against the fold, on random interleavings of human and
machine acts. Every difference must be in ``ALLOWED``, each entry naming the plan line that decided
it; an unexplained difference is a bug in the fold or a gap in the plan."""

from __future__ import annotations

from tests.domain.world_allowed import ALLOWED
from tests.domain.world_compare import run_seed


SEEDS = range(2000)


def test_differential() -> None:
    bad = []
    for seed in SEEDS:
        bad += run_seed(seed, ALLOWED).divergences
    names: dict[str, int] = {}
    for d in bad:
        names[d.field] = names.get(d.field, 0) + 1
    print(names)
    seen = set()
    for d in sorted(bad, key=lambda d: len(d.acts)):
        if (d.field, len(d.acts) > 3) in seen:
            continue
        seen.add((d.field, len(d.acts) > 3))
        print('##', d.seed, d.field, '| doc:', d.doc, '| fold:', d.fold, '|', d.acts)
    assert not bad

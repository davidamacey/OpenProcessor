"""Stable person ids for face clustering.

A person is the set of faces sharing a ``person_id``; there are no person
documents. Given the connected components of the face-similarity graph and the
ids already stored, :func:`assign_person_ids` decides every face's id and name
so that a re-run keeps ids, and a human decision (a name or a merge) is never
overwritten. Pure: no I/O, stdlib only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def is_person_locked(source: Mapping[str, Any]) -> bool:
    """A face is locked once a person was named or merged; clusterers must not write it."""
    return bool(source.get('named_at') or source.get('merged_at'))


@dataclass(frozen=True)
class FaceState:
    face_id: str
    person_id: str | None
    person_name: str | None
    locked: bool


@dataclass(frozen=True)
class PersonWrite:
    face_id: str
    person_id: str
    person_name: str | None


@dataclass
class PersonAssignment:
    writes: list[PersonWrite] = field(default_factory=list)
    persons_total: int = 0
    persons_kept: int = 0
    persons_new: int = 0
    persons_retired: int = 0
    singletons: int = 0
    faces_locked: int = 0


def _units(
    faces: Sequence[FaceState], components: Sequence[Sequence[int]], min_faces: int
) -> list[list[int]]:
    """Components of at least ``min_faces`` stay whole; smaller ones become one-face units."""
    units: list[list[int]] = []
    for comp in components:
        if len(comp) >= min_faces:
            units.append(sorted(comp, key=lambda i: faces[i].face_id))
        else:
            units.extend([i] for i in comp)
    return units


def _claim_ids(
    faces: Sequence[FaceState], units: list[list[int]], keys: list[str]
) -> dict[int, str]:
    """Greedy: largest overlap between a unit and an existing id claims that id first."""
    cands: list[tuple[int, str, str, int]] = []
    for u, members in enumerate(units):
        votes = Counter(faces[i].person_id for i in members if faces[i].person_id)
        cands.extend((-cnt, pid, keys[u], u) for pid, cnt in votes.items() if pid)
    cands.sort()
    owner: dict[int, str] = {}
    claimed: set[str] = set()
    for _, pid, _, u in cands:
        if u in owner or pid in claimed:
            continue
        owner[u] = pid
        claimed.add(pid)
    return owner


def _names(faces: Sequence[FaceState]) -> dict[str, str]:
    votes: dict[str, Counter[str]] = {}
    for f in faces:
        if f.locked and f.person_id and f.person_name:
            votes.setdefault(f.person_id, Counter())[f.person_name] += 1
    return {pid: min(c, key=lambda n: (-c[n], n)) for pid, c in votes.items()}


def assign_person_ids(
    faces: Sequence[FaceState],
    components: Sequence[Sequence[int]],
    *,
    min_faces: int,
    mint: Callable[[], str],
) -> PersonAssignment:
    """Decide the person id and name of every face.

    ``components`` partition ``faces`` by index. Components below ``min_faces``
    split into one-face units, so every face ends up in a person. A unit keeps
    the existing id it overlaps most (ties: id ascending, then the unit's
    smallest face id); units with no candidate get a minted id. Locked faces
    are never written but still vote, so new faces of a named person join it.
    """
    units = _units(faces, components, min_faces)
    keys = [min(faces[i].face_id for i in members) for members in units]
    owner = _claim_ids(faces, units, keys)
    claimed = set(owner.values())
    existing = {f.person_id for f in faces if f.person_id}

    persons_new = 0
    for u in sorted((u for u in range(len(units)) if u not in owner), key=lambda u: keys[u]):
        pid = mint()
        while pid in existing or pid in claimed:
            pid = mint()
        owner[u] = pid
        claimed.add(pid)
        persons_new += 1

    names = _names(faces)
    out = PersonAssignment(persons_new=persons_new)
    final: dict[int, str] = {}
    for u, members in enumerate(units):
        target = owner[u]
        for i in members:
            f = faces[i]
            if f.locked and f.person_id:
                out.faces_locked += 1
                final[i] = f.person_id
                continue
            final[i] = target
            name = names.get(target)
            if (target, name) != (f.person_id, f.person_name):
                out.writes.append(PersonWrite(f.face_id, target, name))

    counts = Counter(final.values())
    out.persons_total = len(counts)
    out.singletons = sum(1 for c in counts.values() if c == 1)
    out.persons_kept = len(claimed & existing)
    out.persons_retired = len(existing - set(counts))
    return out

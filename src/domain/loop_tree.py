"""Node paths, seeded child sampling and the best-first order of the auto-label
loop's partition tree: pure, stdlib only.

A path names a node: ``r10042`` is the root cluster 10042, ``r10042.2`` its
third child, ``r10042.2.0`` that child's first child."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


_PATH = re.compile(r'r(\d+)((?:\.\d+)*)')


def root_path(cluster_id: int) -> str:
    return f'r{cluster_id}'


def child_path(parent: str, index: int) -> str:
    if index < 0:
        msg = 'child index must not be negative'
        raise ValueError(msg)
    parse_path(parent)
    return f'{parent}.{index}'


def parse_path(path: str) -> tuple[int, tuple[int, ...]]:
    """``(root cluster id, child indices)``; anything but the grammar is refused."""
    match = _PATH.fullmatch(path)
    if match is None:
        msg = f'not a node path: {path!r}'
        raise ValueError(msg)
    indices = tuple(int(part) for part in match.group(2).split('.')[1:])
    return int(match.group(1)), indices


def path_level(path: str) -> int:
    return len(parse_path(path)[1])


def parent_path(path: str) -> str | None:
    """The parent's path, or ``None`` for a root."""
    parse_path(path)
    head, dot, _ = path.rpartition('.')
    return head if dot else None


@dataclass(frozen=True)
class ChildSample:
    """``reused`` ids already answered at the parent (no new call); ``draw`` ids
    to ask now. Both sorted."""

    reused: list[str]
    draw: list[str]


def child_sample(
    parent_sampled: list[str], child_ids: list[str], k: int, *, seed: int, path: str
) -> ChildSample:
    """The ids to answer for a child of ``k`` wanted: every parent-sampled id that
    falls in the child is reused, the rest of ``k`` is a seeded draw from the
    child's other members. The result depends on the ids, ``k``, ``seed`` and
    ``path`` only, never on input order."""
    if k < 1:
        msg = 'k must be at least 1'
        raise ValueError(msg)
    members = sorted(set(child_ids))
    answered = set(parent_sampled)
    reused = [crop_id for crop_id in members if crop_id in answered]
    pool = [crop_id for crop_id in members if crop_id not in answered]
    count = min(max(k - len(reused), 0), len(pool))
    draw = random.Random(f'{seed}:{path}').sample(pool, count)
    return ChildSample(reused=reused, draw=sorted(draw))


def best_first_key(n: int, path: str) -> tuple[int, int, tuple[int, ...]]:
    """Sort key putting the largest node first, ties by numeric path, so the budget
    goes to the nodes that label the most items per call."""
    root, indices = parse_path(path)
    return (-n, root, indices)

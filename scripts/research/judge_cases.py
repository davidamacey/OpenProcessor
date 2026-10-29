"""Labelled test cases for the visual judge: synthetic clusters, pair rows and sheet sampling.

A case is built from a pool of labelled public crops (``pool.jsonl``: item_id, label, path, ...).
The truth lives next to the sheets, never on them: ``sheet_truth`` derives it from the sheet
manifest (which cells were actually shown) and the item labels.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.services.labeling.contact_sheet import SheetMember, member_key


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

SIMILAR_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({'car', 'truck', 'bus', 'train', 'van'}),
    frozenset({'bicycle', 'motorcycle'}),
    frozenset({'cat', 'dog'}),
    frozenset({'horse', 'cow', 'sheep'}),
    frozenset({'cup', 'bottle', 'bowl'}),
)
N_PROTOTYPES = 3


@dataclass(frozen=True)
class PoolItem:
    item_id: str
    label: str
    path: str
    group: str = ''
    bbox_norm: tuple[float, float, float, float] | None = None

    def member(self, *, prototype: bool = False, distance: float | None = None) -> SheetMember:
        return SheetMember(
            item_id=self.item_id,
            image=Path(self.path).read_bytes(),
            bbox_norm=self.bbox_norm,
            is_prototype=prototype,
            distance=distance,
        )


@dataclass(frozen=True)
class Case:
    case_id: str
    majority: str
    members: list[SheetMember]
    labels: dict[str, str]
    n_bad: int
    bad_kind: str


def load_pool(path: Path, only: Collection[str] | None = None) -> dict[str, list[PoolItem]]:
    by_label: dict[str, list[PoolItem]] = {}
    for line in path.read_text().splitlines():
        doc = json.loads(line)
        if only is not None and doc['label'] not in only:
            continue
        box = doc.get('bbox_norm')
        item = PoolItem(
            doc['item_id'],
            doc['label'],
            doc['path'],
            doc.get('group', ''),
            tuple(box) if box else None,
        )
        by_label.setdefault(item.label, []).append(item)
    return {k: sorted(v, key=lambda i: i.item_id) for k, v in sorted(by_label.items())}


def _similar(label: str, labels: Sequence[str]) -> list[str]:
    near = [g for g in SIMILAR_GROUPS if label in g]
    return sorted({x for g in near for x in g if x != label and x in labels})


def _contaminant_label(
    pool: Mapping[str, Sequence[PoolItem]], rng: random.Random, majority: str, hard: bool
) -> tuple[str, str]:
    labels = sorted(pool)
    close = _similar(majority, labels)
    if hard and close:
        return rng.choice(close), 'hard'
    return rng.choice([x for x in labels if x != majority and x not in close] or close), 'easy'


def synthetic_cluster(
    pool: Mapping[str, Sequence[PoolItem]],
    rng: random.Random,
    *,
    case_id: str,
    size: int,
    n_bad: int,
    hard_fraction: float = 0.5,
) -> Case:
    """``size - n_bad`` items of one label plus ``n_bad`` items of other labels, with distances.

    Contaminants get the largest distances, like real outliers, so a sheet shows them as such.
    """
    majors = [x for x, items in sorted(pool.items()) if len(items) >= size - n_bad]
    majority = rng.choice(majors)
    chosen = rng.sample(list(pool[majority]), size - n_bad)
    bad_kind = 'none'
    bad: list[PoolItem] = []
    if n_bad:
        label, bad_kind = _contaminant_label(pool, rng, majority, rng.random() < hard_fraction)
        bad = rng.sample(list(pool[label]), min(n_bad, len(pool[label])))
    dists = sorted(rng.uniform(0.0, 0.5) for _ in chosen)
    members = [
        it.member(prototype=i < N_PROTOTYPES, distance=d)
        for i, (it, d) in enumerate(zip(chosen, dists, strict=True))
    ]
    members += [it.member(distance=rng.uniform(0.5, 1.0)) for it in bad]
    labels = {member_key(it.item_id, ''): it.label for it in [*chosen, *bad]}
    return Case(case_id, majority, members, labels, len(bad), bad_kind)


def pair_rows(
    pool: Mapping[str, Sequence[PoolItem]],
    rng: random.Random,
    *,
    n_pairs: int = 12,
    hard_fraction: float = 0.5,
) -> tuple[list[tuple[SheetMember, SheetMember]], dict[str, str]]:
    """Half same-label pairs, half different-label (some from a confusable group)."""
    pairs: list[tuple[SheetMember, SheetMember]] = []
    labels: dict[str, str] = {}
    two_plus = [x for x, items in sorted(pool.items()) if len(items) >= 2]
    for k in range(n_pairs):
        if k % 2 == 0:
            a, b = rng.sample(list(pool[rng.choice(two_plus)]), 2)
        else:
            first = rng.choice(sorted(pool))
            other, _ = _contaminant_label(pool, rng, first, rng.random() < hard_fraction)
            a, b = rng.choice(pool[first]), rng.choice(pool[other])
        pairs.append((a.member(), b.member()))
        labels[member_key(a.item_id, '')] = a.label
        labels[member_key(b.item_id, '')] = b.label
    rng.shuffle(pairs)
    return pairs, labels


def sheet_truth(manifest: Mapping[str, Any], labels: Mapping[str, str]) -> dict[str, Any]:
    """Ground truth of what the sheet shows, from the cells actually drawn."""
    cells = manifest['cells']
    if manifest['layout'] == 'pair':
        rows: dict[int, list[str]] = {}
        for c in cells:
            rows.setdefault(c['row'], []).append(labels[member_key(c['item_id'], c['box_id'])])
        return {'rows': {r: a == b for r, (a, b) in sorted(rows.items())}}
    shown = {c['cell']: labels[member_key(c['item_id'], c['box_id'])] for c in cells}
    counts = Counter(shown.values())
    majority = sorted(counts, key=lambda k: (-counts[k], k))[0]
    odd = sorted(n for n, lab in shown.items() if lab != majority)
    return {'same': not odd, 'odd_cells': odd, 'majority': majority}


def targeted_random_split(tags: Mapping[str, set[str]], n: int, seed: int) -> list[tuple[str, str]]:
    """Half the budget uniform at random over all ids, half from tagged ids; reported apart.

    Returns ``(id, sample)`` with sample ``random`` or ``targeted``. The random half is drawn
    first and uniformly, so purity estimated from it stays unbiased; targeted picks prefer ids
    with more tags and never repeat a random pick.
    """
    rng = random.Random(f'{seed}:split')
    ids = sorted(tags)
    n = min(n, len(ids))
    n_random = n - n // 2
    rand = rng.sample(ids, n_random)
    pool = [i for i in ids if tags[i] and i not in set(rand)]
    rng.shuffle(pool)
    pool.sort(key=lambda i: -len(tags[i]))
    targeted = pool[: n // 2]
    return [(i, 'random') for i in sorted(rand)] + [(i, 'targeted') for i in sorted(targeted)]

"""Pure selection logic of the vehicle-heavy Open Images validation set of experiment Z2 (no network).

Open Images v7 validation images only (CC BY 2.0), disjoint from every earlier pin. Ground truth is
the dataset's own boxes. Four groups, each image in one group, drawn in this order so the scarce
group is not starved by the others:

* ``openset``: the image holds one of the 12 animals the COCO label space has no class for
* ``animal``: the largest box is a class the detector can name (cat, dog, horse, sheep, cow)
* ``vehicle``: the largest box is a vehicle (car, truck, bus, motorcycle, bicycle, train)
* ``rest``: whatever remains (mostly person-led scenes), drawn balanced over the largest-box class
  until the set holds ``TARGET_IMAGES``. The validation split has only about 11,500 licence-eligible
  images with these classes, so the three named groups take nearly every vehicle and open-set image
  and ``rest`` is the filler

Inside a group the draw is balanced over the group's key (round-robin over the key's seeded
shuffle), so a short key hands its places to the others instead of leaving the group short.
"""

from __future__ import annotations

import random
from typing import Any

from scripts.datasets.cluster_eval_selection import OI_RARE, pin_hash


SCHEMA = 'vehicle-eval-pin/1'
SEED = 20261104
QUOTAS = {'openset': 1200, 'animal': 1200, 'vehicle': 5200}
TARGET_IMAGES = 8000
VEHICLES = frozenset({'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'})
DETECTABLE_ANIMALS = frozenset({'cat', 'dog', 'horse', 'sheep', 'cow'})
OPENSET = frozenset(OI_RARE.values())
GROUP_ORDER = ('openset', 'animal', 'vehicle', 'rest')


def balanced_draw(
    pool: list[dict[str, Any]], n: int, seed: str, taken: set[str]
) -> list[dict[str, Any]]:
    """``n`` rows (fewer if the pool runs out) spread evenly over ``row['key']``.

    Round-robin over the keys in name order, each key's rows shuffled by the seed, so a key with
    fewer rows than its share gives the rest to the others. Rows already ``taken`` are skipped.
    """
    rng = random.Random(seed)
    by: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(pool, key=lambda r: r['file_name']):
        if r['file_name'] not in taken:
            by.setdefault(r['key'], []).append(r)
    for rows in by.values():
        rng.shuffle(rows)
    out: list[dict[str, Any]] = []
    cursor = dict.fromkeys(by, 0)
    while len(out) < n and any(cursor[k] < len(by[k]) for k in by):
        for k in sorted(by):
            if len(out) >= n:
                break
            if cursor[k] < len(by[k]):
                out.append(by[k][cursor[k]])
                cursor[k] += 1
    return out


def _group_pool(group: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if group == 'openset':
        # a key is the scarcest of the image's open-set classes, so it is not outvoted
        counts: dict[str, int] = {}
        for r in rows:
            for c in OPENSET & set(r['classes']):
                counts[c] = counts.get(c, 0) + 1
        return [
            dict(r, key=min(OPENSET & set(r['classes']), key=lambda c: (counts[c], c)))
            for r in rows
            if OPENSET & set(r['classes'])
        ]
    if group == 'animal':
        return [dict(r, key=r['primary']) for r in rows if r['primary'] in DETECTABLE_ANIMALS]
    if group == 'vehicle':
        return [dict(r, key=r['primary']) for r in rows if r['primary'] in VEHICLES]
    return [dict(r, key=r['primary']) for r in rows]


def select_base(
    rows: list[dict[str, Any]],
    seed: int = SEED,
    scale: float = 1.0,
    exclude: frozenset[str] = frozenset(),
    quotas: dict[str, int] | None = None,
    target: int = TARGET_IMAGES,
) -> list[dict[str, Any]]:
    """Base images of every group; none in ``exclude`` and none in two groups.

    ``quotas`` (group to image count, default ``QUOTAS``) and ``target`` (total images, default
    ``TARGET_IMAGES``) let a later set reuse the draw with its own sizes; a group without a quota
    is the filler up to ``target``.
    """
    quotas = QUOTAS if quotas is None else quotas
    taken: set[str] = set(exclude)
    out: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        pool = _group_pool(group, rows)
        want = round(quotas[group] * scale) if group in quotas else round(target * scale)
        if group == 'rest':
            want -= len(out)
        pick = balanced_draw(pool, want, f'{seed}:{group}', taken)
        for r in pick:
            r.pop('key', None)
            out.append({**r, 'group': group})
            taken.add(r['file_name'])
    return sorted(out, key=lambda r: r['file_name'])


# The pin omits the author column: the original URL identifies the source, and the authors stay in
# ATTRIBUTION.csv next to the images, outside git.
def build_pin(
    base: list[dict[str, Any]],
    seed: int,
    scale: float,
    annotations_sha256: str,
    quotas: dict[str, int] | None = None,
    target: int = TARGET_IMAGES,
) -> dict:
    keep = [
        'source',
        'image_id',
        'file_name',
        'split',
        'license',
        'attribution_url',
        'group',
        'width',
        'height',
        'bytes',
        'sha256',
    ]
    pin = {
        'schema': SCHEMA,
        'seed': seed,
        'scale': scale,
        'rare_classes': sorted(OPENSET),
        'vehicle_classes': sorted(VEHICLES),
        'quotas': {g: round(q * scale) for g, q in (QUOTAS if quotas is None else quotas).items()},
        'target_images': round(target * scale),
        'annotations_sha256': annotations_sha256,
        'images': [{k: r.get(k) for k in keep} for r in base],
        'near_duplicates': [],
    }
    pin['manifest_sha256'] = pin_hash(pin)
    return pin


def overlap(
    names: set[str], hashes: set[str], other_names: set[str], other_hashes: set[str]
) -> dict[str, list[str]]:
    """Names and content hashes that two sets share (empty lists mean disjoint)."""
    return {'names': sorted(names & other_names), 'hashes': sorted(hashes & other_hashes)}

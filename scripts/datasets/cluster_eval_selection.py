"""Pure selection logic of the 3,000-item cluster evaluation set (#341; no network, no pixels).

The set is public images only: COCO val2017 (Flickr, CC BY / CC BY-SA and the other redistributable
licences of ``fetch_coco_subset.py``) and Open Images v7 validation and test (CC BY 2.0). Ground
truth is the datasets' own boxes, never the product's detector, VLM or clusters.

Image groups, each drawn with a seeded sample so the same inputs and seed give the same pin:

* ``similar``: the largest box is a class from a visually confusable group
* ``rare``: the image holds one of the rarest classes (COCO: fewest train2017 instances; Open
  Images: classes the COCO label set does not have)
* ``lowres``: many boxes under 32 px on a side
* ``mixed``: several distinct classes in one scene
* near-duplicates: ``SOURCES`` drawn from the groups above, each given ``VARIANTS`` seeded
  re-encodes with crop jitter

``fetch_cluster_eval.py`` runs the network and image steps around this.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any


# The stratum rules the scorer and the selection share (the container image carries
# scripts/datasets, not scripts/bench, so the definitions live here).
LOW_RES_PX = 32
SIMILAR_GROUPS = (
    {'car', 'truck', 'bus', 'train'},
    {'bicycle', 'motorcycle'},
    {'cat', 'dog'},
    {'horse', 'cow', 'sheep'},
)
SIMILAR = frozenset().union(*SIMILAR_GROUPS)

SCHEMA = 'cluster-eval-pin/1'
SEED = 20261028
TARGET_ITEMS = 3000
SOURCES = 100
VARIANTS = 2
RARE_COCO = 12
MIN_AREA_FRACTION = 0.005
MIN_KEPT_FRACTION = 0.6
MAX_SOURCE_BOXES = (
    3  # near-duplicate sources are sparse scenes so the stratum stays near 15 percent
)
LOW_RES_BOXES = 6
MAX_MARGIN = 0.06
QUALITY_RANGE = (55, 90)

# Quota of images per group and source. ``scale`` multiplies them.
QUOTAS = {
    ('coco', 'similar'): 220,
    ('coco', 'rare'): 180,
    ('coco', 'lowres'): 40,
    ('coco', 'mixed'): 50,
    ('openimages', 'similar'): 80,
    ('openimages', 'rare'): 260,
    ('openimages', 'mixed'): 20,
}

# Open Images display name -> the class name used in the evaluation labels. Identity by name:
# Cattle is the Open Images word for COCO's cow; every other name maps to its lower-case self.
OI_SIMILAR = {
    'Car': 'car',
    'Truck': 'truck',
    'Bus': 'bus',
    'Train': 'train',
    'Bicycle': 'bicycle',
    'Motorcycle': 'motorcycle',
    'Cat': 'cat',
    'Dog': 'dog',
    'Horse': 'horse',
    'Sheep': 'sheep',
    'Cattle': 'cow',
}
OI_RARE = {
    n: n.lower()
    for n in (
        'Alpaca',
        'Camel',
        'Goat',
        'Kangaroo',
        'Lion',
        'Tiger',
        'Pig',
        'Rabbit',
        'Fox',
        'Squirrel',
        'Penguin',
        'Owl',
    )
}
OI_OTHER = {'Person': 'person'}
OI_CLASSES = {**OI_SIMILAR, **OI_RARE, **OI_OTHER}


def coco_images(
    data: dict[str, Any], allowed_licenses: set[str], names: dict[int, str] | None = None
) -> list[dict[str, Any]]:
    """One row per licence-eligible image: its classes, small-box count and largest-box class."""
    cats = names or {c['id']: c['name'] for c in data['categories']}
    lic = {item['id']: item['name'] for item in data['licenses']}
    per: dict[int, list[dict[str, Any]]] = {}
    for a in data['annotations']:
        if not a.get('iscrowd'):
            per.setdefault(a['image_id'], []).append(a)
    out = []
    for im in sorted(data['images'], key=lambda i: i['id']):
        if lic.get(im['license']) not in allowed_licenses:
            continue
        anns = per.get(im['id'], [])
        if not anns:
            continue
        frame = im['width'] * im['height']
        big = [a for a in anns if a['bbox'][2] * a['bbox'][3] / frame >= MIN_AREA_FRACTION]
        out.append(
            {
                'source': 'coco',
                'image_id': im['id'],
                'file_name': im['file_name'],
                'split': 'val2017',
                'license': lic[im['license']],
                'attribution_url': im.get('flickr_url', ''),
                'author': '',
                'width': im['width'],
                'height': im['height'],
                'classes': dict(Counter(cats[a['category_id']] for a in anns)),
                'small': sum(min(a['bbox'][2], a['bbox'][3]) < LOW_RES_PX for a in anns),
                'primary': cats[max(big, key=lambda a: a['bbox'][2] * a['bbox'][3])['category_id']]
                if big
                else None,
            }
        )
    return out


def rarest_coco_classes(train_data: dict[str, Any], n: int = RARE_COCO) -> list[dict[str, Any]]:
    """The ``n`` COCO classes with the fewest non-crowd train2017 instances (ties by name)."""
    names = {c['id']: c['name'] for c in train_data['categories']}
    counts = Counter(names[a['category_id']] for a in train_data['annotations'] if not a['iscrowd'])
    ranked = sorted(names.values(), key=lambda c: (counts[c], c))[:n]
    return [{'class': c, 'train2017_instances': counts[c]} for c in ranked]


def openimages_images(
    boxes: dict[str, list[dict[str, Any]]], images: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rows shaped like :func:`coco_images` for Open Images ids that have a licence row.

    ``boxes`` maps an ImageID to its non-group-of boxes (``name``, ``x_min``.. normalised);
    ``images`` maps an ImageID to its CC BY 2.0 metadata row.
    """
    out = []
    for image_id in sorted(boxes):
        meta = images.get(image_id)
        if meta is None:
            continue
        bs = [
            b
            for b in boxes[image_id]
            if (b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min']) >= MIN_AREA_FRACTION
        ]
        if not bs:
            continue
        primary = max(bs, key=lambda b: (b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min']))[
            'name'
        ]
        out.append(
            {
                'source': 'openimages',
                'image_id': image_id,
                'file_name': f'oi_{image_id}.jpg',
                'split': meta['split'],
                'license': meta['license'],
                'attribution_url': meta['original_url'],
                'author': meta['author'],
                'classes': dict(Counter(b['name'] for b in bs)),
                'small': 0,
                'primary': primary,
            }
        )
    return out


def _draw(
    pool: list[dict[str, Any]], n: int, seed: str, taken: set[str], per_class: str | None = None
) -> list[dict[str, Any]]:
    free = [r for r in pool if r['file_name'] not in taken]
    free.sort(key=lambda r: r['file_name'])
    rng = random.Random(seed)
    if per_class is None:
        return rng.sample(free, min(n, len(free)))
    by: dict[str, list[dict[str, Any]]] = {}
    for r in free:
        by.setdefault(r[per_class], []).append(r)
    classes = sorted(by)
    share = -(-n // max(len(classes), 1))
    chosen: list[dict[str, Any]] = []
    for cls in classes:
        chosen += rng.sample(by[cls], min(share, len(by[cls])))
    rng.shuffle(chosen)
    return sorted(chosen[:n], key=lambda r: r['file_name'])


def select_base(
    coco: list[dict[str, Any]],
    oi: list[dict[str, Any]],
    rare_coco: set[str],
    seed: int = SEED,
    scale: float = 1.0,
    exclude: frozenset[str] = frozenset(),
    backfill: bool = False,
) -> list[dict[str, Any]]:
    """Base images of every group, each image in one group (similar, rare, lowres, mixed).

    ``exclude`` lists file names that must not be drawn (a second pin disjoint from the first).
    With ``backfill``, a COCO group that the exclusion leaves short of its quota hands the
    shortfall to the Open Images draw of the same group (COCO val2017 has 5,000 images and is the
    only COCO split the detector was not trained on).
    """
    rare = {'coco': rare_coco, 'openimages': set(OI_RARE.values())}
    pools = {'coco': coco, 'openimages': oi}
    taken: set[str] = set(exclude)
    out: list[dict[str, Any]] = []
    short: dict[str, int] = {}
    for (source, group), quota in QUOTAS.items():
        n = round(quota * scale) + (short.get(group, 0) if source == 'openimages' else 0)
        rows = pools[source]
        if group == 'similar':
            pool = [dict(r, key=r['primary']) for r in rows if r['primary'] in SIMILAR]
            pick = _draw(pool, n, f'{seed}:{source}:{group}', taken, 'key')
        elif group == 'rare':
            pool = [r for r in rows if rare[source] & set(r['classes'])]
            pool = [dict(r, key=sorted(rare[source] & set(r['classes']))[0]) for r in pool]
            pick = _draw(pool, n, f'{seed}:{source}:{group}', taken, 'key')
        elif group == 'lowres':
            pick = _draw(
                [r for r in rows if r['small'] >= LOW_RES_BOXES],
                n,
                f'{seed}:{source}:{group}',
                taken,
            )
        else:
            need = 3 if source == 'coco' else 2
            pick = _draw(
                [r for r in rows if len(r['classes']) >= need],
                n,
                f'{seed}:{source}:{group}',
                taken,
            )
        if backfill and source == 'coco':
            short[group] = n - len(pick)
        for r in pick:
            r.pop('key', None)
            out.append({**r, 'group': group})
            taken.add(r['file_name'])
    return sorted(out, key=lambda r: (r['source'], r['file_name']))


def jitter_plan(stem: str, seed: int = SEED, variants: int = VARIANTS) -> list[dict[str, Any]]:
    """Seeded crop margins (fractions of width and height) and JPEG quality for each variant."""
    plan = []
    for k in range(1, variants + 1):
        rng = random.Random(f'{seed}:nd:{stem}:{k}')
        plan.append(
            {
                'file_name': f'nd_{stem}_{k}.jpg',
                'margins': [round(rng.uniform(0.0, MAX_MARGIN), 4) for _ in range(4)],
                'quality': rng.randint(*QUALITY_RANGE),
            }
        )
    return plan


def select_near_duplicates(
    base: list[dict[str, Any]], seed: int = SEED, sources: int = SOURCES
) -> list[dict[str, Any]]:
    """``sources`` base images (seeded) and the jitter plan of their variants."""
    pool = sorted(
        (r for r in base if sum(r['classes'].values()) <= MAX_SOURCE_BOXES),
        key=lambda r: r['file_name'],
    )
    picked = random.Random(f'{seed}:nd-sources').sample(pool, min(sources, len(pool)))
    return [
        {
            'parent': r['file_name'],
            'variants': jitter_plan(r['file_name'].rsplit('.', 1)[0], seed),
        }
        for r in sorted(picked, key=lambda r: r['file_name'])
    ]


def jitter_window(width: int, height: int, margins: list[float]) -> tuple[int, int, int, int]:
    """Pixel crop window (left, top, right, bottom) for margin fractions (l, t, r, b)."""
    left, top, right, bottom = margins
    return (
        round(left * width),
        round(top * height),
        width - round(right * width),
        height - round(bottom * height),
    )


def shift_annotations(
    anns: list[dict[str, Any]], window: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    """Ground-truth boxes in the cropped frame; boxes mostly cut away are dropped."""
    left, top, right, bottom = window
    out = []
    for a in anns:
        x, y, w, h = a['bbox']
        x1, y1, x2, y2 = max(x, left), max(y, top), min(x + w, right), min(y + h, bottom)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < MIN_KEPT_FRACTION * w * h:
            continue
        out.append({**a, 'bbox': [x1 - left, y1 - top, x2 - x1, y2 - y1]})
    return out


def pin_hash(pin: dict[str, Any]) -> str:
    body = {k: v for k, v in pin.items() if k != 'manifest_sha256'}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def build_pin(
    base: list[dict[str, Any]],
    near_dups: list[dict[str, Any]],
    rare_coco: list[dict[str, Any]],
    seed: int,
    scale: float,
    annotations_sha256: str,
) -> dict[str, Any]:
    keep = [
        'source',
        'image_id',
        'file_name',
        'split',
        'license',
        'attribution_url',
        'author',
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
        'target_items': TARGET_ITEMS,
        'low_res_px': LOW_RES_PX,
        'similar_groups': [sorted(g) for g in SIMILAR_GROUPS],
        'rare_classes': sorted([r['class'] for r in rare_coco] + list(OI_RARE.values())),
        'rare_coco_train2017_instances': rare_coco,
        'quotas': {f'{s}:{g}': round(q * scale) for (s, g), q in QUOTAS.items()},
        'annotations_sha256': annotations_sha256,
        'images': [{k: r.get(k) for k in keep} for r in base],
        'near_duplicates': near_dups,
    }
    pin['manifest_sha256'] = pin_hash(pin)
    return pin

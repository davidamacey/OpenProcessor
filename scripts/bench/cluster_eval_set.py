"""Build cluster-evaluation inputs from a project of public COCO images and the COCO annotations.

    cluster_eval_set.py build --api http://127.0.0.1:4603 --project NAME --coco-gt coco_gt.json \\
        --out-dir artifacts_local/cluster_eval/NAME [--min-iou 0.5]

Reads the project's crops through ``GET /curation/projects/{project}/crops`` (read only) and
writes ``labels.jsonl`` and ``partition.jsonl`` for ``cluster_eval.py score``, plus
``manifest.json`` with the counts behind them.

The ground truth never comes from the product's own detector, VLM or clusters. A crop gets the
class of the COCO annotation it overlaps most (IoU at least ``--min-iou``, same image, crowd
annotations ignored). A crop that overlaps no annotation is left out of the labels and counted,
because inventing a label for it would score the pipeline against itself. ``protected`` is
read from the project (holdout, locked or validated crops).

``--eval-manifest`` points at the pinned ``cluster_eval_3000.json``: its ``rare_classes`` replace
the frequency rule (rare means rare in COCO, not rare in this sample) and its seed and hash are
recorded. An image whose ground-truth entry carries ``near_duplicate_of`` (a seeded re-encode
and crop jitter of another image, or that image itself) puts all its crops in ``near_duplicate``.

A crop falls in one stratum, first match wins: ``near_duplicate``, ``low_res`` (box under 32x32 px), ``rare`` (its
class is among the least frequent third of annotated classes), ``similar_classes`` (a class from
a visually confusable group), ``mixed_scene`` (its image has annotations of two or more classes),
else ``other``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.parity_metrics import box_iou
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url


PAGE_SIZE = 500
LOW_RES_PX = 32
MIN_IOU = 0.5
SIMILAR_GROUPS = (
    {'car', 'truck', 'bus', 'train'},
    {'bicycle', 'motorcycle'},
    {'cat', 'dog'},
    {'horse', 'cow', 'sheep'},
)
SIMILAR = frozenset().union(*SIMILAR_GROUPS)


class EvalSetError(Exception):
    """The project and the annotations cannot yield trustworthy labels."""


def fetch_crops(client: httpx.Client, base: str) -> Iterator[dict[str, Any]]:
    page = 1
    while True:
        resp = client.get(
            f'{base}/crops',
            params={
                'page': page,
                'page_size': PAGE_SIZE,
                'include_test': 'true',
                'include_excluded': 'true',
            },
        )
        resp.raise_for_status()
        body = resp.json()
        yield from body['crops']
        if page * PAGE_SIZE >= int(body['total']):
            return
        page += 1


def load_coco(path: Path) -> dict[str, Any]:
    gt = json.loads(path.read_text())
    cats = {c['id']: c['name'] for c in gt['categories']}
    images = {PurePosixPath(i['file_name']).name: i for i in gt['images']}
    near_dup = {n for n, i in images.items() if i.get('near_duplicate_of')}
    near_dup |= {
        PurePosixPath(i['near_duplicate_of']).name
        for i in images.values()
        if i.get('near_duplicate_of')
    }
    anns: dict[int, list[dict[str, Any]]] = {}
    for a in gt['annotations']:
        if not a.get('iscrowd'):
            anns.setdefault(a['image_id'], []).append({**a, 'name': cats[a['category_id']]})
    return {'images': images, 'annotations': anns, 'near_duplicate': near_dup}


def match_class(
    crop: dict[str, Any], coco: dict[str, Any], min_iou: float
) -> tuple[str, float, float] | None:
    """(class name, IoU, box side in pixels) of the best annotation, or None below ``min_iou``."""
    image = coco['images'].get(PurePosixPath(crop['image_path']).name)
    if image is None:
        return None
    w, h = image['width'], image['height']
    x1, y1, x2, y2 = crop['bbox_norm']
    box = np.array([[x1 * w, y1 * h, x2 * w, y2 * h]])
    anns = coco['annotations'].get(image['id'], [])
    if not anns:
        return None
    gt = np.array(
        [
            [a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
            for a in anns
        ]
    )
    iou = box_iou(box, gt)[0]
    best = int(iou.argmax())
    if iou[best] < min_iou:
        return None
    return anns[best]['name'], float(iou[best]), float(min((x2 - x1) * w, (y2 - y1) * h))


def rare_classes(class_counts: Counter[str]) -> set[str]:
    """The least frequent third (rounded up) of the annotated classes."""
    ranked = sorted(class_counts, key=lambda c: (class_counts[c], c))
    return set(ranked[: -(-len(ranked) // 3)])


def stratum(
    label: str, side_px: float, rare: set[str], image_classes: int, *, near_dup: bool = False
) -> str:
    if near_dup:
        return 'near_duplicate'
    if side_px < LOW_RES_PX:
        return 'low_res'
    if label in rare:
        return 'rare'
    if label in SIMILAR:
        return 'similar_classes'
    return 'mixed_scene' if image_classes >= 2 else 'other'


def is_protected(crop: dict[str, Any]) -> bool:
    return bool(
        crop.get('test_holdout')
        or crop.get('label_locked')
        or crop.get('label_validated')
        or crop.get('class_validated')
        or crop.get('import_ids')
    )


def build(
    crops: Sequence[dict[str, Any]],
    coco: dict[str, Any],
    min_iou: float = MIN_IOU,
    rare: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Label rows, partition rows and manifest for the crops that match an annotation."""
    matched = [(c, m) for c in crops if (m := match_class(c, coco, min_iou))]
    if not matched:
        raise EvalSetError('no crop overlaps a COCO annotation; refusing to invent labels')
    class_counts = Counter(m[0] for _, m in matched)
    rare = set(rare) if rare is not None else rare_classes(class_counts)
    per_image: dict[str, set[str]] = {}
    for c, m in matched:
        per_image.setdefault(c['image_id'], set()).add(m[0])
    labels, partition = [], []
    near_dup = coco.get('near_duplicate', set())
    for c, (name, _, side) in matched:
        is_nd = PurePosixPath(c['image_path']).name in near_dup
        labels.append(
            {
                'item_id': c['crop_id'],
                'label': name,
                'stratum': stratum(name, side, rare, len(per_image[c['image_id']]), near_dup=is_nd),
                'protected': is_protected(c),
            }
        )
    for c in crops:
        cid = c.get('cluster_id')
        partition.append(
            {'item_id': c['crop_id'], 'cluster_id': None if cid is None or cid < 0 else cid}
        )
    manifest = {
        'crops_in_project': len(crops),
        'crops_labelled': len(labels),
        'crops_without_annotation': len(crops) - len(labels),
        'min_iou': min_iou,
        'class_counts': dict(sorted(class_counts.items())),
        'rare_classes': sorted(rare),
        'strata': dict(sorted(Counter(r['stratum'] for r in labels).items())),
        'protected': sum(r['protected'] for r in labels),
        'unassigned_in_project': sum(p['cluster_id'] is None for p in partition),
    }
    return labels, partition, manifest


def cmd_build(args: argparse.Namespace, *, transport: httpx.BaseTransport | None = None) -> int:
    base = scoped_url(args.api.rstrip('/'), curation_api_prefix(), args.project, '')
    with httpx.Client(timeout=httpx.Timeout(120.0), transport=transport) as client:
        crops = list(fetch_crops(client, base))
    pin = json.loads(Path(args.eval_manifest).read_text()) if args.eval_manifest else None
    labels, partition, manifest = build(
        crops,
        load_coco(Path(args.coco_gt)),
        args.min_iou,
        set(pin['rare_classes']) if pin else None,
    )
    if pin:
        manifest['eval_manifest_sha256'] = pin['manifest_sha256']
        manifest['eval_seed'] = pin['seed']
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('labels', labels), ('partition', partition)):
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = {'project': args.project, **manifest}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='write labels, partition and manifest for one project')
    b.add_argument('--api', required=True)
    b.add_argument('--project', required=True)
    b.add_argument('--coco-gt', required=True)
    b.add_argument('--out-dir', required=True)
    b.add_argument('--eval-manifest', help='pinned cluster_eval_3000.json (rare classes, seed)')
    b.add_argument('--min-iou', type=float, default=MIN_IOU)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_build(args)
    except EvalSetError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    sys.exit(main())

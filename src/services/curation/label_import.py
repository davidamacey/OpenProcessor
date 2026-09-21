"""Generic YOLO ``.txt`` label import for the curation ingest path.

Near-verbatim port of the private reference generic label importer (see
``docs/design/curation_design_rationale.md`` §2.1 for the citation
convention; the reference file is ~444 LOC and, per that design doc,
was misclassified as domain-specific — it is a plain YOLO-format
parser + IoU matcher with no domain coupling beyond two index-name
constants, which now come from :class:`~src.config.CurationConfig`).

Imports a YOLO-format ``.txt`` label file alongside an already-ingested
image document. Builds ``labels_confirmed`` rows and flips matching
``items`` documents to ``class_validated=true``.

Behavior:

1. Look up the images-index doc for ``image_path`` (must already be
   ingested by :mod:`src.services.curation.ingest`). If missing, no-op
   with a warning.
2. Parse the YOLO ``.txt``: each line ``cls_id cx cy w h`` (normalized).
3. Reject ``cls_id >= len(registry.classes)`` or a deprecated class id.
4. Build ``labels_confirmed`` documents:
   - ``label_id = sha256(image_path + bbox)[:32]``
   - ``bbox_norm = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]``
   - ``class_name`` from the registry entry
   - ``confirmed_at = now()``
5. Bulk index. Also flip matching items documents to
   ``class_validated=true`` (best-IoU match, threshold
   :data:`LABEL_IOU_MATCH`). When no IoU match exists, create a new
   item row — a label corresponds to an object the detector didn't
   propose.

``_crop_id_for`` is deliberately identical to
:func:`src.services.curation.ingest._crop_id` (both delegate to
:func:`src.services.detection.geometry.crop_id`) — a label importer
that computed ids differently would silently fork the id space between
detector-created and label-created items for the same image.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.history import record_class_history
from src.services.detection.geometry import crop_id as _geometry_crop_id, iou as _iou


if TYPE_CHECKING:
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry


logger = get_logger(__name__)


LABEL_IOU_MATCH = 0.5

DEFAULT_LABEL_SOURCE = 'external_label'


def _images_index() -> str:
    return get_curation_config().images_index


def _items_index() -> str:
    return get_curation_config().items_index


def _labels_confirmed_index() -> str:
    return get_curation_config().labels_confirmed_index


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# =============================================================================
# Helpers
# =============================================================================


def _label_id(image_path: str, bbox_norm: list[float]) -> str:
    payload = (
        f'{image_path}|{bbox_norm[0]:.6f},{bbox_norm[1]:.6f},{bbox_norm[2]:.6f},{bbox_norm[3]:.6f}'
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def _crop_id_for(image_id: str, bbox_norm: list[float]) -> str:
    """Same id space as the ingest service's ``_crop_id`` for a given (image, bbox)."""
    return _geometry_crop_id(image_id, bbox_norm)


# =============================================================================
# Lookups
# =============================================================================


async def _lookup_image(
    image_path: str,
    opensearch: AsyncOpenSearch,
) -> dict[str, Any] | None:
    """Fetch the images-index doc whose ``image_path`` matches."""
    body = {
        'size': 1,
        'query': {'term': {'image_path': image_path}},
    }
    try:
        resp = await opensearch.search(index=_images_index(), body=body)
    except Exception as exc:
        logger.warning('label_import_image_lookup_failed', path=image_path, error=str(exc))
        return None
    hits = (resp.get('hits') or {}).get('hits') or []
    if not hits:
        return None
    src = hits[0].get('_source') or {}
    src['_id'] = hits[0]['_id']
    return src


async def _lookup_existing_crops(
    image_id: str,
    opensearch: AsyncOpenSearch,
) -> list[dict[str, Any]]:
    """Pull all items-index rows for ``image_id``.

    Excludes frozen ``test_holdout`` rows from the match candidates —
    this writer is class-only, so a holdout row simply never gets
    IoU-matched and its class fields stay untouched. A re-import that
    would have landed on a holdout row's exact bbox falls through to
    the "no match" branch instead, which is intentional: the frozen
    row's ground truth is what is being protected.
    """
    body = {
        'size': 200,
        'query': {
            'bool': {
                'must': [{'term': {'image_id': image_id}}],
                'must_not': [{'term': {'test_holdout': True}}],
            },
        },
        '_source': [
            'crop_id',
            'bbox_norm',
            'class_id',
            'class_name',
            'class_source',
            'label_source',
            'confidence',
            'class_validated',
            'class_id_history',
        ],
    }
    try:
        resp = await opensearch.search(index=_items_index(), body=body)
    except Exception as exc:
        logger.warning('label_import_crop_lookup_failed', image_id=image_id, error=str(exc))
        return []
    hits = (resp.get('hits') or {}).get('hits') or []
    out: list[dict[str, Any]] = []
    for h in hits:
        src = h.get('_source') or {}
        src['_id'] = h['_id']
        out.append(src)
    return out


# =============================================================================
# YOLO .txt parsing
# =============================================================================


def _parse_yolo_txt(
    txt_path: Path,
    registry: ClassRegistry,
) -> list[tuple[int, list[float]]]:
    """Parse a YOLO ``.txt`` -> list of ``(class_id, bbox_norm)``.

    Each line: ``cls_id cx cy w h`` (normalized 0-1). Malformed or
    deprecated/out-of-range rows are dropped with a warning.
    """
    if not txt_path.exists():
        return []
    out: list[tuple[int, list[float]]] = []
    reg = registry.load()
    n_classes = len(reg.classes)
    deprecated = {c.class_id for c in reg.classes if c.deprecated}

    for raw_line in txt_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 5:
            logger.warning('label_import_bad_row', path=str(txt_path), line=line)
            continue
        try:
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except (TypeError, ValueError):
            logger.warning('label_import_parse_failed', path=str(txt_path), line=line)
            continue
        if cls_id < 0 or cls_id >= n_classes:
            logger.warning(
                'label_import_unmapped_class',
                path=str(txt_path),
                cls_id=cls_id,
                n_classes=n_classes,
            )
            continue
        if cls_id in deprecated:
            logger.warning('label_import_deprecated_class', path=str(txt_path), cls_id=cls_id)
            continue
        bbox = [
            max(0.0, cx - bw / 2.0),
            max(0.0, cy - bh / 2.0),
            min(1.0, cx + bw / 2.0),
            min(1.0, cy + bh / 2.0),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        out.append((cls_id, bbox))
    return out


# =============================================================================
# Public API
# =============================================================================


async def import_yolo_labels(
    image_path: Path,
    label_txt_path: Path,
    registry: ClassRegistry,
    opensearch: AsyncOpenSearch,
    label_source: str = DEFAULT_LABEL_SOURCE,
) -> int:
    """Import a single YOLO label ``.txt`` -> labels_confirmed + item validation.

    Args:
        image_path: Source image path (must already be ingested).
        label_txt_path: YOLO ``.txt`` companion file.
        registry: ClassRegistry (used for deprecated/unmapped checks + class_name).
        opensearch: AsyncOpenSearch client.
        label_source: stored on each labels_confirmed row + item update.

    Returns:
        Number of label rows indexed.
    """
    image_doc = await _lookup_image(str(image_path), opensearch)
    if image_doc is None:
        logger.warning('label_import_no_image_doc', path=str(image_path))
        return 0

    image_id = image_doc.get('image_id') or image_doc.get('_id')
    parsed = _parse_yolo_txt(label_txt_path, registry)
    if not parsed:
        return 0

    reg = registry.load()
    classes_by_id = {c.class_id: c for c in reg.classes}
    existing_crops = await _lookup_existing_crops(str(image_id), opensearch)
    now = _now_iso()

    bulk_body: list[dict[str, Any]] = []

    for cls_id, bbox_norm in parsed:
        class_entry = classes_by_id.get(cls_id)
        class_name = class_entry.class_name if class_entry is not None else f'class_{cls_id}'

        # Best-IoU item match.
        best_iou = 0.0
        best_crop: dict[str, Any] | None = None
        for crop in existing_crops:
            crop_box = crop.get('bbox_norm')
            if not crop_box or len(crop_box) != 4:
                continue
            crop_bbox = (
                float(crop_box[0]),
                float(crop_box[1]),
                float(crop_box[2]),
                float(crop_box[3]),
            )
            new_bbox = (bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3])
            score = _iou(crop_bbox, new_bbox)
            if score > best_iou:
                best_iou = score
                best_crop = crop
        target_crop_id: str
        if best_crop is not None and best_iou >= LABEL_IOU_MATCH:
            target_crop_id = str(best_crop.get('crop_id') or best_crop['_id'])
            bulk_body.append({'update': {'_index': _items_index(), '_id': target_crop_id}})
            bulk_body.append(
                {
                    'doc': {
                        'class_id': cls_id,
                        'class_name': class_name,
                        'class_source': label_source,
                        'class_validated': True,
                        'label_source': label_source,
                        'updated_at': now,
                        'class_id_history': record_class_history(best_crop, writer='label_import'),
                    }
                }
            )
        else:
            # No matching item — the detector missed this object; keep
            # the human label as the source of truth.
            target_crop_id = _crop_id_for(str(image_id), bbox_norm)
            bulk_body.append({'index': {'_index': _items_index(), '_id': target_crop_id}})
            bulk_body.append(
                {
                    'crop_id': target_crop_id,
                    'image_id': image_id,
                    'image_path': str(image_path),
                    'bbox_norm': bbox_norm,
                    'class_id': cls_id,
                    'class_name': class_name,
                    'class_source': label_source,
                    'confidence': 1.0,
                    'class_validated': True,
                    'label_source': label_source,
                    'test_holdout': False,
                    'created_at': now,
                    'updated_at': now,
                }
            )

        label_id = _label_id(str(image_path), bbox_norm)
        bulk_body.append({'index': {'_index': _labels_confirmed_index(), '_id': label_id}})
        bulk_body.append(
            {
                'label_id': label_id,
                'image_path': str(image_path),
                'bbox_norm': bbox_norm,
                'class_id': cls_id,
                'class_name': class_name,
                'label_source': label_source,
                'confirmed_at': now,
                'crop_id': target_crop_id,
            }
        )

    if not bulk_body:
        return 0

    try:
        resp = await opensearch.bulk(body=bulk_body, refresh=False)
    except Exception as exc:
        logger.error('label_import_bulk_failed', path=str(label_txt_path), error=str(exc))
        return 0
    if isinstance(resp, dict) and resp.get('errors'):
        logger.warning(
            'label_import_bulk_partial_errors',
            sample=resp.get('items', [])[:3],
            n=len(parsed),
        )
    return len(parsed)


async def import_labels_batch(
    pairs: list[tuple[Path, Path]],
    registry: ClassRegistry,
    opensearch: AsyncOpenSearch,
    label_source: str = DEFAULT_LABEL_SOURCE,
) -> dict[str, int]:
    """Batch-import many image+label pairs.

    Args:
        pairs: list of ``(image_path, label_txt_path)`` tuples.
        registry: ClassRegistry.
        opensearch: AsyncOpenSearch client.
        label_source: passed to :func:`import_yolo_labels`.

    Returns:
        ``{labels_imported, files_processed, files_failed}``.
    """
    summary = {
        'labels_imported': 0,
        'files_processed': 0,
        'files_failed': 0,
    }
    for image_path, label_path in pairs:
        try:
            n = await import_yolo_labels(
                image_path,
                label_path,
                registry,
                opensearch,
                label_source=label_source,
            )
            summary['labels_imported'] += n
            summary['files_processed'] += 1
        except Exception as exc:
            logger.warning(
                'label_import_batch_item_failed',
                image_path=str(image_path),
                label_path=str(label_path),
                error=str(exc),
            )
            summary['files_failed'] += 1
    return summary


__all__ = [
    'DEFAULT_LABEL_SOURCE',
    'LABEL_IOU_MATCH',
    'import_labels_batch',
    'import_yolo_labels',
]

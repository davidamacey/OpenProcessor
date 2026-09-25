"""Generic YOLO ``.txt`` label import for the curation ingest path.

A plain YOLO-format parser + IoU matcher with no domain coupling beyond
two index-name constants, which come from
:class:`~src.config.CurationConfig` (see
``docs/design/curation_design_rationale.md`` §2.1 for the design
approach).

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

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger
from src.services.curation.history import record_class_history
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.item_doc import region_seed_status
from src.services.detection.cascade_detect import class_provenance
from src.services.detection.geometry import crop_id as _geometry_crop_id, iou as _iou


if TYPE_CHECKING:
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry


logger = get_logger(__name__)


LABEL_IOU_MATCH = 0.5

DEFAULT_LABEL_SOURCE = LABEL_IMPORT_CLASS_SOURCE

# ``kind`` values of the model-vs-label disagreement records produced when
# ``detect_mismatches`` is on.
#   class_mismatch       — label and detector box overlap (IoU >= LABEL_IOU_MATCH)
#                          but the detector said a different class.
#   missed_label         — a label no detector box overlaps: the detector missed it.
#   unmatched_detection  — a detector box no label overlaps: a false positive
#                          (on a background image with an empty/absent label
#                          file, every detection is one).
DISAGREEMENT_CLASS_MISMATCH = 'class_mismatch'
DISAGREEMENT_MISSED_LABEL = 'missed_label'
DISAGREEMENT_UNMATCHED_DETECTION = 'unmatched_detection'


def count_disagreements(records: list[dict[str, Any]]) -> dict[str, int]:
    """``{mismatches, missed_labels, unmatched_detections}`` counts for a record list."""
    kinds = [r.get('kind') for r in records]
    return {
        'mismatches': kinds.count(DISAGREEMENT_CLASS_MISMATCH),
        'missed_labels': kinds.count(DISAGREEMENT_MISSED_LABEL),
        'unmatched_detections': kinds.count(DISAGREEMENT_UNMATCHED_DETECTION),
    }


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
    """Fetch the images-index doc whose ``image_path`` matches.

    Restricted to ``image_id`` — the only field any caller reads
    (previously returned the full doc, including vectors)."""
    body = {
        'size': 1,
        'query': {'term': {'image_path': image_path}},
        '_source': ['image_id'],
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
                'filter': [{'term': {'image_id': image_id}}],
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
    detect_mismatches: bool = False,
    mismatch_sink: list[dict[str, Any]] | None = None,
    image_doc: dict[str, Any] | None = None,
) -> int:
    """Import a single YOLO label ``.txt`` -> labels_confirmed + item validation.

    Args:
        image_path: Source image path (must already be ingested).
        label_txt_path: YOLO ``.txt`` companion file.
        registry: ClassRegistry (used for deprecated/unmapped checks + class_name).
        opensearch: AsyncOpenSearch client.
        label_source: stored on each labels_confirmed row + item update.
        detect_mismatches: When true, build the model-vs-ground-truth
            disagreement report for a re-ingest-and-verify pass. An
            IoU-matched item whose existing detector ``class_id``
            disagrees with the label is stamped ``class_mismatch=true``
            (plus the detector's class and confidence) on its
            ``labels_confirmed`` row; every disagreement — see the
            ``DISAGREEMENT_*`` kinds — is appended to ``mismatch_sink``.
            An empty or absent label file is treated as a background
            image, so every detector item on it is reported. It never
            changes which class is written — the label always wins.
        mismatch_sink: Optional list that disagreement records (each
            carrying a ``kind``) are appended to, so a caller can
            count/inspect them without re-querying.
        image_doc: Pre-resolved images-index doc — pass this when
            the caller already has it (a batch import's ``_msearch``
            page, or an ingest result) to skip the per-file lookup.
            ``None`` (default) falls back to the single-file lookup.

    Returns:
        Number of label rows indexed.
    """
    if image_doc is None:
        image_doc = await _lookup_image(str(image_path), opensearch)
    if image_doc is None:
        logger.warning('label_import_no_image_doc', path=str(image_path))
        return 0

    image_id = image_doc.get('image_id') or image_doc.get('_id')
    parsed = _parse_yolo_txt(label_txt_path, registry)
    if not parsed and not detect_mismatches:
        return 0

    reg = registry.load()
    classes_by_id = {c.class_id: c for c in reg.classes}
    existing_crops = await _lookup_existing_crops(str(image_id), opensearch)
    now = _now_iso()
    # The label file, not a detector, produced these classes — overwrite
    # any detector provenance ingest stamped on an IoU-matched item.
    label_prov = class_provenance(label_source, '1', labeler='label_import', labeled_at=now)
    # Items this import creates (labels the detector missed) need region
    # detection like any ingest-created item; IoU-matched updates leave the
    # existing region status alone.
    seed = region_seed_status()
    region_seed = {get_region_fields().status: seed.value} if seed is not None else {}

    bulk_body: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    matched_crop_ids: set[str] = set()

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
        mismatch: dict[str, Any] | None = None
        if best_crop is not None and best_iou >= LABEL_IOU_MATCH:
            target_crop_id = str(best_crop.get('crop_id') or best_crop['_id'])
            matched_crop_ids.add(target_crop_id)
            if detect_mismatches:
                detector_class_id = best_crop.get('class_id')
                if detector_class_id is not None and int(detector_class_id) != cls_id:
                    mismatch = {
                        'kind': DISAGREEMENT_CLASS_MISMATCH,
                        'crop_id': target_crop_id,
                        'image_path': str(image_path),
                        'bbox_norm': bbox_norm,
                        'label_class_id': cls_id,
                        'label_class_name': class_name,
                        'detector_class_id': int(detector_class_id),
                        'detector_class_name': best_crop.get('class_name'),
                        'detector_class_source': best_crop.get('class_source'),
                        'detector_confidence': best_crop.get('confidence'),
                        'iou': best_iou,
                    }
                    disagreements.append(mismatch)
                    logger.info(
                        'label_import_class_mismatch',
                        crop_id=target_crop_id,
                        label_class_id=cls_id,
                        detector_class_id=int(detector_class_id),
                    )
            bulk_body.append({'update': {'_index': _items_index(), '_id': target_crop_id}})
            bulk_body.append(
                {
                    'doc': {
                        'class_id': cls_id,
                        'class_name': class_name,
                        'class_source': label_source,
                        'class_validated': True,
                        'label_source': label_source,
                        **label_prov,
                        'updated_at': now,
                        'class_id_history': record_class_history(best_crop, writer='label_import'),
                    }
                }
            )
        else:
            # No matching item — the detector missed this object; keep
            # the human label as the source of truth.
            target_crop_id = _crop_id_for(str(image_id), bbox_norm)
            if detect_mismatches:
                disagreements.append(
                    {
                        'kind': DISAGREEMENT_MISSED_LABEL,
                        'crop_id': target_crop_id,
                        'image_path': str(image_path),
                        'bbox_norm': bbox_norm,
                        'label_class_id': cls_id,
                        'label_class_name': class_name,
                        'best_iou': best_iou,
                    }
                )
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
                    **label_prov,
                    **region_seed,
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
                **(
                    {
                        'class_mismatch': True,
                        'detector_class_id': mismatch['detector_class_id'],
                        'detector_class_name': mismatch['detector_class_name'],
                        'detector_confidence': mismatch['detector_confidence'],
                    }
                    if mismatch is not None
                    else {}
                ),
            }
        )

    if detect_mismatches:
        for crop in existing_crops:
            crop_id = str(crop.get('crop_id') or crop['_id'])
            # Validated rows are earlier labels (a prior import or a human),
            # not detector output — never report them as detections.
            if crop_id in matched_crop_ids or crop.get('class_validated'):
                continue
            disagreements.append(
                {
                    'kind': DISAGREEMENT_UNMATCHED_DETECTION,
                    'crop_id': crop_id,
                    'image_path': str(image_path),
                    'bbox_norm': crop.get('bbox_norm'),
                    'detector_class_id': crop.get('class_id'),
                    'detector_class_name': crop.get('class_name'),
                    'detector_class_source': crop.get('class_source'),
                    'detector_confidence': crop.get('confidence'),
                }
            )
        if mismatch_sink is not None:
            mismatch_sink.extend(disagreements)

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


_MSEARCH_CHUNK = 100


async def _msearch_images(
    image_paths: list[str], opensearch: AsyncOpenSearch
) -> dict[str, dict[str, Any]]:
    """Batched image-doc lookup: one ``_msearch`` per
    :data:`_MSEARCH_CHUNK` paths instead of one ``search`` per file.

    Returns ``{image_path: doc}`` for every path that resolved (missing
    paths are simply absent — same "no doc" semantics as
    :func:`_lookup_image` returning ``None``).
    """
    resolved: dict[str, dict[str, Any]] = {}
    index = _images_index()
    for start in range(0, len(image_paths), _MSEARCH_CHUNK):
        chunk = image_paths[start : start + _MSEARCH_CHUNK]
        body: list[dict[str, Any]] = []
        for path in chunk:
            body.append({'index': index})
            body.append(
                {'size': 1, 'query': {'term': {'image_path': path}}, '_source': ['image_id']}
            )
        try:
            resp = await opensearch.msearch(body=body)
        except Exception as exc:
            logger.warning('label_import_msearch_failed', n=len(chunk), error=str(exc))
            continue
        for path, one in zip(chunk, resp.get('responses') or [], strict=True):
            hits = ((one or {}).get('hits') or {}).get('hits') or []
            if not hits:
                continue
            src = hits[0].get('_source') or {}
            src['_id'] = hits[0]['_id']
            resolved[path] = src
    return resolved


async def import_labels_batch(
    pairs: list[tuple[Path, Path]],
    registry: ClassRegistry,
    opensearch: AsyncOpenSearch,
    label_source: str = DEFAULT_LABEL_SOURCE,
    detect_mismatches: bool = False,
    disagreement_sink: list[dict[str, Any]] | None = None,
    image_docs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Batch-import many image+label pairs.

    Args:
        pairs: list of ``(image_path, label_txt_path)`` tuples.
        registry: ClassRegistry.
        opensearch: AsyncOpenSearch client.
        label_source: passed to :func:`import_yolo_labels`.
        detect_mismatches: passed to :func:`import_yolo_labels`; the
            per-file disagreement records are counted by kind into the
            returned summary.
        disagreement_sink: Optional list the per-file disagreement
            records are appended to.
        image_docs: Optional ``{image_path: doc}`` the caller already has
            (e.g. an ingest batch's own just-written results) —
            skips the ``_msearch`` lookup for any path present here. Any
            path NOT present is still resolved via the batched
            ``_msearch`` fallback below.

    Returns:
        ``{labels_imported, files_processed, files_failed, mismatches,
        missed_labels, unmatched_detections}`` (``mismatches`` counts
        class disagreements only).
    """
    summary = {
        'labels_imported': 0,
        'files_processed': 0,
        'files_failed': 0,
        'mismatches': 0,
        'missed_labels': 0,
        'unmatched_detections': 0,
    }
    # Resolve every remaining image doc via chunked _msearch, instead
    # of import_yolo_labels doing one `search` per file — the per-file
    # crop-lookup + bulk write (genuinely per-image-scoped) stay as-is.
    image_docs = dict(image_docs or {})
    missing_paths = [str(p) for p, _ in pairs if str(p) not in image_docs]
    if missing_paths:
        image_docs.update(await _msearch_images(missing_paths, opensearch))
    for image_path, label_path in pairs:
        try:
            sink: list[dict[str, Any]] = []
            n = await import_yolo_labels(
                image_path,
                label_path,
                registry,
                opensearch,
                label_source=label_source,
                detect_mismatches=detect_mismatches,
                mismatch_sink=sink,
                image_doc=image_docs.get(str(image_path)),
            )
            summary['labels_imported'] += n
            for key, value in count_disagreements(sink).items():
                summary[key] += value
            if disagreement_sink is not None:
                disagreement_sink.extend(sink)
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
    'DISAGREEMENT_CLASS_MISMATCH',
    'DISAGREEMENT_MISSED_LABEL',
    'DISAGREEMENT_UNMATCHED_DETECTION',
    'LABEL_IOU_MATCH',
    'count_disagreements',
    'import_labels_batch',
    'import_yolo_labels',
]

"""Curation export endpoints — split out of pipeline.py for the file-size gate.

Backs ``POST {prefix}/export/yolo`` with the generic
:class:`~src.services.curation.export.GenericYoloExportService` and lists
every materialized dataset (multi-class and single-class) for the train
page. The narrowed single-class export lives in
:mod:`src.routers.curation.export_single_class`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.routers.curation._common import (
    ExportYoloRequest,
    OpenSearchDep,
    RegistryDep,
    logger,
    router,
)
from src.services.curation.export import (
    REGISTRY_ARTIFACT_CONTENT_TYPES,
    GenericYoloExportService,
    resolve_current_export_dir,
)
from src.services.curation.export_readiness import NothingToExportError


if TYPE_CHECKING:
    from pathlib import Path


def _resolve_current_export_dir() -> Path:
    """Thin wrapper so tests can patch this module's call site, mirroring
    the reference's ``_resolve_current_vehicle_export_dir`` patch point."""
    return resolve_current_export_dir()


class ExportSkippedItems(BaseModel):
    """Validated items scrolled off the index but left out of the export,
    by reason (see :meth:`GenericYoloExportService._hits_to_rows`)."""

    no_image_id: int = Field(
        default=0, description='Had a usable box + class but no image_id to group on.'
    )
    no_usable_box_or_class: int = Field(
        default=0, description='Missing an item id, a usable box, or a class id.'
    )


@router.post('/export/yolo')
async def export_yolo(
    payload: ExportYoloRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,  # noqa: ARG001 - kept for call-site parity with the reference handler
) -> dict[str, Any]:
    """Kick off a YOLO export job.

    Backed by :class:`GenericYoloExportService` — a multi-class YOLO
    detection dataset export, one image + one label file per source image,
    with a deterministic per-image split and a reproducibility manifest.
    Synchronous; the response includes the export dir + counts
    (``image_count`` / ``split_counts`` are images, ``object_count`` /
    ``split_object_counts`` are label lines). ``422`` (``nothing to export:
    <reason>``) when no image is exportable; nothing is written then.
    """
    from pathlib import Path as _Path

    export_dir = _Path(payload.export_dir) if payload.export_dir else None
    service = GenericYoloExportService(opensearch)
    try:
        result = await service.export_dataset(
            export_dir=export_dir,
            version_tag=payload.version_tag,
            seed=payload.seed,
            max_images=payload.max_images,
            dedup_threshold=payload.dedup_threshold,
            require_fully_labeled_images=payload.require_fully_labeled_images,
        )
    except NothingToExportError as exc:
        # Nothing exportable is the caller's data state, not a server
        # fault; nothing was written and `current` still points at the
        # previous export.
        logger.warning('export_refused_empty', reason=str(exc))
        raise HTTPException(status_code=422, detail=f'nothing to export: {exc}') from exc
    except Exception as exc:
        logger.error('export_failed', error=str(exc))
        raise HTTPException(status_code=500, detail=f'export failed: {exc}') from exc
    return {
        'status': 'success',
        'export_dir': result.export_dir,
        'version_tag': result.version_tag,
        'manifest_path': result.manifest_path,
        'dataset_sha': result.dataset_sha,
        'image_count': result.image_count,
        'object_count': result.object_count,
        'split_counts': result.split_counts.to_dict(),
        'split_object_counts': result.split_object_counts.to_dict(),
        'require_fully_labeled_images': payload.require_fully_labeled_images,
        'unlabeled_items_on_exported_images': result.unlabeled_items_on_exported_images,
        'images_with_unlabeled_items': result.images_with_unlabeled_items,
        'images_dropped_not_fully_labeled': result.images_dropped_not_fully_labeled,
        'skipped_items': ExportSkippedItems(**result.skipped_items),
        'dedup': payload.dedup_threshold,
        'started_at': result.started_at or None,
        'finished_at': result.finished_at or None,
    }


# Dataset kinds on the wire — the same ids the /methods `export` axis
# advertises for POST /export/{kind}.
MULTI_CLASS_DATASET_KIND = 'yolo'
SINGLE_CLASS_DATASET_KIND = 'single_class'


def _read_manifest(d: Path) -> dict[str, Any] | None:
    from src.services.curation.export import ARTIFACT_FILENAMES

    manifest = d / ARTIFACT_FILENAMES['manifest']
    if not manifest.is_file():
        return None
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('export_manifest_skip', dir=str(d), error=str(exc))
        return None
    return meta if isinstance(meta, dict) else None


def _dataset_row(
    d: Path, meta: dict[str, Any], *, kind: str, profile_name: str | None, current: str | None
) -> dict[str, Any]:
    return {
        'kind': kind,
        'profile_name': profile_name,
        'export_dir': str(d),
        'version_tag': meta.get('version_tag') or '',
        'image_count': meta.get('image_count'),
        'object_count': meta.get('object_count'),
        'split_counts': meta.get('split_counts'),
        'dataset_sha': meta.get('dataset_sha'),
        'exported_at': meta.get('exported_at') or meta.get('started_at'),
        'class_count': meta.get('class_count'),
        'is_current': current is not None and str(d) == current,
    }


def _scan_export_datasets() -> list[dict[str, Any]]:
    """Every materialized dataset version under the export root.

    Multi-class versions live directly under the root
    (``<root>/<version>/manifest.json``). Single-class exports live one
    level down under their profile's own root
    (``<root>/<profile_name>/<version>/manifest.json``), each with its own
    ``current`` symlink, so a root child without a manifest is scanned as
    a profile root.
    """
    from src.config import get_curation_config
    from src.services.curation.export_single_class import (
        SingleClassExportProfile,
        resolve_current_single_class_dir,
    )

    cfg = get_curation_config()
    root = cfg.export_root
    if not root.is_dir():
        return []
    try:
        current: str | None = str(resolve_current_export_dir(cfg))
    except FileNotFoundError:
        current = None
    rows: list[dict[str, Any]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == 'current':
            continue
        meta = _read_manifest(d)
        if meta is not None:
            rows.append(
                _dataset_row(
                    d, meta, kind=MULTI_CLASS_DATASET_KIND, profile_name=None, current=current
                )
            )
            continue
        profile = SingleClassExportProfile(name=d.name)
        try:
            profile_current: str | None = str(resolve_current_single_class_dir(profile, cfg))
        except FileNotFoundError:
            profile_current = None
        for v in sorted(d.iterdir()):
            if not v.is_dir() or v.name == profile.current_link_name:
                continue
            vmeta = _read_manifest(v)
            if vmeta is None:
                continue
            rows.append(
                _dataset_row(
                    v,
                    vmeta,
                    kind=SINGLE_CLASS_DATASET_KIND,
                    profile_name=d.name,
                    current=profile_current,
                )
            )
    return rows


@router.get('/export/datasets')
async def list_export_datasets(
    kind: Annotated[
        str | None,
        Query(description=f"'{MULTI_CLASS_DATASET_KIND}' or '{SINGLE_CLASS_DATASET_KIND}'"),
    ] = None,
    profile_name: Annotated[
        str | None, Query(description='Only this single-class export profile.')
    ] = None,
) -> dict[str, Any]:
    """List every materialized dataset version on disk, newest first.

    Lets the train page pick ANY past export (a small sample, a larger
    subset, or the full set) rather than only the latest — the
    prerequisite for retraining / upsizing on the exact same data. Each
    row carries ``kind`` (``yolo`` multi-class / ``single_class``) and
    ``profile_name`` (``null`` for multi-class); ``is_current`` is judged
    against the row's own ``current`` symlink (the multi-class root's, or
    that single-class profile's).
    """
    datasets = _scan_export_datasets()
    if kind is not None:
        datasets = [d for d in datasets if d['kind'] == kind]
    if profile_name is not None:
        datasets = [d for d in datasets if d['profile_name'] == profile_name]
    datasets.sort(key=lambda x: x.get('exported_at') or '', reverse=True)
    return {'datasets': datasets, 'count': len(datasets)}


class ExportSplitCounts(BaseModel):
    """Counts per split (images or objects — see the field using it)."""

    train: int = 0
    val: int = 0
    test: int = 0


class ExportClassSplitCounts(ExportSplitCounts):
    """One class's object (label-line) counts per split, as recorded in the manifest."""

    class_id: int = Field(description='Registry class id.')
    export_id: int = Field(description='Dense class id written into the label files.')
    class_name: str


class ExportStatusResponse(BaseModel):
    """``GET /export/status``: the last completed multi-class export.

    ``idle`` = no export yet (every other field ``null``); ``unknown`` =
    the ``current`` export exists but its manifest is missing or
    unreadable (only ``path``/``export_dir`` set); ``success`` = every
    field below read from that export's ``manifest.json``.
    A field the manifest doesn't record (an export written before it
    existed) is ``null``.

    The export has one image file + one label file per source image:
    ``image_count`` / ``split_counts`` count images, ``object_count`` /
    ``split_object_counts`` count objects (label lines), and
    ``class_split_counts`` counts objects per class per split.
    """

    status: Literal['idle', 'unknown', 'success']
    path: str | None = Field(default=None, description='Resolved export directory.')
    export_dir: str | None = Field(default=None, description='Same as ``path``.')
    last_run: str | None = Field(default=None, description='Finish (else start) time, ISO 8601.')
    version_tag: str | None = None
    dataset_sha: str | None = None
    seed: int | None = None
    group_key: str | None = Field(
        default=None, description='Row attribute the split grouped on (``image_id``).'
    )
    image_count: int | None = Field(default=None, description='Exported images.')
    object_count: int | None = Field(
        default=None, description='Exported objects (label lines) across all images.'
    )
    class_count: int | None = None
    split_counts: ExportSplitCounts | None = Field(default=None, description='Images per split.')
    split_object_counts: ExportSplitCounts | None = Field(
        default=None, description='Objects (label lines) per split.'
    )
    class_split_counts: list[ExportClassSplitCounts] | None = Field(
        default=None, description='Objects per class per split.'
    )
    require_fully_labeled_images: bool | None = Field(
        default=None, description='Whether images with an unlabeled object were left out.'
    )
    unlabeled_items_on_exported_images: int | None = Field(
        default=None,
        description=(
            'Objects on exported images the export did not label (unreviewed, or on a class '
            'it leaves out); learned as background. Excluded / dismissed items never count.'
        ),
    )
    images_with_unlabeled_items: int | None = Field(
        default=None, description='Exported images holding at least one unlabeled object.'
    )
    images_dropped_not_fully_labeled: int | None = Field(
        default=None,
        description='Images left out by require_fully_labeled_images (0 when it was off).',
    )
    skipped_items: ExportSkippedItems | None = Field(
        default=None,
        description=(
            'Validated items left out of the export, by reason. Null for an export written '
            'before this field existed, not a fabricated zero.'
        ),
    )


def _status_from_manifest(target: Path, meta: dict[str, Any]) -> ExportStatusResponse:
    split_counts = meta.get('split_counts')
    split_objects = meta.get('split_object_counts')
    class_rows = meta.get('class_split_counts')
    skipped_items = meta.get('skipped_items')
    return ExportStatusResponse(
        status='success',
        path=str(target),
        export_dir=str(target),
        last_run=meta.get('finished_at') or meta.get('started_at'),
        version_tag=meta.get('version_tag'),
        dataset_sha=meta.get('dataset_sha'),
        seed=meta.get('seed'),
        group_key=meta.get('group_key'),
        image_count=meta.get('image_count'),
        object_count=meta.get('object_count'),
        class_count=meta.get('class_count'),
        split_counts=ExportSplitCounts(**split_counts) if isinstance(split_counts, dict) else None,
        split_object_counts=(
            ExportSplitCounts(**split_objects) if isinstance(split_objects, dict) else None
        ),
        require_fully_labeled_images=meta.get('require_fully_labeled_images'),
        unlabeled_items_on_exported_images=meta.get('unlabeled_items_on_exported_images'),
        images_with_unlabeled_items=meta.get('images_with_unlabeled_items'),
        images_dropped_not_fully_labeled=meta.get('images_dropped_not_fully_labeled'),
        skipped_items=(
            ExportSkippedItems(**skipped_items) if isinstance(skipped_items, dict) else None
        ),
        class_split_counts=(
            [ExportClassSplitCounts(**row) for row in class_rows]
            if isinstance(class_rows, list)
            else None
        ),
    )


@router.get('/export/status', response_model=ExportStatusResponse)
async def export_status() -> ExportStatusResponse:
    """Last completed export — the ``current`` symlink's manifest, if any."""
    try:
        target = _resolve_current_export_dir()
    except FileNotFoundError:
        return ExportStatusResponse(status='idle')
    meta = _read_manifest(target)
    if meta is None:
        return ExportStatusResponse(status='unknown', path=str(target), export_dir=str(target))
    try:
        return _status_from_manifest(target, meta)
    except (TypeError, ValueError) as exc:
        logger.warning('export_manifest_malformed', dir=str(target), error=str(exc))
        return ExportStatusResponse(status='unknown', path=str(target), export_dir=str(target))


@router.get('/export/registry/{artifact}')
async def export_registry_artifact(artifact: str) -> FileResponse:
    """Serve one frozen artifact from the *current* export snapshot.

    Serves ``class_registry.json``, ``data.yaml``, ``manifest.json``, and
    ``label_stats.json`` byte-for-byte from disk, read-only, next to the
    export they were produced with — never the live, possibly-since-changed
    class registry. This matters because these files must be read as one
    mutually consistent set: pairing a live-registry read with a frozen
    ``data.yaml`` could describe a mismatched dataset (different class ids,
    different dense export-id remap) than the one actually exported. Callers
    that need the current, possibly-newer registry should use the live
    registry endpoints instead of this one.

    ``artifact`` is deliberately a plain ``str`` (not a ``Literal``) so an
    unknown value gets a clean 404 here rather than FastAPI's 422. The
    whitelist check happens before any filesystem path is built, and the
    resolved path is re-verified to be inside the real export directory as
    defense-in-depth against a future whitelist-loosening mistake.
    """
    content_type = REGISTRY_ARTIFACT_CONTENT_TYPES.get(artifact)
    if content_type is None:
        raise HTTPException(status_code=404, detail=f'unknown export artifact: {artifact!r}')

    try:
        export_dir = _resolve_current_export_dir()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail='no export available — run POST /curation/export/yolo first',
        ) from exc

    candidate = (export_dir / artifact).resolve()
    if export_dir not in candidate.parents and candidate != export_dir:
        # Defense-in-depth: even though `artifact` came from a fixed
        # whitelist above, never serve anything that resolves outside the
        # export root.
        raise HTTPException(status_code=404, detail=f'unknown export artifact: {artifact!r}')

    if not candidate.is_file():
        logger.warning(
            'export_registry_artifact_missing',
            artifact=artifact,
            export_dir=str(export_dir),
        )
        raise HTTPException(
            status_code=404,
            detail=f'export exists but is missing artifact: {artifact}',
        )

    return FileResponse(
        path=str(candidate),
        media_type=content_type,
        filename=artifact,
        headers={'Cache-Control': 'no-store'},
    )

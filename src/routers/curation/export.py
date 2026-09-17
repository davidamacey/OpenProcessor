"""Curation export endpoints — split out of pipeline.py for the file-size gate.

Backs ``POST /curation/export/yolo`` with the generic
:class:`~src.services.curation.export.GenericYoloExportService` (plan
§3.5/§5 Chunk 9 — the generic half of the reference export router). The
reference implementation's standalone single-class region-dataset
export (``POST /legacy/export/lpr``) is backed by a domain-specific export
service (plan §1's Bucket B — proprietary dataset-family logic, never
ported anywhere in this plan) — so that endpoint is intentionally not
ported here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

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


if TYPE_CHECKING:
    from pathlib import Path


def _resolve_current_export_dir() -> Path:
    """Thin wrapper so tests can patch this module's call site, mirroring
    the reference's ``_resolve_current_vehicle_export_dir`` patch point."""
    return resolve_current_export_dir()


@router.post('/export/yolo')
async def export_yolo(
    payload: ExportYoloRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,  # noqa: ARG001 - kept for call-site parity with the reference handler
) -> dict[str, Any]:
    """Kick off a YOLO export job.

    Backed by :class:`GenericYoloExportService` — a multi-class YOLO
    detection dataset export with a deterministic split and a
    reproducibility manifest. Synchronous; the response includes the
    export dir + counts.
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
        )
    except Exception as exc:
        logger.error('export_failed', error=str(exc))
        raise HTTPException(status_code=500, detail=f'export failed: {exc}') from exc
    return {
        'status': 'success',
        'export_dir': result.export_dir,
        'version_tag': result.version_tag,
        'manifest_path': result.manifest_path,
        'dataset_sha': result.dataset_sha,
        'split_counts': result.split_counts.to_dict(),
        'dedup': payload.dedup_threshold,
        'started_at': result.started_at or None,
        'finished_at': result.finished_at or None,
    }


@router.get('/export/datasets')
async def list_export_datasets() -> dict[str, Any]:
    """List every materialized dataset version on disk, newest first.

    Scans the export root for ``manifest.json`` so the train page can let
    the operator pick ANY past export (a small sample, a larger subset, or
    the full set) rather than only the latest — the prerequisite for
    retraining / upsizing the model on the exact same data.
    """
    from src.config import get_curation_config
    from src.services.curation.export import ARTIFACT_FILENAMES

    cfg = get_curation_config()
    try:
        current = str(resolve_current_export_dir(cfg))
    except FileNotFoundError:
        current = None

    datasets: list[dict[str, Any]] = []
    root = cfg.export_root
    if root.is_dir():
        for d in root.iterdir():
            if not d.is_dir() or d.name == 'current':
                continue
            manifest = d / ARTIFACT_FILENAMES['manifest']
            if not manifest.is_file():
                continue
            try:
                meta = json.loads(manifest.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                logger.warning('export_manifest_skip', dir=str(d), error=str(exc))
                continue
            datasets.append(
                {
                    'export_dir': str(d),
                    'version_tag': meta.get('version_tag') or '',
                    'image_count': meta.get('image_count'),
                    'split_counts': meta.get('split_counts'),
                    'dataset_sha': meta.get('dataset_sha'),
                    'exported_at': meta.get('exported_at') or meta.get('started_at'),
                    'class_count': meta.get('class_count'),
                    'is_current': current is not None and str(d) == current,
                }
            )
    datasets.sort(key=lambda x: x.get('exported_at') or '', reverse=True)
    return {'datasets': datasets, 'count': len(datasets)}


@router.get('/export/status')
async def export_status() -> dict[str, Any]:
    """Last-export status — reads ``current`` symlink + manifest if present."""
    from src.services.curation.export import ARTIFACT_FILENAMES

    try:
        target = _resolve_current_export_dir()
    except FileNotFoundError:
        return {'status': 'idle', 'last_run': None}
    manifest = target / ARTIFACT_FILENAMES['manifest']
    if not manifest.exists():
        return {'status': 'unknown', 'last_run': None, 'export_dir': str(target)}
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('export_manifest_read_failed', error=str(exc))
        return {'status': 'unknown', 'last_run': None, 'export_dir': str(target)}
    return {
        'status': 'success',
        'export_dir': str(target),
        'last_run': meta.get('finished_at') or meta.get('started_at'),
        'dataset_sha': meta.get('dataset_sha'),
        'class_count': meta.get('class_count'),
    }


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

"""Narrowed single-class / class-subset dataset export endpoints.

Sibling to :mod:`src.routers.curation.export` (the multi-class
``POST /export/yolo``), following the same shape deliberately: a
synchronous ``POST`` that returns the finished export's envelope, and a
``GET .../status`` that reports the last run by resolving the export's
``current`` symlink and reading the manifest beside it. That status
pattern is the one the multi-class export already uses — the symlink IS
the job state, which is why it survives an API restart and why two
concurrent readers can never disagree about which directory is current.

Unlike the multi-class export, this one is *profile-scoped*: each
``profile_name`` gets its own output root and its own ``current``
symlink, so a plate-detector export and a signage-detector export
coexist without either clobbering the other or the full multi-class
dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Query

from src.routers.curation._common import ExportSingleClassRequest, OpenSearchDep, logger, router
from src.services.curation.export_readiness import NothingToExportError
from src.services.curation.export_single_class import (
    MANIFEST_FILENAME,
    SingleClassExportProfile,
    SingleClassExportService,
    resolve_current_single_class_dir,
)


def _profile_from_request(payload: ExportSingleClassRequest) -> SingleClassExportProfile:
    return SingleClassExportProfile(
        name=payload.profile_name,
        class_ids=tuple(payload.class_ids),
        box_source=payload.box_source,
        region_class_name=payload.region_class_name,
    )


def _resolve_current_dir(profile_name: str) -> Path:
    """Thin wrapper so tests can patch this module's call site, mirroring
    ``export.py``'s ``_resolve_current_export_dir``."""
    return resolve_current_single_class_dir(SingleClassExportProfile(name=profile_name))


@router.post('/export/single_class')
async def export_single_class(
    payload: ExportSingleClassRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Build a narrowed YOLO dataset for one class or a class subset.

    Returns the export's full integrity envelope — ``dataset_sha`` over
    the written label content, ``frozen_test_sha`` over the test split's
    identity, and the path the ``current`` symlink now points at — so a
    caller can record exactly which dataset a training run consumed
    without re-reading the directory.
    """
    service = SingleClassExportService(opensearch, profile=_profile_from_request(payload))
    try:
        result = await service.export(
            export_dir=Path(payload.export_dir) if payload.export_dir else None,
            version_tag=payload.version_tag,
            seed=payload.seed,
            skip_test_split=payload.skip_test_split,
            empty_bg_ratio=payload.empty_bg_ratio,
            max_positive_images=payload.max_positive_images,
            dedup_threshold=payload.dedup_threshold,
            image_mode=payload.image_mode,
            img_max_side=payload.img_max_side,
            copy_images=payload.copy_images,
        )
    except NothingToExportError as exc:
        # Nothing matched: refused before any directory or symlink is written.
        logger.warning('single_class_export_refused_empty', reason=str(exc))
        raise HTTPException(status_code=422, detail=f'nothing to export: {exc}') from exc
    except ValueError as exc:
        # Configuration the caller can fix (empty class_ids, an image_mode
        # that doesn't apply to the chosen box_source) — a 422, not a 500.
        logger.warning('single_class_export_rejected', error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error('single_class_export_failed', error=str(exc))
        raise HTTPException(status_code=500, detail=f'single-class export failed: {exc}') from exc

    return {
        'status': 'success',
        'export_dir': result.export_dir,
        'version_tag': result.version_tag,
        'manifest_path': result.manifest_path,
        'data_yaml_path': result.data_yaml_path,
        'dataset_sha': result.dataset_sha,
        'frozen_test_sha': result.frozen_test_sha,
        'split_counts': result.split_counts.to_dict(),
        'image_count': result.image_count,
        'class_count': result.class_count,
        'positive_images': result.positive_images,
        'background_images': result.background_images,
        # A dataset with no positives cannot train a detector. Surfaced on
        # the response as well as the manifest so a caller that never opens
        # the export directory still sees it.
        'positives_zero_warning': result.positive_images == 0,
        'current_symlink': result.current_symlink,
        'started_at': result.started_at or None,
        'finished_at': result.finished_at or None,
    }


@router.get('/export/single_class/status')
async def export_single_class_status(
    profile_name: str = Query(
        'single_class',
        description="Export profile to report on — matches the POST body's profile_name.",
    ),
) -> dict[str, Any]:
    """Last narrowed-export status for one profile.

    Same contract as ``GET /export/status``: ``idle`` when the profile has
    never produced an export, ``unknown`` when the directory exists but
    its manifest is missing or unreadable, ``success`` otherwise.
    """
    try:
        target = _resolve_current_dir(profile_name)
    except FileNotFoundError:
        return {'status': 'idle', 'last_run': None, 'profile_name': profile_name}

    manifest = target / MANIFEST_FILENAME
    if not manifest.exists():
        return {
            'status': 'unknown',
            'last_run': None,
            'profile_name': profile_name,
            'export_dir': str(target),
        }
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('single_class_export_manifest_read_failed', error=str(exc))
        return {
            'status': 'unknown',
            'last_run': None,
            'profile_name': profile_name,
            'export_dir': str(target),
        }
    return {
        'status': 'success',
        'profile_name': profile_name,
        'export_dir': str(target),
        'last_run': meta.get('finished_at') or meta.get('started_at'),
        'dataset_kind': meta.get('dataset_kind'),
        'dataset_sha': meta.get('dataset_sha'),
        'frozen_test_sha': meta.get('frozen_test_sha'),
        'class_count': meta.get('class_count'),
        'class_names': meta.get('class_names'),
        'image_count': meta.get('image_count'),
        'positive_images': meta.get('positive_images'),
        'background_images': meta.get('background_images'),
        'false_positive_background_images': meta.get('false_positive_background_images'),
        'positives_zero_warning': meta.get('positives_zero_warning'),
        'split_counts': meta.get('split_counts'),
    }

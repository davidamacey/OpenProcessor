"""Export lineage + build identity, read once at ``POST /curation/train/start``.

Two things a run manifest needs to stay reproducible and comparable:

1. **Which export it trained on** (:class:`ExportIdentity`) — read straight
   from the export's own ``manifest.json``. ``dataset_sha`` is the export's
   own claim and is never recomputed here (there is no independent way to
   verify it without re-scrolling the source index); ``frozen_test_sha`` and
   ``test_label_sha`` describe the frozen test split's on-disk identity and
   content respectively, and an export written before this wave lacks them
   in its manifest -- so they are computed from disk when missing, meaning
   every *new* run records them even against an old export.
2. **Which code built the images that ran it** (:func:`stamp_code_versions`)
   -- the API's own build sha, plus the trainer container's image id and
   baked-in revision label, read over the docker socket when one is
   configured and reachable.

Both are best-effort: a failure anywhere here must never block a training
submission, so every failure path returns ``None`` fields rather than
raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_gpu_arbiter_config
from src.core.logging import get_logger
from src.services.curation.export_support import _code_sha, frozen_test_sha_of, label_content_sha
from src.services.training.gpu_arbiter import _docker_client


logger = get_logger(__name__)


@dataclass(frozen=True)
class ExportIdentity:
    """Lineage facts read off an export directory's ``manifest.json``."""

    dataset_sha: str | None
    frozen_test_sha: str | None
    test_label_sha: str | None
    version_tag: str | None


def _load_manifest(export_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))
    except Exception as exc:  # missing/corrupt manifest -- every field falls back to None
        logger.warning('lineage_manifest_read_failed', export_dir=str(export_dir), error=str(exc))
        return {}


def read_export_identity(export_dir: str | Path) -> ExportIdentity:
    """Read (and best-effort backfill) the lineage fields of one export.

    ``frozen_test_sha`` / ``test_label_sha`` missing from an older
    manifest are computed from the on-disk ``labels/test/`` split so every
    *new* run still records them; ``dataset_sha`` is never computed --
    it's the export's own claim, and missing means ``None``.
    """
    root = Path(export_dir)
    manifest = _load_manifest(root)

    frozen_test_sha = manifest.get('frozen_test_sha') or None
    if frozen_test_sha is None:
        frozen_test_sha = frozen_test_sha_of(root) or None

    test_label_sha = manifest.get('test_label_sha') or None
    if test_label_sha is None:
        test_label_sha = label_content_sha(root, None, truncate=16, split='test') or None

    return ExportIdentity(
        dataset_sha=manifest.get('dataset_sha') or None,
        frozen_test_sha=frozen_test_sha,
        test_label_sha=test_label_sha,
        version_tag=manifest.get('version_tag') or None,
    )


def stamp_code_versions() -> dict[str, str | None]:
    """Best-effort code provenance stamped at submit time.

    ``api_sha`` is this API process's own build identity (``_code_sha()``,
    ``'unknown'`` mapped to ``None`` since "we couldn't determine it" should
    read as absent, not as the literal string ``'unknown'`` in every run's
    lineage). ``trainer_image_id`` / ``trainer_image_revision`` come from
    the docker socket, when :class:`~src.config.gpu_arbiter.GpuArbiterConfig`
    names a trainer container and the socket is reachable -- any failure
    (no socket, container not running, unconfigured) logs at info and
    leaves both ``None`` rather than raising.
    """
    api_sha = _code_sha()
    result: dict[str, str | None] = {
        'api_sha': None if api_sha == 'unknown' else api_sha,
        'trainer_image_id': None,
        'trainer_image_revision': None,
    }

    container_name = get_gpu_arbiter_config().trainer_container
    if not container_name:
        return result

    client = _docker_client()
    if client is None:
        return result

    try:
        container = client.containers.get(container_name)
        image = container.image
        result['trainer_image_id'] = image.id
        result['trainer_image_revision'] = (image.labels or {}).get(
            'org.opencontainers.image.revision'
        ) or None
    except Exception as exc:
        logger.info('lineage_trainer_image_lookup_failed', container=container_name, error=str(exc))
    return result


__all__ = ['ExportIdentity', 'read_export_identity', 'stamp_code_versions']

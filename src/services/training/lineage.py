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
2. **Which API build submitted it** (:func:`api_sha`).

Both are best-effort: a failure anywhere here must never block a training
submission, so every failure path returns ``None`` fields rather than
raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.services.curation.export_support import _code_sha, frozen_test_sha_of, label_content_sha


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
        typed: dict[str, Any] = json.loads(
            (export_dir / 'manifest.json').read_text(encoding='utf-8')
        )
        return typed
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


def api_sha() -> str | None:
    """This API process's build identity; ``None`` when it cannot be determined.

    ``_code_sha()`` answers ``'unknown'`` for "could not tell", which would read as a real value
    in every run's lineage, so it maps to absent. The trainer stamps its own image revision from
    its environment when it writes the manifest.
    """
    sha = _code_sha()
    return None if sha == 'unknown' else sha


__all__ = ['ExportIdentity', 'api_sha', 'read_export_identity']

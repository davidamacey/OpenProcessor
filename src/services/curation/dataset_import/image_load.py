"""Getting one dataset image's bytes into the project (W10.3).

An image under a configured source root (or the upload root) is ingested at
its own path. One that is only readable because it sits under the project's
export root (an OpenProcessor export re-imported where it was written) is
first copied under ``<upload_root>/datasets/<source_sha[:16]>/<rel_path>``:
export directories are pruned by retention, so a stored ``image_path`` must
never point into one.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.paths import resolve_ref
from src.services.curation.image_serving import is_servable_image_path


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.scan import ScanEntry

MAX_IMAGE_BYTES = 256 * 1024**2


class ImageLoadError(Exception):
    """The image cannot be read, is too large, or its path is not allowed."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def stored_path(ctx: ImportContext, entry: ScanEntry) -> Path:
    """The path the images doc will carry for ``entry``."""
    resolved = entry.abs_image_path.resolve()
    if is_servable_image_path(str(resolved)):
        return resolved
    dest_root = ctx.upload_root / 'datasets' / ctx.source_sha[:16]
    dest = resolve_ref(dest_root, entry.rel_path)
    if dest is None:
        raise ImageLoadError('image_path_not_servable', f'{entry.rel_path}: unsafe destination')
    return dest


def load_image_bytes(ctx: ImportContext, entry: ScanEntry) -> tuple[bytes, Path]:
    """``(bytes, stored_path)``; copies an export-root image into the upload
    root the first time. Raises :class:`ImageLoadError`."""
    source = entry.abs_image_path.resolve()
    dest = stored_path(ctx, entry)
    try:
        if source.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageLoadError('image_unreadable', f'{entry.rel_path}: over the image size cap')
        data = source.read_bytes()
    except OSError as exc:
        raise ImageLoadError('image_unreadable', f'{entry.rel_path}: {exc}') from exc
    if dest != source and not dest.exists():
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f'.{dest.name}.tmp')
            shutil.copyfile(source, tmp)
            tmp.replace(dest)
        except OSError as exc:
            raise ImageLoadError(
                'image_unreadable', f'{entry.rel_path}: copy failed: {exc}'
            ) from exc
    return data, dest


__all__ = ['MAX_IMAGE_BYTES', 'ImageLoadError', 'load_image_bytes', 'stored_path']

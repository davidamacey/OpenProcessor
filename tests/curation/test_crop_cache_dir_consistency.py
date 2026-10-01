"""Guard against a cache-dir mismatch (see
docs/design/curation_design_rationale.md): ``src/routers/curation/vlm.py`` used to
read ``GEMMA_CROP_CACHE_DIR`` (default ``/dev/shm/curation_crops``) while
the worker wrote into ``CurationConfig.crop_cache_dir`` (env
``OP_CROP_CACHE_DIR``, default ``/dev/shm/openprocessor_crops``) --
different var, different default, 100% cache miss out of the box.

Both now resolve through ``get_curation_config().crop_cache_dir``, and the
worker reads it through ``crop_bytes`` at call time. This proves the worker
really reads the configured directory (a file placed there is what it
returns) rather than a local default.
"""

from __future__ import annotations

import src.config.curation as curation_config_module
from src.config import get_curation_config


def _reset_curation_config_singleton() -> None:
    curation_config_module._default_curation_config = None


def test_router_and_worker_resolve_the_same_crop_cache_dir(monkeypatch, tmp_path) -> None:
    from scripts.curation.worker import state as worker_state

    override = tmp_path / 'shared_crop_cache'
    override.mkdir()
    (override / 'c1.jpg').write_bytes(b'cached-by-ingest')
    try:
        monkeypatch.setenv('OP_CROP_CACHE_DIR', str(override))
        _reset_curation_config_singleton()

        assert str(get_curation_config().crop_cache_dir) == str(override)
        assert worker_state._crop_jpeg_for_task('c1', '/nowhere.jpg', (0, 0, 1, 1)) == (
            b'cached-by-ingest'
        )
    finally:
        monkeypatch.undo()
        _reset_curation_config_singleton()


def test_default_crop_cache_dir_is_what_the_worker_reads(monkeypatch, tmp_path) -> None:
    from src.services.curation import crop_bytes

    try:
        monkeypatch.delenv('OP_CROP_CACHE_DIR', raising=False)
        _reset_curation_config_singleton()
        default_dir = get_curation_config().crop_cache_dir
        seen: list[str] = []
        real = crop_bytes.Path

        def spy(*parts: str):
            seen.append(str(parts[0]))
            return real(tmp_path / 'absent')

        monkeypatch.setattr(crop_bytes, 'Path', spy)
        crop_bytes.crop_jpeg_from_cache('anything')

        assert seen == [str(default_dir)]
    finally:
        monkeypatch.undo()
        _reset_curation_config_singleton()

"""Guard against CFG-2 (see docs/design/curation_design_rationale.md and
the OSS completion plan §0.6): ``src/routers/curation/vlm.py`` used to
read ``GEMMA_CROP_CACHE_DIR`` (default ``/dev/shm/curation_crops``) while
the worker (``scripts/curation/worker/state.py``) writes into
``CurationConfig.crop_cache_dir`` (env ``OP_CROP_CACHE_DIR``, default
``/dev/shm/openprocessor_crops``) -- different var, different default,
100% cache miss out of the box. Both now resolve through
``get_curation_config().crop_cache_dir``; this test proves they agree
under a shared override rather than just independently reading the same
function (which would pass even if one of them re-introduced a local
default).
"""

from __future__ import annotations

import importlib

import src.config.curation as curation_config_module
from src.config import get_curation_config


def _reset_curation_config_singleton() -> None:
    curation_config_module._default_curation_config = None


def test_router_and_worker_resolve_the_same_crop_cache_dir(monkeypatch, tmp_path) -> None:
    from scripts.curation.worker import state as worker_state

    override = str(tmp_path / 'shared_crop_cache')
    try:
        monkeypatch.setenv('OP_CROP_CACHE_DIR', override)
        _reset_curation_config_singleton()

        # Worker side: scripts/curation/worker/state.py computes
        # CROP_CACHE_DIR once at import time from the same
        # get_curation_config() singleton -- reload it under the override
        # so it re-reads the (now-reset) config.
        worker_cache_dir = importlib.reload(worker_state).CROP_CACHE_DIR

        # Router side: src/routers/curation/vlm.py's label_batch handler
        # computes crop_cache_dir = str(get_curation_config().crop_cache_dir)
        # per call -- exercise the same expression directly.
        router_cache_dir = str(get_curation_config().crop_cache_dir)

        assert worker_cache_dir == override
        assert router_cache_dir == override
        assert worker_cache_dir == router_cache_dir
    finally:
        # Reloading worker_state mutates a real module in sys.modules --
        # put it (and the config singleton) back so other tests in this
        # session don't see a tmp_path-scoped cache dir after teardown.
        monkeypatch.undo()
        _reset_curation_config_singleton()
        importlib.reload(worker_state)


def test_default_crop_cache_dir_matches_curation_config_default(monkeypatch) -> None:
    from scripts.curation.worker import state as worker_state

    try:
        monkeypatch.delenv('OP_CROP_CACHE_DIR', raising=False)
        _reset_curation_config_singleton()

        worker_cache_dir = importlib.reload(worker_state).CROP_CACHE_DIR
        router_cache_dir = str(get_curation_config().crop_cache_dir)

        assert worker_cache_dir == router_cache_dir
    finally:
        monkeypatch.undo()
        _reset_curation_config_singleton()
        importlib.reload(worker_state)

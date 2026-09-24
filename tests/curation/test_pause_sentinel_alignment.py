"""S-4: the GPU-arbiter pause sentinel writer and readers must agree on a path.

Before this fix, ``gpu_arbiter._default_sentinel_path()`` defaulted to
``{state_dir}/training_worker/pause.sentinel`` while the two readers
(``scripts/curation/worker/state.py``'s ``DEFAULT_PAUSE_SENTINEL`` and
``scripts/curation/vlm_worker.py``'s ``--pause-sentinel`` default) both
defaulted to ``{state_dir}/vlm_worker/pause.sentinel`` -- a single-GPU
training claim never actually paused either worker. All three now
resolve through ``CurationConfig.pause_sentinel_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.config.curation import CurationConfig, get_curation_config
from src.services.training import gpu_arbiter as ga


def test_pause_sentinel_path_is_vlm_worker_prefixed() -> None:
    cfg = CurationConfig(state_dir=Path('/tmp/state'))
    assert cfg.pause_sentinel_path == Path('/tmp/state/vlm_worker/pause.sentinel')


def test_gpu_arbiter_default_sentinel_matches_curation_config(
    monkeypatch,
) -> None:
    """The writer (gpu_arbiter) must resolve to exactly the same path the
    curation config exposes -- this was the actual bug: gpu_arbiter had
    its own independent 'training_worker/pause.sentinel' literal.
    """
    custom_state_dir = Path('/tmp/custom_state')
    monkeypatch.setattr(ga, '_state_dir', lambda: custom_state_dir)

    resolved = ga._default_sentinel_path()

    assert resolved == CurationConfig(state_dir=custom_state_dir).pause_sentinel_path
    assert 'training_worker' not in str(resolved)
    assert 'vlm_worker' in str(resolved)


def test_worker_state_default_pause_sentinel_matches_curation_config() -> None:
    """scripts/curation/worker/state.py's module-level default is fixed at
    import time from get_curation_config() -- assert it resolved to the
    same property gpu_arbiter now uses (both default to the same
    OP_STATE_DIR-derived state_dir absent an env override).
    """
    scripts_dir = Path(__file__).resolve().parents[2] / 'scripts' / 'curation'
    if str(scripts_dir.parent.parent) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent.parent))
    from scripts.curation.worker import state as worker_state

    assert get_curation_config().pause_sentinel_path == worker_state.DEFAULT_PAUSE_SENTINEL

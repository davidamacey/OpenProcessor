"""Startup guard for retired env-var names (naming-sweep D4).

Every rename in ``docs/design/naming_sweep_plan.md`` section 3 is a
**clean break** -- no aliases are read. A leftover retired name is not
silently ignored (that would look like a working config that quietly
does nothing, e.g. a stale ``SAM3_URL`` turning the segmenter off with
no error): :func:`reject_retired_env`, called at every process entry
point (``src/main.py``'s lifespan, ``scripts/curation/worker/__main__.py``,
``scripts/curation/vlm_worker.py``'s ``main()``, the bake-off evaluator's
``scripts/curation/bakeoff/bakeoff_runner.py`` ``main()``), raises
``RuntimeError`` naming the replacement for every retired name still set.

This module necessarily spells the retired names, so it (and its test)
are allowlisted from the W8 naming-leak scan.
"""

from __future__ import annotations

import os


# Old name -> new name. Every spelling from plan section 3 that a
# previous wave (W1: the OP_VLM_* renames) did not already retire. A value
# starting with ``removed:`` marks a setting with no one-to-one replacement
# and says what to use instead.
_REMOVED = 'removed:'
_BAKEOFF_TARGET_REMOVED = (
    f'{_REMOVED} bake-offs score every class; restrict with OP_BAKEOFF_PROFILE_CLASS_FILTER'
)
RETIRED_ENV: dict[str, str] = {
    'VLM_URL': 'OP_VLM_URL',
    'GEMMA_URL': 'OP_VLM_URL',
    'OPENWEBUI_BASE_URL': 'OP_VLM_URL',
    'OPENWEBUI_MODEL': 'OP_VLM_MODEL',
    'OPENWEBUI_API_KEY': 'OP_VLM_API_KEY',
    'VLM_IMAGES_PER_CALL': 'OP_VLM_OPEN_IMAGES_PER_CALL',
    'GEMMA_IMAGES_PER_CALL': 'OP_VLM_OPEN_IMAGES_PER_CALL',
    'VLM_HTTPX_MAX_CONNECTIONS': 'OP_VLM_HTTPX_MAX_CONNECTIONS',
    'GEMMA_HTTPX_MAX_CONNECTIONS': 'OP_VLM_HTTPX_MAX_CONNECTIONS',
    'VLM_HTTPX_KEEPALIVE': 'OP_VLM_HTTPX_KEEPALIVE',
    'GEMMA_HTTPX_KEEPALIVE': 'OP_VLM_HTTPX_KEEPALIVE',
    'SAM3_URL': 'OP_SEGMENTER_URL',
    'SAM3_URLS': 'OP_SEGMENTER_URLS',
    'SAM3_HTTPX_MAX_CONNECTIONS': 'OP_SEGMENTER_HTTPX_MAX_CONNECTIONS',
    'SAM3_HTTPX_KEEPALIVE': 'OP_SEGMENTER_HTTPX_KEEPALIVE',
    'SAM3_SKIP_VLM_VERIFY_SCORE': 'OP_SEGMENTER_SKIP_VERIFY_SCORE',
    'SAM3_SKIP_GEMMA_VERIFY_SCORE': 'OP_SEGMENTER_SKIP_VERIFY_SCORE',
    'SAM_WORKER_VLM_CONCURRENCY': 'OP_REGION_WORKER_VLM_CONCURRENCY',
    'SAM_WORKER_GEMMA_CONCURRENCY': 'OP_REGION_WORKER_VLM_CONCURRENCY',
    'SAM_WORKER_VLM_VISIBLE_CONCURRENCY': 'OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY',
    'SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY': 'OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY',
    'SAM_WORKER_METRICS_PORT': 'OP_REGION_WORKER_METRICS_PORT',
    'OP_REGION_DETECTION_SAM_TEXT_PROMPT': 'OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT',
    'GEMMA_CROP_CACHE_DIR': 'OP_CROP_CACHE_DIR',
    # docs/design/generic_model_comparison_plan.md (W3): profiles lost the
    # single target class.
    'OP_BAKEOFF_PROFILE_TARGET_CLASS_ID': _BAKEOFF_TARGET_REMOVED,
    'OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME': _BAKEOFF_TARGET_REMOVED,
}


def reject_retired_env() -> None:
    """Fail loudly if any retired env var (see :data:`RETIRED_ENV`) is set.

    Raises ``RuntimeError`` listing every retired name still set and its
    replacement. A no-op when none are set.
    """
    stale = sorted(name for name in RETIRED_ENV if name in os.environ)
    if not stale:
        return
    renames = '; '.join(
        f'{name} was {RETIRED_ENV[name]}'
        if RETIRED_ENV[name].startswith(_REMOVED)
        else f'{name} was renamed to {RETIRED_ENV[name]}'
        for name in stale
    )
    msg = f'retired env var(s) set: {renames}'
    raise RuntimeError(msg)

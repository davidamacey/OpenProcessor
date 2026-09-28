"""Curation detection worker package.

Split into focused submodules (9 files, 3458 LOC combined). The top-level
shim file is preserved as a re-export so legacy invocation shapes
(``python -m scripts.curation.region_worker_main`` and direct file runs)
keep working — see ``scripts/curation/region_worker_main.py``.

Sub-modules:
    state        — constants, ``_ItemTask`` dataclass, crop IO helpers
    cascade      — pending fetch, ``SegmenterClient`` re-export, geometry helpers
    verify       — VLM verify + region doc builders + auto-confirm + verdicts_to_boxes
    bulk_writer  — ``_bulk_update`` + ``_publish_region_events``
    client       — SAM3 HTTP client with circuit breaker
    runner       — the long-running ``run()`` entry point (the live
                   multi-box pipeline; the pre-W8 single-candidate
                   ``_process_crop``/``combined.py`` cohort path was
                   deleted once this became the only production cascade)
    __main__     — ``parse_args`` + ``main`` for ``python -m`` invocation

The shim file uses ``from scripts.curation.worker import *`` to pull
in the **public** names below (Python's ``*`` skips underscored names
unless ``__all__`` is defined, which it deliberately is not here). The
shim then imports underscored helpers it needs *directly from the
sub-modules* — so this ``__init__`` does NOT need to re-export private
helpers like ``_ItemTask`` / ``_bulk_update``.
"""

from __future__ import annotations

# Public, top-level re-exports. These are needed because:
#   1. ``scripts/curation/region_worker_main.py`` does ``from ... import *`` —
#      these names are what ``*`` picks up.
#   2. ``tests/curation/test_region_worker.py`` monkeypatches the heavy-IO
#      constructors (``AsyncTritonPool``, ``make_script_opensearch``,
#      ``SegmenterClient``, ``VlmLabeler``) on the shim module; the runner
#      looks them up via the shim (``from scripts.curation import
#      region_worker_main as _wkr; _wkr.AsyncTritonPool(...)``), so the
#      patch must reach the shim's namespace.
from scripts.curation.worker.cascade import SegmenterClient  # noqa: F401
from scripts.curation.worker.runner import run  # noqa: F401
from scripts.curation.worker.state import (  # noqa: F401
    DEFAULT_OPENSEARCH,
    DEFAULT_PAUSE_SENTINEL,
    DEFAULT_SEGMENTER_URL,
    DEFAULT_SEGMENTER_URLS,
    DEFAULT_TRITON,
    DEFAULT_VLM_URL,
    JPEG_QUALITY,
    STATUS_PENDING_DETECTION,
    STATUS_PENDING_VERIFICATION,
    items_index,
)
from src.clients.triton_pool import AsyncTritonPool  # noqa: F401
from src.services.labeling.vlm_labeler import VlmLabeler  # noqa: F401
from src.services.projects.guard import make_script_opensearch  # noqa: F401

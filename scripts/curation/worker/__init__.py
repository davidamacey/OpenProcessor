"""Curation detection worker package.

Ported from the reference detection-worker package (9 files,
3458 LOC). The top-level shim file is preserved as a re-export so
legacy invocation shapes (``python -m scripts.curation.sam_worker_main``
and direct file runs) keep working — see
``scripts/curation/sam_worker_main.py``.

Sub-modules:
    state        — constants, ``_ItemTask`` dataclass, crop IO helpers
    cascade      — pending fetch, ``Sam3Client``, geometry helpers, ``_process_crop``
    verify       — VLM verify + region doc builders + auto-confirm
    bulk_writer  — ``_bulk_update`` + ``_publish_region_events``
    combined     — B-PR5 combined class+region+OCR cohort routing
    client       — SAM3 HTTP client with circuit breaker
    runner       — the long-running ``run()`` entry point
    __main__     — ``parse_args`` + ``main`` for ``python -m`` invocation

The shim file uses ``from scripts.curation.worker import *`` to pull
in the **public** names below (Python's ``*`` skips underscored names
unless ``__all__`` is defined, which it deliberately is not here). The
shim then imports underscored helpers it needs *directly from the
sub-modules* — so this ``__init__`` does NOT need to re-export private
helpers like ``_ItemTask`` / ``_process_crop`` / ``_bulk_update``.
"""

from __future__ import annotations

# Public, top-level re-exports. These are needed because:
#   1. ``scripts/curation/sam_worker_main.py`` does ``from ... import *`` —
#      these names are what ``*`` picks up.
#   2. ``tests/curation/test_sam_worker.py`` monkeypatches the heavy-IO
#      constructors (``AsyncTritonPool``, ``AsyncOpenSearch``,
#      ``Sam3Client``, ``VlmLabeler``) on the shim module; the runner
#      looks them up via the shim (``from scripts.curation import
#      sam_worker_main as _wkr; _wkr.AsyncTritonPool(...)``), so the
#      patch must reach the shim's namespace.
from opensearchpy import AsyncOpenSearch  # noqa: F401

from scripts.curation.worker.cascade import Sam3Client  # noqa: F401
from scripts.curation.worker.runner import run  # noqa: F401
from scripts.curation.worker.state import (  # noqa: F401
    CURATION_ITEMS_INDEX,
    DEFAULT_GEMMA,
    DEFAULT_OPENSEARCH,
    DEFAULT_PAUSE_SENTINEL,
    DEFAULT_SAM3,
    DEFAULT_SAM3_URLS,
    DEFAULT_TRITON,
    JPEG_QUALITY,
    SECONDARY_SHAPE_GROUPS,
    STATUS_PENDING_DETECTION,
    STATUS_PENDING_VERIFICATION,
)
from src.clients.triton_pool import AsyncTritonPool  # noqa: F401
from src.services.labeling.vlm_labeler import VlmLabeler  # noqa: F401

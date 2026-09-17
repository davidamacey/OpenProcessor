"""Curation router package.

Aggregates endpoints split across many sub-modules onto a single
`router` object owned by `_common`. The submodule imports below are
side-effect imports — each registers `@router.<verb>(...)` handlers.
This list grows one entry per router as later chunks port them (see
docs/design/oss_genericization_phase2_plan.md §2.4 step 3).

Re-exports only the symbols ported tests actually patch by string
(`src.routers.curation.<symbol>`) — unlike the reference package's
full back-compat `__all__`, the public branch has no pre-existing test
history to preserve, so everything else is imported from its real
module.
"""

from __future__ import annotations

from src.clients.curation_opensearch import get_class_registry

# Side-effect imports: each module registers its endpoints on `router`.
# Chunk 1 ports only `_common` (no endpoint sub-modules yet); later
# waves add one import per router here in the same commit that ports
# it. Chunk 2 adds curation_images (two routers: `router`, `crops_router`
# — both live outside this package, mirroring the reference layout).
from src.routers.curation._common import _ensure_indexes, router
from src.routers.curation_images import crops_router as images_crops_router, router as images_router


__all__ = [
    '_ensure_indexes',
    'get_class_registry',
    'images_crops_router',
    'images_router',
    'router',
]

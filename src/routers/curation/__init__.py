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

import src.routers.curation.bakeoff
import src.routers.curation.classes
import src.routers.curation.clusters
import src.routers.curation.crops
import src.routers.curation.events
import src.routers.curation.export
import src.routers.curation.ingest
import src.routers.curation.methods
import src.routers.curation.models
import src.routers.curation.pipeline_control
import src.routers.curation.pipeline_events
import src.routers.curation.pipeline_health
import src.routers.curation.regions
import src.routers.curation.regions_fp
import src.routers.curation.review
import src.routers.curation.scores
import src.routers.curation.search
import src.routers.curation.select
import src.routers.curation.stats
import src.routers.curation.viz
import src.routers.curation.vlm  # noqa: F401 - side-effect import
from src.clients.curation_opensearch import get_class_registry

# Side-effect imports: each module registers its endpoints on `router`.
# Chunk 1 ports only `_common` (no endpoint sub-modules yet); later
# waves add one import per router here in the same commit that ports
# it. Chunk 2 adds curation_images (two routers: `router`, `crops_router`
# — both live outside this package, mirroring the reference layout).
# Chunk 4 adds `clusters` and `viz` (both register on this package's
# shared `router`); the UMAP-rebuild operator router
# (`src/routers/curation_umap.py`) lives outside this package like
# curation_images, and is registered directly in src/main.py.
# Chunk 5 adds `review`, `scores`, `select` and `methods` (all register
# on this package's shared `router`). Chunk 6 adds `bakeoff` (registers
# on this package's shared `router`; the training router,
# `src/routers/curation_train.py`, lives outside this package like
# curation_images/curation_umap and is registered directly in
# src/main.py). Chunk 7 adds `vlm` (registers on this package's shared
# `router`; re-exports `_get_vlm_labeler` because the ported tests
# patch it by string, mirroring the reference package's
# `_get_gemma_labeler` re-export). Chunk 8 adds `regions` and
# `regions_fp` (both register on this package's shared `router`);
# `regions_fp` imports helpers from `regions` directly (not through
# this package's `__init__`), so there is no import-order requirement
# between the two side-effect imports above. Chunk 9 (final content
# wave) adds the remaining 11 leaf routers: `classes`, `crops`,
# `events`, `export`, `ingest`, `models`, `search`, `stats`,
# `pipeline_control`, `pipeline_events`, `pipeline_health` — all
# register on this package's shared `router`. `classes` imports
# `list_crops` from `crops` directly (not through this package's
# `__init__`); `models` imports `_get_vlm_labeler` from `vlm` directly;
# `pipeline_events` lazily imports `stats.stats_dataset` and
# `autolabel.job` inside its generator function body (not at module
# load time) to avoid a load-order dependency. This package now
# aggregates all 21 curation routers (plan §5 Chunk 9's final count).
from src.routers.curation._common import _ensure_indexes, _raw_opensearch_dep, router
from src.routers.curation.vlm import _get_vlm_labeler
from src.routers.curation_images import crops_router as images_crops_router, router as images_router


__all__ = [
    '_ensure_indexes',
    '_get_vlm_labeler',
    '_raw_opensearch_dep',
    'get_class_registry',
    'images_crops_router',
    'images_router',
    'router',
]

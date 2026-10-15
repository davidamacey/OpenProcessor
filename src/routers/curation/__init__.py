"""Curation router package.

Aggregates endpoints split across many sub-modules onto a single
`router` object owned by `_common`. The submodule imports below are
side-effect imports — each registers `@router.<verb>(...)` handlers.
New routers add one import here in the same commit that adds them (see
docs/design/curation_design_rationale.md for the design approach).

Re-exports only the symbols tests actually patch by string
(`src.routers.curation.<symbol>`); everything else is imported from its
real module.
"""

from __future__ import annotations

import src.routers.curation._project_pause
import src.routers.curation.audit
import src.routers.curation.bakeoff


import src.routers.curation.crops_list  # isort: skip
import src.routers.curation.crops_write  # isort: skip
import src.routers.curation.crops_exclude  # isort: skip
import src.routers.curation.classes  # isort: skip
import src.routers.curation.classes_write  # isort: skip
import src.routers.curation.classes_merge  # isort: skip
import src.routers.curation.classes_crops  # isort: skip
import src.routers.curation.ingest  # isort: skip
import src.routers.curation.class_seed
import src.routers.curation.clusters
import src.routers.curation.cohorts
import src.routers.curation.config_test
import src.routers.curation.config_vocabulary
import src.routers.curation.crop_context
import src.routers.curation.datasets
import src.routers.curation.detections
import src.routers.curation.edit_undo
import src.routers.curation.events
import src.routers.curation.export
import src.routers.curation.export_single_class


import src.routers.curation.vlm  # isort: skip
import src.routers.curation.assist
import src.routers.curation.health
import src.routers.curation.ingest_bulk
import src.routers.curation.ingest_bulk_preview
import src.routers.curation.ingest_objects
import src.routers.curation.ingest_policy
import src.routers.curation.ingest_sources
import src.routers.curation.ingest_upload
import src.routers.curation.keymap
import src.routers.curation.label_undo
import src.routers.curation.methods
import src.routers.curation.models
import src.routers.curation.open_vocab
import src.routers.curation.open_vocab_test


import src.routers.curation.pipeline_start  # isort: skip
import src.routers.curation.pipeline_public  # isort: skip
import src.routers.curation.pipeline_control
import src.routers.curation.pipeline_events
import src.routers.curation.probe
import src.routers.curation.prompt_packs
import src.routers.curation.region_profiles
import src.routers.curation.region_stage
import src.routers.curation.regions
import src.routers.curation.regions_boxes_edit
import src.routers.curation.regions_edit
import src.routers.curation.regions_fp
import src.routers.curation.reprocess


import src.routers.curation.review_terms  # isort: skip
import src.routers.curation.review  # isort: skip
import src.routers.curation.holdout_freeze  # isort: skip
import src.routers.curation.review_proposals
import src.routers.curation.review_resolve
import src.routers.curation.scores
import src.routers.curation.search
import src.routers.curation.select
import src.routers.curation.settings
import src.routers.curation.stats
import src.routers.curation.viz
import src.routers.curation.vlm_activation
import src.routers.curation.vlm_policy  # noqa: F401 - side-effect import
from src.clients.curation_opensearch.registry import get_class_registry
from src.routers.curation._common import _raw_opensearch_dep, router
from src.routers.curation.images import crops_router as images_crops_router, router as images_router


# Side-effect imports: each module registers its endpoints on `router`.
# `images` (two routers: `router`, `crops_router`), the UMAP-rebuild operator
# router (`umap.py`) and the training router (`train/`) are separate router
# objects, mounted by `_mounting.py` rather than registered on `router`. `regions_fp`
# imports helpers from `regions` directly (not through this package's
# `__init__`), so there is no import-order requirement between those two
# side-effect imports. `classes_crops` imports `list_crops` from `crops_list`
# directly (not through this package's `__init__`); `pipeline_events` lazily
# imports `stats.stats_dataset` and `autolabel.job` inside its generator
# function body (not at module load time) to avoid a load-order
# dependency. `settings` registers `GET,PUT /curation/settings` on this
# same shared `router`.


__all__ = [
    '_raw_opensearch_dep',
    'get_class_registry',
    'images_crops_router',
    'images_router',
    'router',
]

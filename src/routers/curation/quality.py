"""Curation router sub-module: the dataset-level "ready to train" checklist.

``GET /quality/readiness`` answers the questions Train's preflight asks of a request, asked of the
dataset alone (:mod:`src.services.curation.dataset_readiness`).
"""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.schemas.curation.quality import DatasetReadiness
from src.services.curation.dataset_readiness import build_readiness, gather_facts
from src.services.near_duplicates.status import status as near_dup_status


@router.get('/quality/readiness', response_model=DatasetReadiness)
async def quality_readiness(
    opensearch: OpenSearchDep, registry: RegistryDep, session: DbSessionDep
) -> DatasetReadiness:
    """Is the dataset ready to train on? A `verdict` (`ready`, `warn`, `blocked`), one entry in
    `checks` per question (class minimums, the frozen test set, machine labels nobody has
    checked, the accuracy audit, near-duplicate groups, unlabelled objects on exportable photos,
    the newest export), each with its `label`, `severity`, `message`, the `count` it is about, the
    worst `classes` and a `link` to the items that need work, and `coverage` per class: validated,
    machine-set, detector-only and test items. The class and test-set rows are Train preflight's
    own checks (`POST /train/preflight`), so the two never disagree."""
    near = await near_dup_status(session, current_project_id())
    stale = [f'{lvl.level}: {reason}' for lvl in near.levels for reason in lvl.stale_reasons]
    doc = await registry.snapshot()
    return build_readiness(await gather_facts(opensearch, list(doc.classes), stale))

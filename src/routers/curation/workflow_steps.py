"""Curation router sub-module: the loop's next steps and the Workspace default view."""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep
from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.schemas.curation.workflow_steps import WorkflowSteps
from src.services.curation.class_counts import fetch_class_counts
from src.services.curation.workflow_steps import build_steps, gather_facts


@router.get('/next_steps', response_model=WorkflowSteps, tags=['Workflow'])
async def get_next_steps(session: DbSessionDep, opensearch: OpenSearchDep) -> WorkflowSteps:
    """The curation loop in order, which steps are done, the first one that is not, and the view
    the Workspace opens on (`clusters` when an item clustering serves, else `gallery`)."""
    registry = await get_class_registry().snapshot()
    counts, _aggs = await fetch_class_counts(opensearch)
    facts = await gather_facts(
        session, current_project_id(), opensearch, list(registry.classes), counts
    )
    return build_steps(facts)

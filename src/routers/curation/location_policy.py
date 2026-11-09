"""Curation router sub-module: the project's location policy.

``GET/PUT /curation/projects/{project}/location_policy``. The policy decides what ingest stores
of a photo's GPS fix and what any route serves: ``store`` (default) as read, ``store_coarse``
rounded to about 1 km, ``drop`` nothing. A switch to a policy that keeps less starts the purge of
the coordinates already stored and answers its job id.
"""

from src.core.db_session import DbSessionDep
from src.routers.curation._api_error import api_error
from src.routers.curation._common import router
from src.schemas.curation.location_policy import LocationPolicyResponse, LocationPolicyUpdate
from src.services.curation import location_policy as service
from src.services.curation.location_policy import RevisionConflictError


@router.get('/location_policy', response_model=LocationPolicyResponse, tags=['Location policy'])
async def get_location_policy_route() -> LocationPolicyResponse:
    """The project's location policy and the settings revision to send back on a change."""
    return await service.get_policy()


@router.put('/location_policy', response_model=LocationPolicyResponse, tags=['Location policy'])
async def put_location_policy_route(
    body: LocationPolicyUpdate, session: DbSessionDep
) -> LocationPolicyResponse:
    """Change the policy (409 `revision_conflict` on a stale `expected_revision`)."""
    try:
        return await service.change_policy(
            session, body.location_policy, expected_revision=body.expected_revision
        )
    except RevisionConflictError as exc:
        raise api_error(409, 'revision_conflict', f'settings changed: {exc}') from exc

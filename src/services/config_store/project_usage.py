"""Read something from every project's own config.

Activation is per project while promoted models and VLM endpoints are deployment-wide, so "who
uses X" is answered from every project's LIVE activations in the database, never a cached
snapshot: a delete or an unshare must not race a stale view of another project's activation. A
read that fails raises, so a caller that refuses on the result fails closed.
"""

from __future__ import annotations

from src.db.repos import config as repo
from src.services.config_store.documents import tx


# The "profile" a project's own ingest detector is reported under.
INGEST_DETECTOR_USER = 'ingest detector'


async def active_detector_users(model_name: str) -> list[tuple[str, str]]:
    """``(project slug, active profile name)`` for every project whose ACTIVE
    detection profile (the revision it pinned) names ``model_name`` as its
    ``detector_model``."""
    async with tx('who uses a detector') as session:
        active = await repo.list_active_bodies(session, 'detection_profile')
    region_users = [
        (a.slug, a.name) for a in active if str(a.body.get('detector_model') or '') == model_name
    ]
    ingest_users = [
        (slug, INGEST_DETECTOR_USER) for slug in await ingest_detector_users(model_name)
    ]
    return sorted([*region_users, *ingest_users])


async def ingest_models_by_project() -> dict[str, frozenset[str]]:
    """``slug -> models`` its ACTIVE ingest profile names as primary or secondary
    ingest detector (empty when none is active)."""
    from src.schemas.curation.ingest_profile import IngestProfileBody

    async with tx('ingest models in use') as session:
        active = await repo.list_active_bodies(session, 'ingest_profile')
        projects = await repo.list_axis_state(session, 'ingest_profile')
    out: dict[str, frozenset[str]] = {slug: frozenset() for slug in projects}
    for a in active:
        body = IngestProfileBody.model_validate(a.body)
        out[a.slug] = frozenset(
            spec.model for spec in (body.primary, body.secondary) if spec is not None
        )
    return out


async def ingest_detector_users(model_name: str) -> list[str]:
    """Slugs of the projects whose ACTIVE ingest profile names ``model_name`` as
    their primary or secondary ingest detector."""
    return [
        slug for slug, models in (await ingest_models_by_project()).items() if model_name in models
    ]


async def model_dependents(owner_slug: str, models: list[str]) -> list[dict[str, str]]:
    """``{project, profile, model}`` for every project OTHER than
    ``owner_slug`` whose active detection profile names one of ``models``."""
    return [
        {'project': slug, 'profile': profile, 'model': model}
        for model in models
        for slug, profile in await active_detector_users(model)
        if slug != owner_slug
    ]


__all__ = [
    'INGEST_DETECTOR_USER',
    'active_detector_users',
    'ingest_detector_users',
    'ingest_models_by_project',
    'model_dependents',
]

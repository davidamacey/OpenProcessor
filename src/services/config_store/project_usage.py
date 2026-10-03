"""Read something from every project's own config index.

Activation is per project while promoted models and VLM endpoints are
deployment-wide, so "who uses X" is answered from each project's LIVE
config docs (a read-only bind per project, straight from OpenSearch --
never a cached snapshot: a delete or an unshare must not race a stale view
of another project's activation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.project_context import bind_project
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# The "profile" a project's own ingest detector is reported under.
INGEST_DETECTOR_USER = 'ingest detector'


async def read_each_project[T](read: Callable[[str], Awaitable[T]]) -> dict[str, T]:
    """``slug -> read(configs_index)`` for every project that exists and is
    not being deleted (``ProjectRegistry.existing_projects``), each call inside a
    read-only bind of that project. A read that raises propagates: callers
    use this to REFUSE a change, so an unreadable project must block it."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    found: dict[str, T] = {}
    for record in sorted(registry.existing_projects(), key=lambda r: r.slug):
        with bind_project(record, read_only=True):
            found[record.slug] = await read(get_curation_config().configs_index)
    return found


async def active_detector_users(client: Any, model_name: str) -> list[tuple[str, str]]:
    """``(project slug, active profile name)`` for every project whose ACTIVE
    detection profile (the revision it pinned) names ``model_name`` as its
    ``detector_model``."""
    from opensearchpy.exceptions import NotFoundError

    from src.services.config_store.index import config_doc_id, get_activation

    async def active_detector(index: str) -> tuple[str, str] | None:
        activation = await get_activation(client, index, 'detection_profile')
        name = (activation or {}).get('name')
        revision = (activation or {}).get('revision')
        if not name or revision is None:
            return None
        try:
            doc = await client.get(
                index=index, id=config_doc_id('region_profile', name, int(revision))
            )
        except NotFoundError:
            return None
        return name, str((doc['_source'].get('body') or {}).get('detector_model') or '')

    region_users = [
        (slug, found[0])
        for slug, found in (await read_each_project(active_detector)).items()
        if found is not None and found[1] == model_name
    ]
    ingest_users = [
        (slug, INGEST_DETECTOR_USER) for slug in await ingest_detector_users(client, model_name)
    ]
    return sorted([*region_users, *ingest_users])


async def ingest_detector_users(client: Any, model_name: str) -> list[str]:
    """Slugs of the projects whose ingest policy names ``model_name`` as their
    own ingest detector."""
    from src.services.curation.ingest_policy_store import get_ingest_policy

    async def own_detector(_index: str) -> str | None:
        override = (await get_ingest_policy(client)).detector
        return override.model if override is not None else None

    return [
        slug
        for slug, model in (await read_each_project(own_detector)).items()
        if model == model_name
    ]


async def model_dependents(client: Any, owner_slug: str, models: list[str]) -> list[dict[str, str]]:
    """``{project, profile, model}`` for every project OTHER than
    ``owner_slug`` whose active detection profile names one of ``models``.
    A project that cannot be read raises, so a caller that refuses on the
    result fails closed."""
    return [
        {'project': slug, 'profile': profile, 'model': model}
        for model in models
        for slug, profile in await active_detector_users(client, model)
        if slug != owner_slug
    ]


__all__ = [
    'INGEST_DETECTOR_USER',
    'active_detector_users',
    'ingest_detector_users',
    'model_dependents',
    'read_each_project',
]

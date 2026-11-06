"""Preview-only combine warnings that read the sources' stored state:
``embedding_model_mismatch`` and ``region_profiles_differ``. Both are
informational; neither blocks a combine."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from src.config.curation_roles import IndexRole
from src.config.project_context import bind_project
from src.domain.embedding_spaces import Target
from src.domain.embedding_state import embedded_clause
from src.services.projects.combine.models import CombineIssue
from src.services.projects.combine.spaces import primary_space


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.projects.combine.models import CombineRequest

_HAS_VECTOR = embedded_clause()


async def _items_with_vectors(client: Any, record: ProjectRecord) -> int:
    index = record.resources.indexes[IndexRole.ITEMS]
    with bind_project(record, read_only=True):
        return int((await client.count(index=index, body={'query': _HAS_VECTOR})).get('count') or 0)


async def embedding_model_mismatch(
    client: Any, records: list[ProjectRecord], target_keys: frozenset[str]
) -> list[CombineIssue]:
    """One warning per source whose primary item space is not a space the target will have.

    The comparison is by ``space_key`` (model and version), not by dimension: two models can
    share a dimension and still produce vectors that mean different things. The executor
    (``transform_item``) drops those vectors and defers the items for re-embedding; this reports
    the cost first."""
    out: list[CombineIssue] = []
    for record in records:
        held = await primary_space(record, Target.ITEM)
        if held is None or held.space_key in target_keys:
            continue
        count = await _items_with_vectors(client, record)
        if not count:
            continue
        out.append(
            CombineIssue(
                code='embedding_model_mismatch',
                severity='warning',
                project=record.slug,
                message=f"'{record.slug}' has up to {count} items embedded in the space "
                f"'{held.space_key}'; the target will not have it ({sorted(target_keys)}). "
                'Those vectors are not copied and the items are deferred for re-embedding',
                detail={
                    'project': record.slug,
                    'items': count,
                    'source_space': held.space_key,
                    'target_spaces': sorted(target_keys),
                },
            )
        )
    return out


def _profile_key(name: str, body: Any) -> dict[str, str | None]:
    canonical = json.dumps(body, sort_keys=True, default=str)
    return {'name': name, 'hash': hashlib.sha256(canonical.encode()).hexdigest()[:16]}


async def region_profiles_differ(
    request: CombineRequest, records: list[ProjectRecord]
) -> list[CombineIssue]:
    """A warning when the sources' active region profiles differ by name or
    content. The target gets the ``settings_from`` source's activations, or
    none when that is unset."""
    from src.services.projects.clone_activation import read_activated_config

    profiles: list[dict[str, Any]] = []
    for record in records:
        active = await read_activated_config(record, 'detection_profile')
        key = _profile_key(active.name, active.body) if active else {'name': None, 'hash': None}
        profiles.append({'project': record.slug, **key})
    if len({(p['name'], p['hash']) for p in profiles}) < 2:
        return []
    target = next((p for p in profiles if p['project'] == request.settings_from), None)
    return [
        CombineIssue(
            code='region_profiles_differ',
            severity='warning',
            message='the sources use different region profiles; boxes made under one may not '
            'satisfy another. The target takes the profile of settings_from, or none when '
            'it is unset',
            detail={
                'profiles': profiles,
                'target_profile': {'name': target['name'], 'hash': target['hash']}
                if target and target['name']
                else None,
            },
        )
    ]


__all__ = ['embedding_model_mismatch', 'region_profiles_differ']

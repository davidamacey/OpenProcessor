"""Preview-only combine warnings that read the sources' stored state:
``embedding_model_mismatch`` and ``region_profiles_differ``. Both are
informational; neither blocks a combine."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from src.config.curation import ITEM_EMBEDDING_FIELD, IndexRole
from src.config.project_context import bind_project
from src.services.projects.combine.models import CombineIssue


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.projects.combine.models import CombineRequest

_HAS_VECTOR = {'exists': {'field': ITEM_EMBEDDING_FIELD}}


async def _stored_vector_dim(client: Any, record: ProjectRecord) -> tuple[int, int | None]:
    """``(items with a vector, the vector length of one of them)``. The index
    mapping fixes one dimension per index, so one sample speaks for all."""
    index = record.resources.indexes[IndexRole.ITEMS]
    with bind_project(record, read_only=True):
        count = (await client.count(index=index, body={'query': _HAS_VECTOR})).get('count') or 0
        if not count:
            return 0, None
        resp = await client.search(
            index=index,
            body={'size': 1, 'query': _HAS_VECTOR, '_source': [ITEM_EMBEDDING_FIELD]},
        )
    hits = (resp.get('hits') or {}).get('hits') or []
    vector = (hits[0].get('_source') or {}).get(ITEM_EMBEDDING_FIELD) if hits else None
    return int(count), len(vector) if isinstance(vector, list) else None


async def embedding_model_mismatch(
    client: Any, records: list[ProjectRecord], target_dim: int
) -> list[CombineIssue]:
    """One warning per source whose stored vectors have a dimension the target
    encoder does not produce. The executor (``transform_item``) drops those
    vectors and defers the items for re-embedding; this reports the cost
    first. Items carry no encoder id, so the dimension is the whole check."""
    out: list[CombineIssue] = []
    for record in records:
        count, dim = await _stored_vector_dim(client, record)
        if dim is None or dim == target_dim:
            continue
        out.append(
            CombineIssue(
                code='embedding_model_mismatch',
                severity='warning',
                project=record.slug,
                message=f"'{record.slug}' has up to {count} items embedded at {dim} "
                f'dimensions; the target encoder uses {target_dim}. Those vectors are not '
                'copied and the items are deferred for re-embedding',
                detail={
                    'project': record.slug,
                    'items': count,
                    'source_dim': dim,
                    'target_dim': target_dim,
                },
            )
        )
    return out


def _profile_key(name: str, body: Any) -> dict[str, str | None]:
    canonical = json.dumps(body, sort_keys=True, default=str)
    return {'name': name, 'hash': hashlib.sha256(canonical.encode()).hexdigest()[:16]}


async def region_profiles_differ(
    client: Any, request: CombineRequest, records: list[ProjectRecord]
) -> list[CombineIssue]:
    """A warning when the sources' active region profiles differ by name or
    content. The target gets the ``settings_from`` source's activations, or
    none when that is unset."""
    from src.services.projects.clone import read_activated_config

    profiles: list[dict[str, Any]] = []
    for record in records:
        active = await read_activated_config(client, record, 'detection_profile', 'region_profile')
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

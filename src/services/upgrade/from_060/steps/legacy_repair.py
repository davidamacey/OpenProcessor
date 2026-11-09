"""Step ``legacy_repair``: what 0.6.0 repaired at every API start, done once, per project.

Moved here from ``src/services/projects/legacy_repair.py`` (deleted). Three repairs:

* the active region profile's class is in the class registry (a project created by an earlier
  release may lack it);
* a project with no VLM activation calls no VLM: 0.7.0 has no "inherit the environment endpoint"
  state, so a project that was on the 0.6.0 ``env`` built-in is listed in the report to activate
  an endpoint;
* items that have a vector but no ``embedding_state`` record ``embedded`` (an ``update_by_query``
  on the items index, throttled for a small-heap node and retried while OpenSearch answers 429).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.domain.embedding_spaces import Target, legacy_names
from src.domain.embedding_state import legacy_embedded_backfill_body, legacy_embedded_clause
from src.services.upgrade.from_060 import control_rows
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from src.services.upgrade.from_060.model import ProjectData
    from src.services.upgrade.from_060.steps import StepContext

ORDER = 30
REGION_CLASS_GROUP = 'region'
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 2.0
_TOO_MANY_REQUESTS = 429
_BACKFILL_REQUESTS_PER_SECOND = 500  # gentle on a small-heap node


def region_class_name(pd: ProjectData) -> str:
    """``region_class_name`` of the project's active region profile revision, or ``''``."""
    current = [e for e in pd.activations if e.axis == 'detection_profile']
    if not current:
        return ''
    active = current[-1]
    for doc in pd.configs:
        if doc.kind == 'region_profile' and doc.name == active.to_name:
            for rev in doc.revisions:
                if rev.revision == active.to_revision:
                    return str((rev.body or {}).get('region_class_name') or '').strip()
    return ''


def adds_region_class(pd: ProjectData) -> bool:
    """Whether the repair adds a class: the active profile names a region class that no active
    registry class carries (by the one name-equality rule)."""
    name = normalize_class_name(region_class_name(pd))
    if not name:
        return False
    active = {
        normalize_class_name(c.name)
        for c in (pd.registry.classes if pd.registry else ())
        if not c.deprecated
    }
    return name not in active


async def backfill_embedded_state(ctx: StepContext, index: str) -> int:
    """Record ``embedded`` on every legacy item that has its vector. Idempotent: after one pass
    nothing matches, and a cheap count says so without starting a task."""
    names = legacy_names(Target.ITEM)
    state_field = names.state_field or ''
    if await ctx.writer.count(index, legacy_embedded_clause(names.vector_field, state_field)) == 0:
        return 0
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await ctx.writer.update_by_query(
                index,
                legacy_embedded_backfill_body(names.vector_field, state_field),
                requests_per_second=_BACKFILL_REQUESTS_PER_SECOND,
            )
        except Exception as exc:
            if (
                getattr(exc, 'status_code', None) != _TOO_MANY_REQUESTS
                or attempt == _RETRY_ATTEMPTS - 1
            ):
                raise
            await asyncio.sleep(_RETRY_BASE_DELAY_S * 2**attempt)
    return 0  # unreachable: the last attempt returns or raises


async def run(ctx: StepContext) -> None:
    summary: dict[str, Any] = {}
    for pd in ctx.parsed.projects:
        slug = pd.project.slug
        entry: dict[str, Any] = {}
        name = region_class_name(pd)
        if name:
            entry['region_class_id'] = await control_rows.ensure_class(
                ctx.session,
                ctx.project_ids[slug],
                name,
                group=REGION_CLASS_GROUP,
                notes='seeded from the active region profile',
            )
        if not any(e.axis == 'vlm' for e in pd.activations):
            entry['vlm_active'] = False
        entry['embedded_state_backfilled'] = await backfill_embedded_state(
            ctx, pd.project.indexes['items']
        )
        ctx.report.add('legacy_repair_embedded_state', entry['embedded_state_backfilled'])
        summary[slug] = entry
    ctx.notes['legacy_repair'] = summary

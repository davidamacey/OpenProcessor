"""Step ``near_duplicates``: what a 0.6.0 install carries of the old near-duplicate scoring.

* The per-item ``dup_*`` values and the core index's ``duplicate_*`` values are removed from every
  document with one throttled ``update_by_query`` per index (a mapping cannot drop a field
  without a reindex, so the old mapping stays; the values do not). They were grouped at a
  cosine cut the new run tables do not know, so keeping them would show groups no run owns.
* Each loaded project gets revision 1 of its ``near_dup_profile`` from the code defaults, the
  same document a new project is created with.
* No run exists for a loaded project, so both levels report ``no_run`` and ask for a full
  build; nothing is written for that. The report lists the projects that should also start the
  ``source_backfill`` job: the exact-content key (``image_contents``) covers only images ingested
  after the upgrade until that job has read the files the upgrade carried over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.schemas.curation.near_duplicates import NearDupProfileBody
from src.services.near_duplicates.profiles import (
    AXIS,
    DEFAULT_NAME,
    KIND,
    NoActiveProfileError,
    active_profile,
)
from src.services.upgrade.from_060 import control_rows
from src.services.upgrade.from_060.model import ActivationSrc, ConfigDocSrc, ConfigRevisionSrc


if TYPE_CHECKING:
    from src.services.upgrade.from_060.steps import StepContext

ORDER = 40
ITEM_FIELDS = (
    'dup_group_id',
    'dup_group_size',
    'dup_is_representative',
    'dup_threshold',
    'dup_method',
    'dup_scored_at',
)
CORE_FIELDS = ('duplicate_group_id', 'is_duplicate_primary', 'duplicate_score')
CORE_INDEX = 'visual_search_global'
REQUESTS_PER_SECOND = 500  # gentle on a small-heap node


def holds_any(fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        'bool': {'should': [{'exists': {'field': f}} for f in fields], 'minimum_should_match': 1}
    }


def removal(fields: tuple[str, ...]) -> dict[str, Any]:
    """The ``update_by_query`` body that removes ``fields`` from the documents that hold any."""
    return {
        'query': holds_any(fields),
        'script': {
            'lang': 'painless',
            'source': ''.join(f"ctx._source.remove('{f}');" for f in fields),
        },
    }


async def clear_index(ctx: StepContext, index: str, fields: tuple[str, ...]) -> int:
    """Remove ``fields`` from ``index``; idempotent, a count of nothing starts no task."""
    if await ctx.writer.count(index, holds_any(fields)) == 0:
        return 0
    return await ctx.writer.update_by_query(
        index, removal(fields), requests_per_second=REQUESTS_PER_SECOND
    )


async def load_profile(ctx: StepContext, slug: str) -> None:
    """Revision 1 of the project's ``default`` profile, active, with the ids every load gives it."""
    project_id = ctx.project_ids[slug]
    doc = ConfigDocSrc(
        project_slug=slug,
        kind=KIND,
        name=DEFAULT_NAME,
        description='The code defaults: 0.98 cosine on the frame and on the crop.',
        revisions=(
            ConfigRevisionSrc(1, NearDupProfileBody().model_dump(mode='json'), ctx.now, None),
        ),
        created_at=ctx.now,
        updated_at=ctx.now,
    )
    config_id = await control_rows.load_config(ctx.session, project_id, doc)
    ctx.config_ids[(slug, KIND, DEFAULT_NAME)] = config_id
    await control_rows.load_activations(
        ctx.session,
        project_id,
        [ActivationSrc(AXIS, None, None, DEFAULT_NAME, 1, KIND, ctx.now)],
        {(KIND, DEFAULT_NAME): config_id},
    )
    ctx.report.add('near_dup_profiles')


async def run(ctx: StepContext) -> None:
    cleared: dict[str, int] = {}
    for pd in ctx.parsed.projects:
        slug = pd.project.slug
        cleared[pd.project.indexes['items']] = await clear_index(
            ctx, pd.project.indexes['items'], ITEM_FIELDS
        )
        await load_profile(ctx, slug)
    if await ctx.reader.index_exists(CORE_INDEX):
        cleared[CORE_INDEX] = await clear_index(ctx, CORE_INDEX, CORE_FIELDS)
    ctx.report.add('near_dup_values_cleared', sum(cleared.values()))
    ctx.notes['near_duplicates'] = {
        'values_cleared_by_index': cleared,
        'levels_needing_a_full_build': ['image', 'item'],
        'start_source_backfill': sorted(pd.project.slug for pd in ctx.parsed.projects),
    }


async def verify(ctx: StepContext) -> list[str]:
    """No document keeps an old value, and every loaded project has its profile."""
    problems: list[str] = []
    for pd in ctx.parsed.projects:
        slug = pd.project.slug
        index = pd.project.indexes['items']
        if left := await ctx.writer.count(index, holds_any(ITEM_FIELDS)):
            problems.append(f'{slug}: {left} items still hold a dup_* value')
        try:
            await active_profile(ctx.session, ctx.project_ids[slug])
        except NoActiveProfileError:
            problems.append(f'{slug}: no active near_dup_profile')
    if await ctx.reader.index_exists(CORE_INDEX) and (
        left := await ctx.writer.count(CORE_INDEX, holds_any(CORE_FIELDS))
    ):
        problems.append(f'{CORE_INDEX}: {left} documents still hold a duplicate_* value')
    return problems

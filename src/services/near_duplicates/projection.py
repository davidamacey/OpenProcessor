"""Apply the current run to OpenSearch: the ``near_dup_*`` fields of images and items.

Only fields the run owns are written (machine fields; a class or a box is never touched, which
is what the lock rule asks). A full run sets every grouped member and then removes the group
fields from every document that carries another run's id; an incremental run writes the members
whose group, role or size changed and clears the members that left every group, so the cost
follows the change, not the project. A run that is no longer current is skipped: a newer run
projects instead, and an older projection can never overwrite a newer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.db.repos import near_dup as repo
from src.domain import near_dup as rules


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from src.services.near_duplicates.runs import Env

FIELDS = (
    'near_dup_group_id',
    'near_dup_leak_group_id',
    'near_dup_is_representative',
    'near_dup_group_size',
    'near_dup_run_id',
)
# What a clear removes: the project fields and the core-only cosine.
_CLEARED = (*FIELDS, 'near_dup_cosine_to_rep')
BULK_CHUNK = 500
_CLEAR = {
    'lang': 'painless',
    'source': ''.join(f"ctx._source.remove('{f}');" for f in _CLEARED),
}


@dataclass(frozen=True)
class ProjectionResult:
    skipped: bool
    written: int
    cleared: int


def _fields(
    run_id: uuid.UUID,
    dedup: repo.MemberState | None,
    leak: repo.MemberState | None,
    *,
    core: bool,
) -> dict[str, Any]:
    """The fields of one member. The core global index also carries the cosine to the
    representative (its ``/query/duplicates`` wire shape shows it)."""
    out: dict[str, Any] = {'near_dup_run_id': str(run_id)}
    if dedup is not None:
        out.update(
            near_dup_group_id=str(dedup.lineage_id),
            near_dup_is_representative=dedup.is_representative,
            near_dup_group_size=dedup.group_size,
        )
        if core:
            out['near_dup_cosine_to_rep'] = dedup.cosine_to_rep
    if leak is not None:
        out['near_dup_leak_group_id'] = str(leak.lineage_id)
    return out


def _by_member(
    states: Mapping[tuple[str, str], repo.MemberState],
) -> dict[str, tuple[repo.MemberState | None, repo.MemberState | None]]:
    members = {m for _, m in states}
    return {m: (states.get(('dedup', m)), states.get(('leak', m))) for m in members}


def _changed(
    now: dict[str, tuple[repo.MemberState | None, repo.MemberState | None]],
    before: dict[str, tuple[repo.MemberState | None, repo.MemberState | None]] | None,
) -> tuple[list[str], list[str]]:
    """Members to write and members to clear, against the parent run (``None``: everything)."""
    if before is None:
        return sorted(now), []
    write = sorted(m for m, state in now.items() if before.get(m) != state)
    clear = sorted(set(before) - set(now))
    return write, clear


async def _bulk(env: Env, ops: list[dict[str, Any]]) -> None:
    for start in range(0, len(ops), BULK_CHUNK * 2):
        response = await env.client.bulk(body=ops[start : start + BULK_CHUNK * 2], refresh=False)
        for item in response.get('items') or []:
            result = item.get('update') or {}
            status = int(result.get('status', 0))
            if status not in (200, 404):  # a document deleted since is simply gone
                msg = f'near-duplicate projection failed: {result.get("error") or status}'
                raise RuntimeError(msg)


async def apply_run(env: Env, run_id: uuid.UUID, task_id: uuid.UUID | None) -> ProjectionResult:
    """Project ``run_id`` if it is still the level's current run."""
    async with env.scope() as session:
        run = await repo.get_run(session, run_id)
        current = await repo.current_run(session, env.project_id, env.plan.level)
        if run is None or current is None or current.id != run_id:
            if task_id is not None:
                await repo.mark_projection_done(session, task_id)
            return ProjectionResult(skipped=True, written=0, cleared=0)
        now = _by_member(await repo.member_states(session, run_id))
        parent = await repo.parent_member_states(session, run_id, run.mode)
    before = None if parent is None else _by_member(parent)
    write, clear = _changed(now, before)
    index = env.plan.index
    ops: list[dict[str, Any]] = []
    for member in write:
        ops.append({'update': {'_index': index, '_id': member}})
        ops.append({'doc': _fields(run_id, *now[member], core=env.plan.level == rules.CORE)})
    await _bulk(env, ops)
    clear_ops: list[dict[str, Any]] = []
    for member in clear:
        clear_ops.append({'update': {'_index': index, '_id': member}})
        clear_ops.append({'script': _CLEAR})
    await _bulk(env, clear_ops)
    if before is None:
        await env.client.update_by_query(
            index=index,
            body={
                'script': _CLEAR,
                'query': {
                    'bool': {
                        'filter': [{'exists': {'field': 'near_dup_run_id'}}],
                        'must_not': [{'term': {'near_dup_run_id': str(run_id)}}],
                    }
                },
            },
            conflicts='proceed',
            refresh=False,
        )
    if task_id is not None:
        async with env.scope() as session:
            await repo.mark_projection_done(session, task_id)
    return ProjectionResult(skipped=False, written=len(write), cleared=len(clear))

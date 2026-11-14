"""A batch's content claims and pending rows, against the real tables.

The offline tier proves the verdicts (``test_claim_in_session_table.py``); this one proves what
only PostgreSQL can: the advisory locks of two concurrent batches that share contents never wait on
each other in a cycle (they are taken in one sorted order whatever order the batch lists its
photos in), each content is stored once, and only stored photos get pending rows.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import func, select

from src.db.models.near_dup import ImageContent, NearDupPending
from src.db.session import async_session_factory, project_tx
from src.services.near_duplicates import content_key, ingest_claim
from src.services.near_duplicates.content_key import ContentClaim, content_image_id

from .conftest import make_project


if TYPE_CHECKING:
    import uuid
    from pathlib import Path

pytestmark = [pytest.mark.db]

CONTENTS = 40
ROUNDS = 5


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}

    async def project_id() -> Any:
        return state['project']

    monkeypatch.setattr(content_key, 'bound_project_id', project_id)
    monkeypatch.setattr(ingest_claim, 'bound_project_id', project_id)
    return state


def _photo(directory: Path, tag: str, n: int) -> ingest_claim.PhotoWrite:
    """A photo whose file is on disk: a second claim of the content confirms it against that file."""
    data = f'{tag}-{n}'.encode().ljust(64, b'\0')
    path = directory / f'{tag}-{n}.jpg'
    path.write_bytes(data)
    imohash = f'{tag}-{n:04d}'.ljust(32, '0')
    image_id = content_image_id(imohash)
    return ingest_claim.PhotoWrite(
        item_ids=[f'{image_id}-item-{k}' for k in range(2)],
        claim=ContentClaim(imohash, image_id, str(path), data),
    )


async def _count(project: uuid.UUID, model: Any, *where: Any) -> int:
    async with async_session_factory()() as session, project_tx(session, project):
        return (
            await session.execute(select(func.count()).select_from(model).where(*where))
        ).scalar_one()


@pytest.mark.asyncio
async def test_a_batch_marks_pending_rows_for_the_stored_photos_only(
    pg_registry: Any, bound: dict[str, Any], tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    first = [_photo(tmp_path, 'a', n) for n in range(3)]
    assert [o.holder for o in await ingest_claim.claim_and_mark_bound(first)] == [None] * 3

    # the same three contents again plus two new ones, and an image that already has its document
    again = [*first, _photo(tmp_path, 'b', 0), _photo(tmp_path, 'b', 1)]
    existing = ingest_claim.PhotoWrite(item_ids=['kept-item'], claim=None)
    outcomes = await ingest_claim.claim_and_mark_bound([*again, existing])

    assert [o.holder for o in outcomes[:3]] == [p.claim.image_id for p in first if p.claim]
    assert [o.holder for o in outcomes[3:]] == [None, None, None]
    assert await _count(project, ImageContent) == 5
    assert await _count(project, NearDupPending, NearDupPending.level == 'image') == 5
    # 5 stored photos with 2 items each, plus the existing image's item
    assert await _count(project, NearDupPending, NearDupPending.level == 'item') == 11


@pytest.mark.asyncio
async def test_two_batches_of_the_same_contents_in_opposite_order_do_not_deadlock(
    pg_registry: Any, bound: dict[str, Any], tmp_path: Path
) -> None:
    for round_ in range(ROUNDS):
        bound['project'] = project = await make_project()
        photos = [_photo(tmp_path, f'r{round_}', n) for n in range(CONTENTS)]
        forward, backward = photos, list(reversed(photos))
        got_forward, got_backward = await asyncio.wait_for(
            asyncio.gather(
                ingest_claim.claim_and_mark_bound(forward),
                ingest_claim.claim_and_mark_bound(backward),
            ),
            timeout=60,
        )
        won = {
            p.claim.image_id
            for p, o in zip(forward, got_forward, strict=True)
            if p.claim and o.holder is None
        }
        lost = {
            p.claim.image_id
            for p, o in zip(backward, got_backward, strict=True)
            if p.claim and o.holder is None
        }
        assert won.isdisjoint(lost)  # each content was stored by exactly one of the two batches
        assert len(won) + len(lost) == CONTENTS
        assert await _count(project, ImageContent) == CONTENTS
        assert await _count(project, NearDupPending, NearDupPending.level == 'image') == CONTENTS


@pytest.mark.asyncio
async def test_a_batch_and_a_claim_of_one_of_its_contents_store_it_once(
    pg_registry: Any, bound: dict[str, Any], tmp_path: Path
) -> None:
    bound['project'] = await make_project()
    photos = [_photo(tmp_path, 'm', n) for n in range(10)]
    single = photos[3].claim
    assert single is not None
    stored_alone, batch = await asyncio.wait_for(
        asyncio.gather(
            content_key.claim_bound(
                single.imohash, single.image_id, single.image_path, single.data
            ),
            ingest_claim.claim_and_mark_bound(photos),
        ),
        timeout=60,
    )
    stored_in_batch = [o.holder is None for o in batch]
    assert (stored_alone is None) != stored_in_batch[3]  # exactly one of the two stored content 3
    assert stored_in_batch[:3] + stored_in_batch[4:] == [True] * 9

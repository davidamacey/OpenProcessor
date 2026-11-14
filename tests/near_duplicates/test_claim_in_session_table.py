"""One batch claim answers each image as a claim of its own does.

The table is an in-memory stand-in for ``image_contents`` behind the repository functions, so the
decision code (:func:`~src.services.near_duplicates.content_key.claim_in_session`) runs for real:
every case below is decided once alone, on a fresh table, and once in a batch with the others; the
answers must be equal. The real table, the locks and two concurrent batches are the database tier
(``tests/db/test_content_claim_batches.py``).
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from src.db.repos import image_contents as repo
from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import ClaimOutcome, ContentClaim, content_image_id


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class Table:
    """The rows of one project and the repository functions over them."""

    def __init__(self) -> None:
        self.rows: list[repo.ContentRow] = []
        self.imohashes: dict[int, str] = {}
        self.lock_calls: list[list[str]] = []

    def add(self, imohash: str, image_id: str, path: str, size: int, sha: str | None) -> None:
        row = repo.ContentRow(len(self.rows) + 1, image_id, path, size, sha)
        self.rows.append(row)
        self.imohashes[row.id] = imohash

    def _taken(self, new: repo.NewContent) -> bool:
        return any(
            r.image_id == new.image_id
            or (new.sha256 is not None and r.content_sha256 == new.sha256)
            for r in self.rows
        )

    async def lock_imohashes(self, _s: Any, _p: Any, imohashes: Any) -> None:
        self.lock_calls.append(list(imohashes))

    async def rows_for_imohashes(self, _s: Any, _p: Any, imohashes: Any) -> dict[str, list[Any]]:
        wanted = set(imohashes)
        found: dict[str, list[Any]] = {h: [] for h in wanted}
        for row in self.rows:
            if self.imohashes[row.id] in wanted:
                found[self.imohashes[row.id]].append(row)
        return found

    async def insert_contents(self, _s: Any, _p: Any, rows: Any) -> set[str]:
        stored: set[str] = set()
        for new in rows:
            if not self._taken(new):
                self.add(new.imohash, new.image_id, new.image_path, new.size_bytes, new.sha256)
                stored.add(new.image_id)
        return stored

    async def insert_content(self, _s: Any, _p: Any, **kw: Any) -> str | None:
        new = repo.NewContent(
            kw['imohash'], kw['sha256'], kw['image_id'], kw['image_path'], kw['size_bytes']
        )
        if not self._taken(new):
            self.add(new.imohash, new.image_id, new.image_path, new.size_bytes, new.sha256)
            return None
        held = next(self.imohashes[r.id] for r in self.rows if r.image_id == new.image_id)
        if held != new.imohash:
            raise repo.ContentKeyConflictError(f'image {new.image_id} already holds other content')
        return new.image_id

    async def set_sha256(self, _s: Any, row_id: int, sha: str) -> None:
        for row in self.rows:
            if row.id == row_id and row.content_sha256 is None:
                row.content_sha256 = sha


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install(monkeypatch: pytest.MonkeyPatch, table: Table) -> None:
    for name in (
        'lock_imohashes',
        'rows_for_imohashes',
        'insert_contents',
        'insert_content',
        'set_sha256',
    ):
        monkeypatch.setattr(repo, name, getattr(table, name))


def _file(directory: Path, name: str, data: bytes) -> str:
    path = directory / name
    path.write_bytes(data)
    return str(path)


def _cases(directory: Path) -> tuple[Callable[[], Table], list[ContentClaim], list[str]]:
    """``(a fresh table, the claims, what each should answer)``: the stored state and the claims
    are built so every imohash occurs once among the claims."""
    stored = b'stored-file' + b'\0' * 64
    other_bytes = b'other-file' + b'\0' * 64  # an imohash collision with `stored`
    new = b'new-file' + b'\1' * 64
    held_unhashed = b'held-unhashed' + b'\2' * 64
    taken_path = _file(directory, 'stored.bin', stored)
    held_path = _file(directory, 'held.bin', held_unhashed)
    retried = b'retried' + b'\3' * 64
    retry_path = _file(directory, 'retried.bin', retried)

    def table() -> Table:
        t = Table()
        t.add('H-stored', 'img-stored', taken_path, len(stored), _sha(stored))
        t.add('H-held', 'img-held', held_path, len(held_unhashed), None)  # SHA-256 not computed yet
        t.add('H-other-key', 'img-bound', 'p', 1, None)
        t.add('H-elsewhere', content_image_id('H-conflict'), 'p', 1, None)
        t.add('H-retry', content_image_id('H-retry'), retry_path, len(retried), None)
        return t

    claims = [
        ContentClaim('H-new', content_image_id('H-new'), '/p/new.jpg', new),
        ContentClaim('H-stored', 'img-x', '/p/dup.jpg', stored),  # the same bytes as a stored image
        ContentClaim('H-held', 'img-y', '/p/dup2.jpg', held_unhashed),  # holder has no SHA-256 yet
        ContentClaim('H-stored2', content_image_id('H-stored2'), '/p/c.jpg', new + b'x'),
        ContentClaim('H-stale', content_image_id('H-stale'), '/p/stale.jpg', other_bytes),
        ContentClaim('H-bound', 'img-bound', '/p/bound.jpg', b'z'),  # not the id of these bytes
        ContentClaim('H-conflict', content_image_id('H-conflict'), '/p/c2.jpg', b'y' * 64),
        ContentClaim('H-retry', content_image_id('H-retry'), '/p/retry.jpg', retried),
    ]
    retry_id = content_image_id('H-retry')
    expected = ['stored', 'img-stored', 'img-held', 'stored', 'stored', 'error', 'error', retry_id]
    return table, claims, expected


def _kind(outcome: ClaimOutcome) -> str:
    if outcome.error is not None:
        return 'error'
    return 'stored' if outcome.holder is None else outcome.holder


@pytest.mark.asyncio
async def test_a_batch_answers_each_claim_as_a_claim_of_its_own_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh, claims, expected = _cases(tmp_path)
    alone: list[ClaimOutcome] = []
    for claim in claims:
        _install(monkeypatch, fresh())
        alone.extend(await content_key.claim_in_session(object(), uuid.uuid4(), [claim]))  # type: ignore[arg-type]
    table = fresh()
    _install(monkeypatch, table)
    together = await content_key.claim_in_session(object(), uuid.uuid4(), claims)  # type: ignore[arg-type]
    assert [_kind(o) for o in alone] == expected
    assert [_kind(o) for o in together] == expected
    assert [type(o.error) for o in together] == [type(o.error) for o in alone]
    assert [type(o.error) for o in together if o.error] == [
        content_key.ContentKeyChangedError,
        repo.ContentKeyConflictError,
    ]
    # the locks of all claims are taken once, in sorted order
    assert table.lock_calls == [sorted(c.imohash for c in claims)]


@pytest.mark.asyncio
async def test_a_key_changed_claim_is_refused_with_the_error_of_the_per_photo_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = b'stored-file' + b'\0' * 64
    other = b'other-file' + b'\0' * 64
    table = Table()
    table.add('H', 'img-stored', _file(tmp_path, 's.bin', stored), len(stored), _sha(stored))
    _install(monkeypatch, table)
    [outcome] = await content_key.claim_in_session(
        object(),  # type: ignore[arg-type]
        uuid.uuid4(),
        [ContentClaim('H', content_image_id('H'), '/p/o.jpg', other)],
    )
    assert isinstance(outcome.error, content_key.ContentKeyChangedError)
    assert outcome.holder is None
    assert len(table.rows) == 1  # nothing was written


@pytest.mark.asyncio
async def test_two_claims_of_one_imohash_are_not_decided_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, Table())
    data = b'x' * 64
    claims = [ContentClaim('H', 'a', '/a', data), ContentClaim('H', 'b', '/b', data)]
    with pytest.raises(ValueError, match='once'):
        await content_key.claim_in_session(object(), uuid.uuid4(), claims)  # type: ignore[arg-type]

"""The exact key is ``imohash`` first and SHA-256 only to confirm; the store decides under an
advisory lock, so concurrent ingests of one content store it once (#358)."""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any

import imohash
import pytest
from prometheus_client import REGISTRY

from src.db.repos import image_contents as repo
from src.db.session import async_session_factory, project_tx
from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import (
    ContentKeyChangedError,
    content_image_id,
    content_imohash,
    content_sha256,
)

from .conftest import make_project


if TYPE_CHECKING:
    import uuid
    from pathlib import Path

pytestmark = [pytest.mark.db]

SIZE = 300_000


def _pair() -> tuple[bytes, bytes]:
    """Two same-size files that differ only outside the three sampled regions (start, middle,
    end: 16 KiB each), so they share an ``imohash`` and not a SHA-256."""
    a = bytes(range(256)) * (SIZE // 256) + bytes(SIZE % 256)
    b = bytearray(a)
    b[50_000] ^= 0xFF
    return a, bytes(b)


def _file(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}

    async def project_id() -> Any:
        return state['project']

    monkeypatch.setattr(content_key, 'bound_project_id', project_id)
    return state


@pytest.fixture
def sha_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The length of every full-byte SHA-256 the key computes."""
    calls: list[int] = []

    def counting(data: bytes) -> str:
        calls.append(len(data))
        return content_sha256(data)

    monkeypatch.setattr(content_key, 'content_sha256', counting)
    return calls


def _collisions() -> float:
    return REGISTRY.get_sample_value('op_exact_key_imohash_collisions_total') or 0.0


async def _rows(project: uuid.UUID, imohash: str) -> list[repo.ContentRow]:
    async with async_session_factory()() as session, project_tx(session, project):
        return await repo.rows_for_imohash(session, project, imohash)


async def _resolve(imohash: str, data: bytes) -> content_key.Verdict:
    return (await content_key.resolve_many_bound([(imohash, data)]))[0]


async def _claim(data: bytes, path: str, image_id: str | None = None) -> str | None:
    key = content_imohash(data)
    return await content_key.claim_bound(key, image_id or content_image_id(key), path, data)


def test_the_pair_shares_an_imohash_and_not_a_sha256() -> None:
    a, b = _pair()
    assert imohash.hashfileobject(io.BytesIO(a)) == imohash.hashfileobject(io.BytesIO(b))
    assert content_sha256(a) != content_sha256(b)


@pytest.mark.asyncio
async def test_a_file_with_no_imohash_match_is_stored_without_a_sha256(
    pg_registry, bound, sha_calls, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, _ = _pair()
    results = await content_key.resolve_many_bound([(content_imohash(a), a)])
    assert results == [content_key.Verdict(None, content_imohash(a), None)]
    assert await _claim(a, _file(tmp_path, 'a', a)) is None

    (row,) = await _rows(project, content_imohash(a))
    assert row.content_sha256 is None
    assert row.size_bytes == len(a)
    assert sha_calls == []


@pytest.mark.asyncio
async def test_a_true_reupload_is_a_duplicate_of_the_first_image(
    pg_registry, bound, sha_calls, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, _ = _pair()
    first = content_image_id(content_imohash(a))
    assert await _claim(a, _file(tmp_path, 'a', a)) is None

    verdict = await _resolve(content_imohash(a), a)
    assert (verdict.holder, verdict.key) == (first, content_imohash(a))
    assert await _claim(a, _file(tmp_path, 'copy', a)) == first

    (row,) = await _rows(project, content_imohash(a))
    assert row.content_sha256 == content_sha256(a)  # filled from the stored file
    assert sha_calls  # the confirmation hashed both


@pytest.mark.asyncio
async def test_an_imohash_collision_keeps_both_images(pg_registry, bound, tmp_path: Path) -> None:
    bound['project'] = project = await make_project()
    a, b = _pair()
    shared = content_imohash(a)
    before = _collisions()
    assert await _claim(a, _file(tmp_path, 'a', a)) is None

    verdict = await _resolve(shared, b)
    assert verdict.holder is None
    assert verdict.key == content_sha256(b)  # the colliding file's id derives from its SHA-256
    b_id = content_image_id(verdict.key)
    assert b_id != content_image_id(shared)
    assert await _claim(b, _file(tmp_path, 'b', b), b_id) is None

    rows = await _rows(project, shared)
    assert {r.image_id: r.content_sha256 for r in rows} == {
        content_image_id(shared): content_sha256(a),
        b_id: content_sha256(b),
    }
    assert _collisions() == before + 1
    # each one is now a duplicate of itself, and neither of the other
    assert (await _resolve(shared, a)).holder == content_image_id(shared)
    assert (await _resolve(shared, b)).holder == b_id


@pytest.mark.asyncio
async def test_a_claim_with_a_stale_id_after_a_collision_is_refused(
    pg_registry, bound, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, b = _pair()
    assert await _claim(a, _file(tmp_path, 'a', a)) is None
    with pytest.raises(ContentKeyChangedError):
        await _claim(b, _file(tmp_path, 'b', b))  # id derived from the imohash: it is A's id
    assert len(await _rows(project, content_imohash(a))) == 1


@pytest.mark.asyncio
async def test_a_stored_file_that_is_gone_cannot_hold_a_duplicate(
    pg_registry, bound, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, _ = _pair()
    stored = _file(tmp_path, 'a', a)
    assert await _claim(a, stored) is None
    (tmp_path / 'a').unlink()

    verdict = await _resolve(content_imohash(a), a)
    assert verdict.holder is None  # unconfirmable: kept as its own image, never lost
    (row,) = await _rows(project, content_imohash(a))
    assert row.content_sha256 is None


@pytest.mark.asyncio
async def test_concurrent_claims_of_one_file_store_it_once(
    pg_registry, bound, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, _ = _pair()
    image_id = content_image_id(content_imohash(a))
    results = await asyncio.gather(
        *[_claim(a, _file(tmp_path, f'copy{i}', a), image_id) for i in range(8)]
    )

    assert results.count(None) == 1
    assert {r for r in results if r is not None} == {image_id}
    assert len(await _rows(project, content_imohash(a))) == 1


@pytest.mark.asyncio
async def test_concurrent_claims_of_a_colliding_pair_store_each_once(
    pg_registry, bound, tmp_path: Path
) -> None:
    bound['project'] = project = await make_project()
    a, b = _pair()
    shared = content_imohash(a)
    a_id, b_id = content_image_id(shared), content_image_id(content_sha256(b))
    assert await _claim(a, _file(tmp_path, 'a', a)) is None
    jobs = [_claim(b, _file(tmp_path, f'b{i}', b), b_id) for i in range(4)]
    jobs += [_claim(a, _file(tmp_path, f'a{i}', a), a_id) for i in range(4)]
    results = await asyncio.gather(*jobs)

    assert results[:4].count(None) == 1
    assert {r for r in results[:4] if r is not None} == {b_id}
    assert set(results[4:]) == {a_id}
    assert {r.image_id for r in await _rows(project, shared)} == {a_id, b_id}


@pytest.mark.asyncio
async def test_release_frees_the_content_for_a_new_image(
    pg_registry, bound, tmp_path: Path
) -> None:
    bound['project'] = await make_project()
    a, _ = _pair()
    image_id = content_image_id(content_imohash(a))
    path = _file(tmp_path, 'a', a)
    assert await _claim(a, path) is None
    assert await content_key.release_bound([image_id]) == 1
    assert await _claim(a, path) is None


@pytest.mark.asyncio
async def test_another_project_does_not_see_the_content(pg_registry, bound, tmp_path: Path) -> None:
    one, other = await make_project(), await make_project()
    a, _ = _pair()
    path = _file(tmp_path, 'a', a)
    bound['project'] = one
    assert await _claim(a, path) is None
    bound['project'] = other
    assert (await _resolve(content_imohash(a), a)).holder is None
    assert await _claim(a, path) is None

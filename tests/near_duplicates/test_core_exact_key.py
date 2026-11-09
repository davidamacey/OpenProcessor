"""The core global index uses the same exact key as the curation ingest: ``imohash`` first, SHA-256
only to confirm, rows with no project, decided under an advisory lock so two concurrent ingests of
one file store it once (#358). The core API keeps no copy of an upload, so each new core image's
SHA-256 is stored with its row (a stored file could not be re-read to confirm it later)."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY

from src.db.repos import image_contents as repo
from src.db.session import async_session_factory, core_tx
from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import CoreEntry, content_imohash, content_sha256

from .conftest import make_project
from .test_exact_key_imohash import _pair


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _entry(data: bytes, image_id: str, path: str = '') -> CoreEntry:
    return CoreEntry(content_imohash(data), image_id, path or f'/up/{image_id}.jpg', data)


async def _rows(imohash: str) -> list[repo.ContentRow]:
    async with async_session_factory()() as session, core_tx(session, 'test'):
        return await repo.rows_for_imohash(session, None, imohash)


def _collisions() -> float:
    return REGISTRY.get_sample_value('op_exact_key_imohash_collisions_total') or 0.0


async def test_a_new_file_is_stored_with_its_sha256_and_no_project(pg_registry) -> None:
    a, _ = _pair()
    claims = await content_key.claim_many_core([_entry(a, 'core-new-1')])

    assert (claims.duplicates, claims.conflicts) == ({}, {})
    (row,) = await _rows(content_imohash(a))
    assert (row.image_id, row.content_sha256, row.size_bytes) == (
        'core-new-1',
        content_sha256(a),
        len(a),
    )


async def test_a_reupload_under_another_id_is_a_duplicate_of_the_first(pg_registry) -> None:
    a, _ = _pair()
    await content_key.claim_many_core([_entry(a, 'core-first')])

    (verdict,) = await content_key.resolve_many_core([(content_imohash(a), a)])
    assert (verdict.holder, verdict.holder_path) == ('core-first', '/up/core-first.jpg')
    claims = await content_key.claim_many_core([_entry(a, 'core-copy')])

    assert claims.duplicates['core-copy'].holder == 'core-first'
    assert len(await _rows(content_imohash(a))) == 1


async def test_an_imohash_collision_keeps_both_images(pg_registry) -> None:
    a, b = _pair()
    assert content_imohash(a) == content_imohash(b)
    before = _collisions()
    await content_key.claim_many_core([_entry(a, 'core-a')])

    claims = await content_key.claim_many_core([_entry(b, 'core-b')])

    assert claims.duplicates == {}
    assert {r.image_id: r.content_sha256 for r in await _rows(content_imohash(a))} == {
        'core-a': content_sha256(a),
        'core-b': content_sha256(b),
    }
    assert _collisions() == before + 1
    assert (await content_key.resolve_many_core([(content_imohash(b), b)]))[0].holder == 'core-b'


async def test_concurrent_requests_with_one_file_store_it_once(pg_registry) -> None:
    a, _ = _pair()
    results = await asyncio.gather(
        *[content_key.claim_many_core([_entry(a, f'core-race-{i}')]) for i in range(8)]
    )

    stored = [i for i, r in enumerate(results) if not r.duplicates]
    assert len(stored) == 1
    winner = f'core-race-{stored[0]}'
    assert {v.holder for r in results for v in r.duplicates.values()} == {winner}
    assert len(await _rows(content_imohash(a))) == 1


async def test_two_equal_files_in_one_batch_store_the_first_only(pg_registry) -> None:
    _, b = _pair()
    claims = await content_key.claim_many_core([_entry(b, 'core-x'), _entry(b, 'core-y')])

    assert len(claims.duplicates) == 1
    assert len(await _rows(content_imohash(b))) == 1


async def test_an_image_id_that_holds_other_bytes_is_refused_not_overwritten(pg_registry) -> None:
    a, b = _pair()
    other = bytes(reversed(a))
    await content_key.claim_many_core([_entry(a, 'core-taken')])

    claims = await content_key.claim_many_core([_entry(other, 'core-taken')])

    assert 'core-taken' in claims.conflicts
    assert [r.image_id for r in await _rows(content_imohash(a))] == ['core-taken']
    del b


async def test_release_forgets_the_content_and_a_project_never_sees_core_rows(pg_registry) -> None:
    a, _ = _pair()
    await content_key.claim_many_core([_entry(a, 'core-gone')])

    from src.db.session import project_tx

    project = await make_project()
    async with async_session_factory()() as session, project_tx(session, project):
        assert await repo.rows_for_imohash(session, project, content_imohash(a)) == []

    assert await content_key.release_core(['core-gone']) == 1
    assert await _rows(content_imohash(a)) == []
    assert (await content_key.resolve_many_core([(content_imohash(a), a)]))[0].holder is None

"""The exact duplicate key: ``imohash`` first, SHA-256 only to confirm, one ``image_contents`` row
per image.

``imohash`` (py-imohash: 16 KiB samples of the start, middle and end plus the size) is cheap and
is the key of every image. It is not a content identity (its README says not to use it for file
verification), so a second file that lands on a stored ``imohash`` is compared by SHA-256:

* no row has the ``imohash``: the file is new, the row is written with no SHA-256 and the bytes
  are never fully hashed;
* a row has it: the new bytes are hashed, each row without a SHA-256 is filled from its stored
  file, and an equal SHA-256 is a duplicate of that image;
* no SHA-256 is equal: an ``imohash`` collision. The file is a new image whose id derives from
  its SHA-256, and ``op_exact_key_imohash_collisions_total`` counts each one stored.

The lookup and the insert are one transaction under ``pg_advisory_xact_lock`` on the project's
``imohash`` (:func:`src.db.repos.image_contents.lock_imohash`), so workers on any process meet
there and the second one is told which image holds the content instead of racing past an index
that has not refreshed (#358). A row is written before the images document it describes and is
removed when the image is deleted.

The core global index (``visual_search_global``) keeps its rows in the same table with no project
(``*_core`` functions, under ``core_tx``): it is the same decision under the same lock, except that
a core image's id is chosen by the caller, so a collision stores the new file under its own id.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import imohash
from prometheus_client import Counter

from src.core.logging import get_logger
from src.db.repos import image_contents as repo
from src.db.session import async_session_factory, core_tx, project_tx
from src.services.config_store.bound_project import bound_project_id
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator, Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)

IMOHASH_COLLISIONS = Counter(
    'op_exact_key_imohash_collisions_total',
    'Files whose imohash matched a stored image but whose SHA-256 did not (both are kept).',
)


class ContentKeyChangedError(ValueError):
    """The bytes resolved to another image id between the lookup and the claim (an ``imohash``
    collision that another writer stored in between): nothing was written, a retry resolves it."""


@dataclass(frozen=True)
class Verdict:
    """What the content key says about one file."""

    holder: str | None
    """The image that already holds these exact bytes, else ``None``."""
    key: str
    """The string a NEW image of these bytes derives its id from: the ``imohash``, or the
    SHA-256 after an ``imohash`` collision."""
    sha256: str | None
    """The SHA-256, when the decision had to compute it."""
    holder_path: str | None = None
    """Where the holder's bytes are stored (the core duplicate answer reports it)."""


def content_imohash(data: bytes) -> str:
    """Lowercase hex ``imohash`` of the bytes (32 characters)."""
    with stage_timer('hash', nbytes=len(data)):
        digest: str = imohash.hashfileobject(io.BytesIO(data)).hex()
        return digest


def content_sha256(data: bytes) -> str:
    """Lowercase hex SHA-256 of the whole byte string."""
    with stage_timer('hash', nbytes=len(data)):
        return hashlib.sha256(data).hexdigest()


def content_image_id(key: str) -> str:
    """``sha256(key)[:32]``: the images-doc id every ingest path derives. It names the content,
    not the path, so the ``create`` of the images doc stays a second guard behind the
    ``image_contents`` row when copies race (#358)."""
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def read_stored_file(path: str) -> bytes | None:
    """The stored image's bytes, or ``None`` when its file cannot be read."""
    if not path:
        return None
    try:
        with stage_timer('file_read'):
            return Path(path).read_bytes()
    except OSError:
        return None


async def _fill_sha256(session: AsyncSession, rows: list[repo.ContentRow]) -> None:
    """Compute and record the SHA-256 of every row that has none, from its stored file.

    A row whose file is gone or no longer has the recorded size cannot be confirmed and stays
    empty. A row whose bytes equal another row's (two images stored before the key existed) keeps
    none either: the SHA-256 is unique, and the first row stands for the content.
    """
    taken = {r.content_sha256 for r in rows if r.content_sha256}
    for row in rows:
        if row.content_sha256 is not None:
            continue
        stored = await asyncio.to_thread(read_stored_file, row.image_path)
        if stored is None or len(stored) != row.size_bytes:
            logger.warning('content_key_stored_file_unverifiable', image_id=row.image_id)
            continue
        sha = await asyncio.to_thread(content_sha256, stored)
        row.content_sha256 = sha
        if sha in taken:
            continue
        taken.add(sha)
        await repo.set_sha256(session, row.id, sha)


async def _decide(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    imohash_hex: str,
    data: bytes,
    known_sha: str | None = None,
) -> Verdict:
    """The verdict for ``data``; takes the project's ``imohash`` lock, so call it in a
    transaction that ends right after the insert it may be followed by. ``known_sha`` spares
    the hash when the caller computed it already."""
    await repo.lock_imohash(session, project_id, imohash_hex)
    rows = await repo.rows_for_imohash(session, project_id, imohash_hex)
    if not rows:
        return Verdict(None, imohash_hex, None)
    sha = known_sha if known_sha is not None else await asyncio.to_thread(content_sha256, data)
    await _fill_sha256(session, rows)
    for row in rows:
        if row.content_sha256 == sha:
            return Verdict(row.image_id, imohash_hex, sha, row.image_path)
    return Verdict(None, sha, sha)


async def resolve_many_bound(entries: Sequence[tuple[str, bytes]]) -> list[Verdict]:
    """One verdict per ``(imohash, bytes)``. One query finds the imohashes the project stores;
    only those files are hashed and compared, each in its own locked transaction."""
    if not entries:
        return []
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        known = await repo.imohashes_with_rows(session, project_id, (h for h, _ in entries))
    verdicts: list[Verdict] = []
    for imohash_hex, data in entries:
        if imohash_hex not in known:
            verdicts.append(Verdict(None, imohash_hex, None))
            continue
        async with async_session_factory()() as session, project_tx(session, project_id):
            verdicts.append(await _decide(session, project_id, imohash_hex, data))
    return verdicts


async def claim_bound(imohash_hex: str, image_id: str, image_path: str, data: bytes) -> str | None:
    """Store the content for ``image_id``; ``None`` when stored now, else the holder's image id.

    The lookup and the insert share one transaction under the ``imohash`` lock. Raises
    :class:`ContentKeyChangedError` when ``image_id`` is not the id these bytes now derive (an
    ``imohash`` collision stored by another writer since the lookup).
    """
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        verdict = await _decide(session, project_id, imohash_hex, data)
        if verdict.holder is not None:
            return verdict.holder
        if content_image_id(verdict.key) != image_id:
            msg = f'image {image_id} no longer names these bytes (imohash collision stored first)'
            raise ContentKeyChangedError(msg)
        holder = await repo.insert_content(
            session,
            project_id,
            imohash=imohash_hex,
            sha256=verdict.sha256,
            image_id=image_id,
            image_path=image_path,
            size_bytes=len(data),
        )
        if holder is None and verdict.key != imohash_hex:
            IMOHASH_COLLISIONS.inc()
            logger.warning('exact_key_imohash_collision', imohash=imohash_hex, image_id=image_id)
        return holder


async def record_copy_bound(imohash_hex: str, image_id: str, image_path: str, size: int) -> None:
    """Record the content of an image a combine copied under its own id (it names the path,
    not the content), so a later ingest of the same file finds it. No bytes are read: the
    SHA-256 stays empty until an ``imohash`` match needs it. A resumed chunk records it again
    as a no-op; the insert takes the same ``imohash`` lock as every claim."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        await repo.lock_imohash(session, project_id, imohash_hex)
        await repo.insert_content(
            session,
            project_id,
            imohash=imohash_hex,
            sha256=None,
            image_id=image_id,
            image_path=image_path,
            size_bytes=size,
        )


async def release_bound(image_ids: Iterable[str]) -> int:
    """Forget the contents of deleted (or never written) images of the bound project."""
    ids = list(image_ids)
    if not ids:
        return 0
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.release_contents(session, project_id, ids)


# --- The core global index: the same key, rows with no project ---------------------------------


@dataclass(frozen=True)
class CoreEntry:
    """One core image to claim: its ``imohash``, the caller's image id, where it is stored (a
    label: the API keeps no copy of an upload), and the bytes."""

    imohash: str
    image_id: str
    image_path: str
    data: bytes


@dataclass(frozen=True)
class CoreClaims:
    """The outcome of one batch of claims; an image in neither map was stored."""

    duplicates: dict[str, Verdict]
    """Image id to the verdict that names the image already holding these exact bytes."""
    conflicts: dict[str, str]
    """Image id to why it was refused: the id already holds other bytes."""


@asynccontextmanager
async def _core_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory()() as session, core_tx(session, 'core exact content key'):
        yield session


async def resolve_many_core(entries: Sequence[tuple[str, bytes]]) -> list[Verdict]:
    """:func:`resolve_many_bound` for the core index: one verdict per ``(imohash, bytes)``."""
    if not entries:
        return []
    async with _core_session() as session:
        known = await repo.imohashes_with_rows(session, None, (h for h, _ in entries))
    verdicts: list[Verdict] = []
    for imohash_hex, data in entries:
        if imohash_hex not in known:
            verdicts.append(Verdict(None, imohash_hex, None))
            continue
        async with _core_session() as session:
            verdicts.append(await _decide(session, None, imohash_hex, data))
    return verdicts


async def claim_many_core(entries: Sequence[CoreEntry]) -> CoreClaims:
    """Store the content of each core image in one transaction, in ``imohash`` order (so two
    concurrent batches never wait on each other in a cycle), under each ``imohash`` lock.

    The core API keeps no copy of an upload, so a stored row could never be confirmed later from
    its file: the SHA-256 of every new core image is computed here, from the bytes in hand, and
    stored with its row. A duplicate within the batch is found the same way: the first of two
    equal files inserts its row and the second sees it.
    """
    if not entries:
        return CoreClaims({}, {})
    shas = await asyncio.to_thread(lambda: [content_sha256(e.data) for e in entries])
    duplicates: dict[str, Verdict] = {}
    conflicts: dict[str, str] = {}
    async with _core_session() as session:
        for i in sorted(range(len(entries)), key=lambda n: entries[n].imohash):
            entry, sha = entries[i], shas[i]
            verdict = await _decide(session, None, entry.imohash, entry.data, known_sha=sha)
            if verdict.holder is not None:
                duplicates[entry.image_id] = verdict
                continue
            try:
                holder = await repo.insert_content(
                    session,
                    None,
                    imohash=entry.imohash,
                    sha256=sha,
                    image_id=entry.image_id,
                    image_path=entry.image_path,
                    size_bytes=len(entry.data),
                )
            except repo.ContentKeyConflictError as exc:
                conflicts[entry.image_id] = str(exc)
                continue
            if holder is not None:  # the image's own row, from a retry: it is stored already
                duplicates[entry.image_id] = Verdict(holder, entry.imohash, sha, entry.image_path)
            elif verdict.key != entry.imohash:
                IMOHASH_COLLISIONS.inc()
                logger.warning(
                    'exact_key_imohash_collision', imohash=entry.imohash, image_id=entry.image_id
                )
    return CoreClaims(duplicates, conflicts)


async def release_core(image_ids: Iterable[str]) -> int:
    """Forget the contents of deleted (or never indexed) core images."""
    ids = list(image_ids)
    if not ids:
        return 0
    async with _core_session() as session:
        return await repo.release_contents(session, None, ids)

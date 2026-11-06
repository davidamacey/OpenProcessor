"""``COPY`` loads of the ledger tables (review fix R-25).

``dataset_import_images``, ``label_operation_members`` and ``ingest_source_files`` can hold a
million rows; row inserts through the ORM would hold the load for hours. psycopg's
``cursor.copy()`` streams them in the caller's transaction on the caller's connection, so a
failed load rolls them back with everything else
(https://www.psycopg.org/psycopg3/docs/basic/copy.html).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql


if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession


async def copy_rows(
    session: AsyncSession, table: str, columns: tuple[str, ...], rows: Iterable[tuple[Any, ...]]
) -> int:
    """Stream ``rows`` into ``table`` with ``COPY ... FROM STDIN``; returns the row count."""
    connection = await session.connection()
    raw = await connection.get_raw_connection()
    driver = raw.driver_connection
    assert driver is not None  # a psycopg connection is checked out
    statement = sql.SQL('COPY {} ({}) FROM STDIN').format(
        sql.Identifier(table), sql.SQL(', ').join(sql.Identifier(c) for c in columns)
    )
    count = 0
    async with driver.cursor() as cursor, cursor.copy(statement) as copy:
        for row in rows:
            await copy.write_row(row)
            count += 1
    return count


IMPORT_IMAGE_COLUMNS = ('import_id', 'image_key', 'image_id', 'item_ids', 'status', 'error')
INGEST_FILE_COLUMNS = (
    'source_id',
    'file_key',
    'status',
    'attempts',
    'error',
    'enqueued_at',
    'updated_at',
)


async def copy_import_images(session: AsyncSession, rows: Iterable[tuple[Any, ...]]) -> int:
    return await copy_rows(session, 'dataset_import_images', IMPORT_IMAGE_COLUMNS, rows)


async def copy_ingest_files(session: AsyncSession, rows: Iterable[tuple[Any, ...]]) -> int:
    return await copy_rows(session, 'ingest_source_files', INGEST_FILE_COLUMNS, rows)

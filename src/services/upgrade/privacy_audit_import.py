"""Import the old ``data/audit/*.jsonl`` purge records into ``privacy_purges``, once.

Each line becomes a row keyed by ``(kind, at)``, so a second run inserts nothing. The files are
left untouched. Only numbers and flags are kept; any other field of a line is dropped.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.db.repos import privacy_purges


if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

FILES = {'faces': 'faces_purge.jsonl', 'ocr': 'ocr_purge.jsonl'}
IMPORTED_BY = 'import:jsonl'


async def import_audit_files(session: AsyncSession, audit_dir: Path) -> int:
    """Rows inserted by this call. The caller owns the transaction."""
    inserted = 0
    for kind, name in FILES.items():
        path = audit_dir / name
        if not path.is_file():
            continue
        added = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            at = dt.datetime.fromisoformat(record.pop('at'))
            counts = {k: v for k, v in record.items() if isinstance(v, int | float | bool)}
            if await privacy_purges.import_row(
                session, kind, counts, at=at, actor=IMPORTED_BY, host=IMPORTED_BY
            ):
                added += 1
        logger.warning('privacy_audit_imported', kind=kind, file=str(path), rows=added)
        inserted += added
    return inserted
